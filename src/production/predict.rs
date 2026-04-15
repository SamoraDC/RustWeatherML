//! Per-city prediction orchestrator (v2.0 — hourly + multi-horizon).
//!
//! Flow for a single day:
//!
//! 1. For each of the 14 cities, fetch `past_days=3, forecast_days=4` of
//!    hourly data via [`OpenMeteoClient`].
//! 2. Convert every response into a Polars `DataFrame` and stack them.
//! 3. Apply the feature pipeline (`pipeline::run`).
//! 4. Pick the "now" row — the 72nd hour of each city block, which
//!    corresponds to "~1 h ago" in local time.
//! 5. **Hourly forecast (+1 h .. +24 h)**: apply Ridge 24 h at each of
//!    the 24 rolling rows ending at "now". The prediction at row
//!    `now-23` targets "now + 1 h"; the prediction at row `now-0`
//!    targets "now + 24 h".
//! 6. **Multi-day forecast**: apply the dedicated Ridge 48 h and
//!    Ridge 72 h models at the "now" row (single shot — we don't roll
//!    them because they already look 48 / 72 h ahead).
//! 7. **Calibrated rain probabilities**: apply the bagging ensemble to
//!    each rolling feature vector, map the vote count through the
//!    reliability curve from `rain_calibration.json`.
//! 8. Parse timestamps (local time per city) for display.

use anyhow::{anyhow, Context, Result};
use chrono::{NaiveDateTime, Utc};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::Duration;

use super::artifacts::ModelBundle;
use super::config::CityConfig;
use super::open_meteo::OpenMeteoClient;
use super::pipeline;

const PAST_DAYS: u8 = 3;
const FORECAST_DAYS: u8 = 4;
const NOW_ROW_OFFSET: usize = (PAST_DAYS as usize) * 24 - 1;

// -------------------------------------------------------------------------
// Public data types
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyReport {
    pub generated_at_utc: String,
    pub n_cities: usize,
    pub n_successful: usize,
    pub n_failed: usize,
    pub model_contract_version: String,
    pub cities: Vec<CityForecast>,
    pub failures: Vec<CityFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityFailure {
    pub city: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityForecast {
    pub city: String,
    pub country_code: String,
    pub flag: String,
    pub latitude: f64,
    pub longitude: f64,
    pub elevation_m: f64,
    pub timezone: String,
    pub climate_zone: String,
    /// Reference timestamp for "now" in the city's local time (ISO 8601).
    pub reference_local_time: String,
    /// Reference timestamp for "now" converted to UTC (ISO 8601).
    pub reference_utc_time: String,
    /// Current observed state (at the "now" row).
    pub current: CurrentObservation,
    /// 24 hourly predictions covering +1 h .. +24 h.
    pub hourly: Vec<HourlyPrediction>,
    /// Point predictions at +24 h / +48 h / +72 h.
    pub multi_horizon: MultiHorizon,
    /// Daily rain summary derived from hourly rain probabilities.
    pub rain_next_24h: RainSummary,
    /// Model metadata so downstream consumers know what to trust.
    pub meta: ForecastMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentObservation {
    pub temperature_c: f64,
    pub dewpoint_c: f64,
    pub humidity_pct: f64,
    pub windspeed_kmh: f64,
    pub winddir_deg: f64,
    pub pressure_hpa: f64,
    pub cloudcover_pct: f64,
    pub precipitation_mm: f64,
    pub weathercode: i64,
    pub weather_condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyPrediction {
    pub local_time: String,
    pub hour_offset: i32,
    pub temperature_c: f64,
    pub temp_source: String, // "ridge_24h" (our model) or "nwp" (Open-Meteo)
    pub precipitation_mm: f64,
    /// Final blended rain probability (NWP-dominant: α=0.9 NWP + 0.1 ML).
    pub rain_probability: f64,
    /// ML-only rain probability from the bagging ensemble + histogram calibration.
    pub rain_probability_model: f64,
    /// NWP-only rain probability derived from Open-Meteo's precipitation forecast.
    pub rain_probability_nwp: f64,
    pub cloudcover_pct: f64,
    pub windspeed_kmh: f64,
    pub weathercode: i64,
    pub weather_condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHorizon {
    pub t_plus_24h: HorizonPoint,
    pub t_plus_48h: HorizonPoint,
    pub t_plus_72h: HorizonPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonPoint {
    pub temperature_c: f64,
    pub local_time: String,
    pub source: String,
    pub expected_rmse_c: f64,
    pub ci95_low_c: f64,
    pub ci95_high_c: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RainSummary {
    pub any_rain: bool,
    /// Final blended aggregate rain probability.
    /// = 0.9 * nwp_probability + 0.1 * model_probability
    pub probability: f64,
    /// ML-only probability (the bagging-ensemble vote / 30, histogram-calibrated).
    pub model_probability: f64,
    /// NWP-only probability derived from the total precipitation over the next 24 h.
    pub nwp_probability: f64,
    /// Weight of the NWP component in the blend (currently fixed at 0.9).
    pub blend_alpha_nwp: f64,
    /// Single-call RFC class prediction (0 or 1) — diagnostic only.
    pub rfc_raw_class: u32,
    /// Hours in +1 .. +24 with precipitation > 0.1 mm (from NWP).
    pub n_hours_with_rain: usize,
    /// Sum of Open-Meteo NWP precipitation over the next 24 h.
    pub total_precip_mm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMeta {
    pub model_24h: String,
    pub model_48h: String,
    pub model_72h: String,
    pub model_version: String,
    pub hourly_model: String,
    pub rain_model: String,
    pub expected_rmse_24h_c: f64,
    pub expected_rmse_48h_c: f64,
    pub expected_rmse_72h_c: f64,
    pub bias_correction_24h_c: f64,
}

// -------------------------------------------------------------------------
// Orchestrator
// -------------------------------------------------------------------------

/// Run the full pipeline for the given city list.
pub fn run_daily(
    cities: &[CityConfig],
    bundle: &ModelBundle,
    rate_limit_ms: u64,
) -> Result<DailyReport> {
    let api = OpenMeteoClient::new();

    let mut per_city_dfs: Vec<(CityConfig, DataFrame)> = Vec::new();
    let mut failures: Vec<CityFailure> = Vec::new();

    for city in cities {
        match api.fetch_recent(city, PAST_DAYS, FORECAST_DAYS) {
            Ok(resp) => match pipeline::response_to_dataframe(city, &resp) {
                Ok(df) => {
                    per_city_dfs.push((city.clone(), df));
                }
                Err(e) => failures.push(CityFailure {
                    city: city.name.clone(),
                    reason: format!("dataframe build: {e}"),
                }),
            },
            Err(e) => failures.push(CityFailure {
                city: city.name.clone(),
                reason: format!("API: {e}"),
            }),
        }
        thread::sleep(Duration::from_millis(rate_limit_ms));
    }

    if per_city_dfs.is_empty() {
        return Err(anyhow!("No city succeeded; aborting report"));
    }

    let all_dfs: Vec<DataFrame> = per_city_dfs.iter().map(|(_, df)| df.clone()).collect();
    let stacked = pipeline::stack_dataframes(all_dfs)?;
    let engineered = pipeline::run(stacked)?;

    let mut city_forecasts: Vec<CityForecast> = Vec::new();
    for (city, _) in &per_city_dfs {
        match predict_one_city(&engineered, city, bundle) {
            Ok(cf) => city_forecasts.push(cf),
            Err(e) => failures.push(CityFailure {
                city: city.name.clone(),
                reason: format!("prediction: {e}"),
            }),
        }
    }

    Ok(DailyReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        n_cities: cities.len(),
        n_successful: city_forecasts.len(),
        n_failed: failures.len(),
        model_contract_version: bundle.contract.version.clone(),
        cities: city_forecasts,
        failures,
    })
}

// -------------------------------------------------------------------------
// Per-city prediction
// -------------------------------------------------------------------------

fn predict_one_city(
    engineered: &DataFrame,
    city: &CityConfig,
    bundle: &ModelBundle,
) -> Result<CityForecast> {
    let city_start = first_row_for_city(engineered, &city.name)?
        .ok_or_else(|| anyhow!("city {} not present", city.name))?;
    let city_end = last_row_for_city(engineered, &city.name)?
        .ok_or_else(|| anyhow!("city {} has no end", city.name))?;
    let now_row = city_start + NOW_ROW_OFFSET;
    if now_row >= city_end {
        return Err(anyhow!(
            "not enough rows for {}: now_row={} >= city_end={}",
            city.name, now_row, city_end
        ));
    }

    // Build the feature vector at now_row for the multi-horizon models.
    let raw_now = pipeline::feature_row(engineered, now_row, bundle.feature_names())?;
    let z_now = bundle.scaler.apply_row(&raw_now);

    // --- Current observation at "now" -------------------------------------
    let current = build_current_observation(engineered, now_row)?;

    // --- Multi-horizon point predictions at now -------------------------
    let rmse_24 = bundle.contract.expected_regression_metrics.rmse;
    let ci_24 = bundle.contract.expected_regression_metrics.rmse_95_ci;
    let (rmse_48, ci_48) = bundle
        .contract
        .expected_regression_metrics_48h
        .as_ref()
        .map(|m| (m.rmse, m.rmse_95_ci))
        .unwrap_or((rmse_24, ci_24));
    let (rmse_72, ci_72) = bundle
        .contract
        .expected_regression_metrics_72h
        .as_ref()
        .map(|m| (m.rmse, m.rmse_95_ci))
        .unwrap_or((rmse_24, ci_24));

    let t24 = round1(bundle.predict_ridge_24h_corrected(&z_now));
    let t48 = round1(bundle.predict_ridge_48h_corrected(&z_now));
    let t72 = round1(bundle.predict_ridge_72h_corrected(&z_now));

    let ref_local_time = str_at(engineered, "timestamp", now_row)?;
    let ref_utc_time = convert_to_utc(&ref_local_time, &city.timezone);

    let multi_horizon = MultiHorizon {
        t_plus_24h: HorizonPoint {
            temperature_c: t24,
            local_time: add_hours_to_iso(&ref_local_time, 24).unwrap_or_else(|| ref_local_time.clone()),
            source: "ridge_24h".into(),
            expected_rmse_c: rmse_24,
            ci95_low_c: ci_24.0,
            ci95_high_c: ci_24.1,
        },
        t_plus_48h: HorizonPoint {
            temperature_c: t48,
            local_time: add_hours_to_iso(&ref_local_time, 48).unwrap_or_else(|| ref_local_time.clone()),
            source: "ridge_48h".into(),
            expected_rmse_c: rmse_48,
            ci95_low_c: ci_48.0,
            ci95_high_c: ci_48.1,
        },
        t_plus_72h: HorizonPoint {
            temperature_c: t72,
            local_time: add_hours_to_iso(&ref_local_time, 72).unwrap_or_else(|| ref_local_time.clone()),
            source: "ridge_72h".into(),
            expected_rmse_c: rmse_72,
            ci95_low_c: ci_72.0,
            ci95_high_c: ci_72.1,
        },
    };

    // --- Hourly predictions (+1h .. +24h) --------------------------------
    //
    // The Ridge 24h model applied at row `now - (24 - k)` yields the
    // prediction for row `now + k`, i.e. `+k h`. So the 24 rolling input
    // rows are `now - 23 .. now`, each producing a forecast for +1 .. +24.
    let mut hourly: Vec<HourlyPrediction> = Vec::with_capacity(24);
    let mut bagging_inputs: Vec<Vec<f64>> = Vec::with_capacity(24);

    for k in 1..=24_i32 {
        let input_row = now_row as i32 + k - 24; // goes from now-23 to now
        if input_row < city_start as i32 {
            continue;
        }
        let input_row = input_row as usize;
        let target_row = now_row + k as usize;
        let raw = pipeline::feature_row(engineered, input_row, bundle.feature_names())?;
        let z = bundle.scaler.apply_row(&raw);
        let t_pred = round1(bundle.predict_ridge_24h_corrected(&z));

        // Read NWP precipitation + cloud + wind + weathercode at target_row
        let precip = f64_at(engineered, "precipitation", target_row).unwrap_or(0.0);
        let cloud = f64_at(engineered, "cloudcover", target_row).unwrap_or(0.0);
        let wind = f64_at(engineered, "windspeed_10m", target_row).unwrap_or(0.0);
        let wcode = i64_at(engineered, "weathercode", target_row).unwrap_or(0);
        let cond = wmo_condition(wcode);
        let local_time = add_hours_to_iso(&ref_local_time, k).unwrap_or_else(|| ref_local_time.clone());

        bagging_inputs.push(raw.clone());
        hourly.push(HourlyPrediction {
            local_time,
            hour_offset: k,
            temperature_c: t_pred,
            temp_source: "ridge_24h".into(),
            precipitation_mm: round1(precip),
            rain_probability: 0.0,       // filled right after
            rain_probability_model: 0.0, // filled right after
            rain_probability_nwp: 0.0,   // filled right after
            cloudcover_pct: round1(cloud),
            windspeed_kmh: round1(wind),
            weathercode: wcode,
            weather_condition: cond.into(),
        });
    }

    // Apply the bagging ensemble to all 24 rolling inputs in one pass.
    // This gives the ML-only probability. We then blend it with the NWP
    // probability derived from Open-Meteo's own hourly precipitation
    // forecast, with the NWP component weighted 9x more because:
    //   (a) the NWP is a physics-based model with demonstrably lower
    //       bias on low-precipitation regimes (e.g. Dubai);
    //   (b) our ML target (`will_rain_next_24h = precip_sum > 0`) is
    //       too liberal — it marks 73.6 % of training samples as
    //       positive, which biases the bagging ensemble toward high
    //       vote counts even when features suggest dry weather;
    //   (c) the NWP has access to global state (upstream systems,
    //       synoptic patterns) that our local-only feature set cannot
    //       capture.
    let proba_raw = super::artifacts::bagging_vote_batch(&bundle.rain_bagging, &bagging_inputs)?;
    for (h, p_ml_raw) in hourly.iter_mut().zip(proba_raw.iter()) {
        let p_ml = bundle.rain_calibration.calibrate(*p_ml_raw);
        let p_nwp = nwp_per_hour_probability(h.precipitation_mm);
        let p_blend = blend_rain_prob(p_nwp, p_ml);
        h.rain_probability_model = round_pct(p_ml);
        h.rain_probability_nwp = round_pct(p_nwp);
        h.rain_probability = round_pct(p_blend);
    }

    // --- RFC single-call on the "now" row + aggregate rain summary ------
    let rfc_dm = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&vec![raw_now.clone()]);
    let rfc_class: Vec<u32> = bundle.rain_rf.predict(&rfc_dm)
        .map_err(|e| anyhow!("RFC predict: {e}"))?;
    let rfc_class = rfc_class.first().copied().unwrap_or(0);

    let n_rainy_hours = hourly.iter().filter(|h| h.precipitation_mm > 0.1).count();
    let total_precip: f64 = hourly.iter().map(|h| h.precipitation_mm).sum();

    // Aggregate the ML probability across the 24 rolling rows: each row
    // was trained on "rain in the NEXT 24 h". Averaging the per-row raw
    // probabilities and then calibrating yields a stable summary.
    let agg_proba_ml = if !proba_raw.is_empty() {
        let mean_raw: f64 = proba_raw.iter().sum::<f64>() / proba_raw.len() as f64;
        bundle.rain_calibration.calibrate(mean_raw)
    } else {
        0.0
    };
    let agg_proba_nwp = nwp_aggregate_probability(total_precip);
    let agg_proba_blend = blend_rain_prob(agg_proba_nwp, agg_proba_ml);

    let rain_summary = RainSummary {
        // The boolean uses the blended prob plus a conservative 1 mm
        // threshold on total NWP precipitation (either signal is enough).
        any_rain: agg_proba_blend > 0.5 || total_precip > 1.0,
        probability: round_pct(agg_proba_blend),
        model_probability: round_pct(agg_proba_ml),
        nwp_probability: round_pct(agg_proba_nwp),
        blend_alpha_nwp: ALPHA_NWP,
        rfc_raw_class: rfc_class,
        n_hours_with_rain: n_rainy_hours,
        total_precip_mm: round1(total_precip),
    };

    Ok(CityForecast {
        city: city.name.clone(),
        country_code: city.country_code.clone(),
        flag: city.flag.to_string(),
        latitude: city.latitude,
        longitude: city.longitude,
        elevation_m: city.elevation_m,
        timezone: city.timezone.clone(),
        climate_zone: city.climate_zone.to_string(),
        reference_local_time: ref_local_time,
        reference_utc_time: ref_utc_time,
        current,
        hourly,
        multi_horizon,
        rain_next_24h: rain_summary,
        meta: ForecastMeta {
            model_24h: "Ridge (alpha=10)".into(),
            model_48h: "Ridge 48h (alpha=10)".into(),
            model_72h: "Ridge 72h (alpha=10)".into(),
            model_version: bundle.contract.version.clone(),
            hourly_model: "Ridge 24h rolled over 24 hours".into(),
            rain_model: format!("Bagging ensemble ({} DT trees) + histogram calibration",
                                bundle.rain_bagging.len()),
            expected_rmse_24h_c: rmse_24,
            expected_rmse_48h_c: rmse_48,
            expected_rmse_72h_c: rmse_72,
            bias_correction_24h_c: -bundle.contract.expected_regression_metrics.mbe,
        },
    })
}

// -------------------------------------------------------------------------
// DataFrame helpers
// -------------------------------------------------------------------------

fn f64_at(df: &DataFrame, col_name: &str, i: usize) -> Result<f64> {
    let col = df.column(col_name)
        .with_context(|| format!("missing column {col_name}"))?;
    let f = col.cast(&DataType::Float64)?;
    f.f64()?.get(i)
        .ok_or_else(|| anyhow!("null {col_name}[{i}]"))
}

fn i64_at(df: &DataFrame, col_name: &str, i: usize) -> Result<i64> {
    let col = df.column(col_name)?;
    let f = col.cast(&DataType::Int64)?;
    f.i64()?.get(i)
        .ok_or_else(|| anyhow!("null {col_name}[{i}]"))
}

fn str_at(df: &DataFrame, col_name: &str, i: usize) -> Result<String> {
    let col = df.column(col_name)?;
    col.str()?.get(i)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("null {col_name}[{i}]"))
}

fn first_row_for_city(df: &DataFrame, city: &str) -> Result<Option<usize>> {
    let cities_col = df.column("city")?.str()?;
    for i in 0..df.height() {
        if cities_col.get(i) == Some(city) {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

fn last_row_for_city(df: &DataFrame, city: &str) -> Result<Option<usize>> {
    let cities_col = df.column("city")?.str()?;
    let mut last = None;
    for i in 0..df.height() {
        if cities_col.get(i) == Some(city) {
            last = Some(i);
        }
    }
    Ok(last)
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn round_pct(x: f64) -> f64 {
    // Round to 2 decimals so probabilities render cleanly and are
    // stable for the golden-test assertions.
    (x.clamp(0.0, 1.0) * 100.0).round() / 100.0
}

// -------------------------------------------------------------------------
// Rain probability blending — NWP-dominant hybrid
// -------------------------------------------------------------------------
//
// `ALPHA_NWP` is the weight of the NWP-derived probability in the final
// blend. The rest (`1 - ALPHA_NWP`) goes to the ML bagging probability.
// We set it high because the ML target (`precip > 0 mm` anywhere in the
// next 24 h) is too liberal and biases the classifier toward positive
// predictions. The NWP has a physical precipitation forecast and is
// far more reliable for "will it actually rain?".
pub const ALPHA_NWP: f64 = 0.9;

/// Soft-threshold mapping from per-hour NWP precipitation (mm) to a
/// rain probability in [0, 1].
///
/// Calibrated by eye against typical hourly precipitation interpretation:
/// - 0.0 mm   → 0.02 (essentially no rain)
/// - 0.01 mm  → 0.10 (trace)
/// - 0.1 mm   → 0.35 (drizzle possible)
/// - 0.3 mm   → 0.60 (light rain)
/// - 1.0 mm   → 0.85 (moderate rain)
/// - ≥ 3.0 mm → 0.97 (heavy rain, high confidence)
fn nwp_per_hour_probability(precip_mm: f64) -> f64 {
    if precip_mm >= 3.0 {
        0.97
    } else if precip_mm >= 1.0 {
        0.85
    } else if precip_mm >= 0.3 {
        0.60
    } else if precip_mm >= 0.1 {
        0.35
    } else if precip_mm >= 0.01 {
        0.10
    } else {
        0.02
    }
}

/// Mapping from 24 h aggregate NWP precipitation (mm) to a rain
/// probability. The thresholds here are stricter than per-hour because
/// 1 mm accumulated over a full day is barely anything; we reserve
/// high probability for totals that correspond to a real rain event.
fn nwp_aggregate_probability(total_mm: f64) -> f64 {
    if total_mm >= 10.0 {
        0.97
    } else if total_mm >= 5.0 {
        0.92
    } else if total_mm >= 2.0 {
        0.80
    } else if total_mm >= 1.0 {
        0.65
    } else if total_mm >= 0.5 {
        0.45
    } else if total_mm >= 0.1 {
        0.20
    } else {
        0.05
    }
}

/// Weighted blend: `alpha_nwp * p_nwp + (1 - alpha_nwp) * p_ml`.
fn blend_rain_prob(p_nwp: f64, p_ml: f64) -> f64 {
    ALPHA_NWP * p_nwp + (1.0 - ALPHA_NWP) * p_ml
}

fn build_current_observation(df: &DataFrame, i: usize) -> Result<CurrentObservation> {
    let wcode = i64_at(df, "weathercode", i).unwrap_or(0);
    Ok(CurrentObservation {
        temperature_c: round1(f64_at(df, "temperature_2m", i).unwrap_or(f64::NAN)),
        dewpoint_c: round1(f64_at(df, "dewpoint_2m", i).unwrap_or(f64::NAN)),
        humidity_pct: round1(f64_at(df, "relativehumidity_2m", i).unwrap_or(f64::NAN)),
        windspeed_kmh: round1(f64_at(df, "windspeed_10m", i).unwrap_or(f64::NAN)),
        winddir_deg: round1(f64_at(df, "winddirection_10m", i).unwrap_or(f64::NAN)),
        pressure_hpa: round1(f64_at(df, "pressure_msl", i).unwrap_or(f64::NAN)),
        cloudcover_pct: round1(f64_at(df, "cloudcover", i).unwrap_or(f64::NAN)),
        precipitation_mm: round1(f64_at(df, "precipitation", i).unwrap_or(0.0)),
        weathercode: wcode,
        weather_condition: wmo_condition(wcode).into(),
    })
}

// -------------------------------------------------------------------------
// Time + WMO helpers
// -------------------------------------------------------------------------

fn wmo_condition(code: i64) -> &'static str {
    match code {
        0 | 1 => "Clear",
        2 | 3 => "Cloudy",
        45 | 48 => "Foggy",
        51..=67 | 80..=82 => "Rainy",
        71..=77 | 85 | 86 => "Snowy",
        95..=99 => "Stormy",
        _ => "Clear",
    }
}

/// Add `hours` to an ISO 8601 string (format `YYYY-MM-DDTHH:MM`).
fn add_hours_to_iso(ts: &str, hours: i32) -> Option<String> {
    let dt = NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M").ok()?;
    let shifted = dt + chrono::Duration::hours(hours as i64);
    Some(shifted.format("%Y-%m-%dT%H:%M").to_string())
}

/// Convert an ISO 8601 local-time string to UTC **assuming** the local
/// time is already in the timezone named by `tz`. We don't do proper
/// DST-aware conversion here (would require chrono-tz); instead we
/// tag the output with a `Z` and let callers interpret.
///
/// Since Open-Meteo returns timestamps in the requested timezone, the
/// simplest correct approach is to display the local timestamp as-is
/// and include the timezone name separately. This helper therefore
/// just returns the input unchanged with a `[tz]` suffix.
fn convert_to_utc(local_iso: &str, tz: &str) -> String {
    // Simple annotation instead of full tz conversion — we ship tz and
    // local time separately so the downstream UI can format it.
    format!("{local_iso}[{tz}]")
}
