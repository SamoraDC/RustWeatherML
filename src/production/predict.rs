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
    pub rain_probability: f64,
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
    pub probability: f64,         // calibrated aggregate probability
    pub rfc_raw_class: u32,       // single-call RFC prediction (0 or 1)
    pub n_hours_with_rain: usize, // hours in +1..+24 with precip > 0.1 mm
    pub total_precip_mm: f64,     // sum of Open-Meteo NWP precip over next 24 h
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
            rain_probability: 0.0, // filled right after
            cloudcover_pct: round1(cloud),
            windspeed_kmh: round1(wind),
            weathercode: wcode,
            weather_condition: cond.into(),
        });
    }

    // Apply the bagging ensemble to all 24 rolling inputs in one pass.
    let proba_raw = super::artifacts::bagging_vote_batch(&bundle.rain_bagging, &bagging_inputs)?;
    for (h, p) in hourly.iter_mut().zip(proba_raw.iter()) {
        h.rain_probability = (bundle.rain_calibration.calibrate(*p) * 100.0).round() / 100.0;
    }

    // --- RFC single-call on the "now" row + aggregate rain summary ------
    let rfc_dm = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&vec![raw_now.clone()]);
    let rfc_class: Vec<u32> = bundle.rain_rf.predict(&rfc_dm)
        .map_err(|e| anyhow!("RFC predict: {e}"))?;
    let rfc_class = rfc_class.first().copied().unwrap_or(0);

    let n_rainy_hours = hourly.iter().filter(|h| h.precipitation_mm > 0.1).count();
    let total_precip: f64 = hourly.iter().map(|h| h.precipitation_mm).sum();

    // Aggregate probability: take the mean calibrated proba across the 24
    // hourly inputs (each one is "will it rain in the 24 h starting from
    // that row?" — the target RFC was trained on). This is a reasonable
    // proxy for "will it rain at some point in the next 24 h".
    let agg_proba_cal = if !proba_raw.is_empty() {
        let mean_raw: f64 = proba_raw.iter().sum::<f64>() / proba_raw.len() as f64;
        bundle.rain_calibration.calibrate(mean_raw)
    } else {
        0.0
    };
    let rain_summary = RainSummary {
        any_rain: agg_proba_cal > 0.5 || rfc_class == 1,
        probability: (agg_proba_cal * 100.0).round() / 100.0,
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
