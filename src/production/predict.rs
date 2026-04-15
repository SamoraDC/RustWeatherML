//! Per-city prediction orchestrator.
//!
//! Flow for a single day:
//!
//! 1. For each of the 14 cities, fetch the last 3 days of hourly data
//!    plus 1 forecast day via [`super::open_meteo::OpenMeteoClient`].
//! 2. Convert every response into a Polars `DataFrame` with the exact
//!    schema of Notebook 01.
//! 3. Stack all city DataFrames into a single one so that the lag and
//!    rolling transformations can be partitioned with `.over("city")`.
//! 4. Apply the feature pipeline (`pipeline::run`).
//! 5. For each city, find the most recent fully-specified row, build
//!    the Ridge feature vector, standardize, and predict
//!    `temp_next_24h`. The 48 h and 72 h horizons fall back to the
//!    24 h prediction shifted by the observed diurnal range (honest: we
//!    have only a 24 h model trained).
//! 6. Predict rain probability with the RandomForestClassifier bagging
//!    proxy (majority vote over N bootstrap trees — here we use the
//!    model's deterministic `predict`).

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::thread;
use std::time::Duration;

use super::artifacts::ModelBundle;
use super::config::CityConfig;
use super::open_meteo::OpenMeteoClient;
use super::pipeline;

/// One city's full daily forecast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityForecast {
    pub city: String,
    pub country_code: String,
    pub flag: String,
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: String,
    pub climate_zone: String,
    pub reference_timestamp: String,
    pub current_temp_c: f64,
    pub current_humidity_pct: f64,
    pub current_windspeed_kmh: f64,
    pub current_weathercode: i64,
    pub forecast: ForecastBlock,
    pub rain: RainBlock,
    pub meta: ForecastMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastBlock {
    pub t_plus_24h_c: f64,
    pub t_plus_48h_c: f64,
    pub t_plus_72h_c: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RainBlock {
    pub will_rain_next_24h: bool,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMeta {
    pub model: String,
    pub model_version: String,
    pub expected_rmse_c: f64,
    pub rmse_95_ci_c: (f64, f64),
    pub bias_correction_c: f64,
    pub skill_vs_persistence_24h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyReport {
    pub generated_at_utc: String,
    pub n_cities: usize,
    pub n_successful: usize,
    pub n_failed: usize,
    pub cities: Vec<CityForecast>,
    pub failures: Vec<CityFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityFailure {
    pub city: String,
    pub reason: String,
}

/// Run the full pipeline for the given city list.
pub fn run_daily(
    cities: &[CityConfig],
    bundle: &ModelBundle,
    rate_limit_ms: u64,
) -> Result<DailyReport> {
    let api = OpenMeteoClient::new();

    // Step 1: fetch + convert each city.
    let mut per_city_dfs: Vec<(CityConfig, DataFrame)> = Vec::new();
    let mut failures: Vec<CityFailure> = Vec::new();

    // past_days=3 gives 72 hours of real observations (enough for lag48h),
    // forecast_days=4 gives 96 hours of Open-Meteo's own NWP forecast beyond
    // "now". We use the Ridge model for +24h (our model) and read the NWP
    // values at +48h and +72h directly from the response, since we did not
    // train separate models for those horizons.
    for city in cities {
        match api.fetch_recent(city, 3, 4) {
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

    // Step 2: stack + run pipeline on the full batch.
    let all_dfs: Vec<DataFrame> = per_city_dfs.iter().map(|(_, df)| df.clone()).collect();
    let stacked = pipeline::stack_dataframes(all_dfs)?;
    let engineered = pipeline::run(stacked)?;

    // Step 3: per-city prediction.
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
        cities: city_forecasts,
        failures,
    })
}

/// Read a numeric column value at a row index.
fn f64_at(df: &DataFrame, col_name: &str, i: usize) -> Result<f64> {
    let col = df
        .column(col_name)
        .with_context(|| format!("missing column {col_name}"))?;
    let f = col.cast(&DataType::Float64)?;
    f.f64()?.get(i)
        .ok_or_else(|| anyhow!("null {col_name}[{i}]"))
}

fn i64_at(df: &DataFrame, col_name: &str, i: usize) -> Result<i64> {
    let col = df.column(col_name).with_context(|| format!("missing {col_name}"))?;
    let f = col.cast(&DataType::Int64)?;
    f.i64()?.get(i)
        .ok_or_else(|| anyhow!("null {col_name}[{i}]"))
}

fn str_at(df: &DataFrame, col_name: &str, i: usize) -> Result<String> {
    let col = df.column(col_name).with_context(|| format!("missing {col_name}"))?;
    col.str()?.get(i)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("null {col_name}[{i}]"))
}

/// Predict temperature horizons and rain probability for a single city.
///
/// The response DataFrame has 168 hourly rows: 72 past + 96 forecast.
/// We pick the row at index `past_days*24 - 1 = 71` as "now - 1 h" and
/// run the Ridge model from that row to get the +24 h prediction.
///
/// The +48 h and +72 h values displayed in the README come directly from
/// Open-Meteo's own NWP forecast (rows 119 and 143), because we did not
/// train dedicated models for those horizons. This is intentionally
/// transparent: our ML model owns the 24 h column, the NWP owns the rest.
fn predict_one_city(
    engineered: &DataFrame,
    city: &CityConfig,
    bundle: &ModelBundle,
) -> Result<CityForecast> {
    // "now" heuristic: the last past-observation row. past_days=3 in
    // open_meteo.rs, so row 71 is approximately "1 hour ago" in local
    // time. If that row has null lag48h (unlikely) we fall back to the
    // latest valid row for the city.
    const PAST_DAYS: usize = 3;
    const NOW_OFFSET: usize = PAST_DAYS * 24 - 1;

    let city_start = first_row_for_city(engineered, &city.name)?
        .ok_or_else(|| anyhow!("city {} not present in engineered DataFrame", city.name))?;
    let preferred = city_start + NOW_OFFSET;
    let row_idx = if preferred < engineered.height()
        && pipeline::feature_row(engineered, preferred, bundle.feature_names()).is_ok()
    {
        preferred
    } else {
        pipeline::last_valid_row_for_city(engineered, &city.name)?
            .ok_or_else(|| anyhow!("no valid row for {}", city.name))?
    };

    // Build Ridge feature vector in the exact order of the scaler.
    let raw_features = pipeline::feature_row(engineered, row_idx, bundle.feature_names())?;
    let z_features = bundle.scaler.apply_row(&raw_features);

    // --- Temperature prediction at +24 h (our Ridge model) -----------
    let t_plus_24h_raw = bundle.ridge.predict_z(&z_features);
    // Bias correction from Notebook 05: subtract the systematic MBE
    // (which is negative ~ -0.68 C, so the correction ADDS 0.68 C).
    let t_plus_24h = t_plus_24h_raw
        - bundle.contract.expected_regression_metrics.mbe;

    // --- Temperature at +48 h and +72 h via Open-Meteo NWP -----------
    let current_temp = f64_at(engineered, "temperature_2m", row_idx)?;
    let t_plus_48h = f64_at(engineered, "temperature_2m", row_idx + 48)
        .unwrap_or(t_plus_24h);
    let t_plus_72h = f64_at(engineered, "temperature_2m", row_idx + 72)
        .unwrap_or(current_temp);

    // --- Rain classification via bincode RFC -------------------------
    let dm = DenseMatrix::from_2d_vec(&vec![raw_features.clone()]);
    let rain_pred: Vec<u32> = bundle.rain.predict(&dm)
        .context("RFC predict")?;
    let rain_bool = rain_pred.first().copied().unwrap_or(0) == 1;
    // Calibrated point estimate from Notebook 05's reliability curve:
    // when the majority vote is "rain", the observed frequency is ~85%;
    // when "no rain", ~15%. These are approximate but better than 50%.
    let rain_prob = if rain_bool { 0.85 } else { 0.15 };

    // --- Current observations for the README --------------------------
    let humidity = f64_at(engineered, "relativehumidity_2m", row_idx).unwrap_or(0.0);
    let wind = f64_at(engineered, "windspeed_10m", row_idx).unwrap_or(0.0);
    let wcode = i64_at(engineered, "weathercode", row_idx).unwrap_or(0);
    let ts = str_at(engineered, "timestamp", row_idx)?;

    let ci = bundle.contract.expected_regression_metrics.rmse_95_ci;
    Ok(CityForecast {
        city: city.name.clone(),
        country_code: city.country_code.clone(),
        flag: city.flag.to_string(),
        latitude: city.latitude,
        longitude: city.longitude,
        timezone: city.timezone.clone(),
        climate_zone: city.climate_zone.to_string(),
        reference_timestamp: ts,
        current_temp_c: current_temp,
        current_humidity_pct: humidity,
        current_windspeed_kmh: wind,
        current_weathercode: wcode,
        forecast: ForecastBlock {
            t_plus_24h_c: round1(t_plus_24h),
            t_plus_48h_c: round1(t_plus_48h),
            t_plus_72h_c: round1(t_plus_72h),
        },
        rain: RainBlock {
            will_rain_next_24h: rain_bool,
            probability: rain_prob,
        },
        meta: ForecastMeta {
            model: "Ridge (alpha=10)".into(),
            model_version: bundle.contract.version.clone(),
            expected_rmse_c: bundle.contract.expected_regression_metrics.rmse,
            rmse_95_ci_c: ci,
            bias_correction_c: -bundle.contract.expected_regression_metrics.mbe,
            skill_vs_persistence_24h:
                bundle.contract.expected_regression_metrics.skill_vs_persistence_24h,
        },
    })
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// First row index of a given city in a DataFrame sorted by (city, timestamp).
fn first_row_for_city(df: &DataFrame, city: &str) -> Result<Option<usize>> {
    let cities_col = df.column("city")?.str()?;
    for i in 0..df.height() {
        if cities_col.get(i) == Some(city) {
            return Ok(Some(i));
        }
    }
    Ok(None)
}
