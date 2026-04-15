//! Thin Open-Meteo client for production predictions.
//!
//! We use the `/v1/forecast` endpoint with `past_days=3` so every call
//! returns about 96 hourly observations: the last ~72 h (needed to
//! compute lags up to `temp_lag48h`) plus the next ~24 h (a few entries
//! only, we predict temperature ourselves). The archive endpoint has a
//! ~5-day delay and is unsuitable for "today + now".

use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::time::Duration;

use super::config::CityConfig;

const FORECAST_URL: &str = "https://api.open-meteo.com/v1/forecast";

const HOURLY_PARAMS: &str =
    "temperature_2m,apparent_temperature,dewpoint_2m,\
     precipitation,rain,snowfall,\
     windspeed_10m,windgusts_10m,winddirection_10m,\
     pressure_msl,surface_pressure,cloudcover,\
     shortwave_radiation,direct_radiation,\
     relativehumidity_2m,weathercode";

#[derive(Debug, Deserialize)]
pub struct ForecastResponse {
    pub latitude: f64,
    pub longitude: f64,
    #[serde(default)]
    pub timezone: String,
    pub hourly: HourlyBlock,
}

#[derive(Debug, Deserialize)]
pub struct HourlyBlock {
    pub time: Vec<String>,
    #[serde(default)] pub temperature_2m:       Option<Vec<Option<f64>>>,
    #[serde(default)] pub apparent_temperature: Option<Vec<Option<f64>>>,
    #[serde(default)] pub dewpoint_2m:          Option<Vec<Option<f64>>>,
    #[serde(default)] pub precipitation:        Option<Vec<Option<f64>>>,
    #[serde(default)] pub rain:                 Option<Vec<Option<f64>>>,
    #[serde(default)] pub snowfall:             Option<Vec<Option<f64>>>,
    #[serde(default)] pub windspeed_10m:        Option<Vec<Option<f64>>>,
    #[serde(default)] pub windgusts_10m:        Option<Vec<Option<f64>>>,
    #[serde(default)] pub winddirection_10m:    Option<Vec<Option<f64>>>,
    #[serde(default)] pub pressure_msl:         Option<Vec<Option<f64>>>,
    #[serde(default)] pub surface_pressure:     Option<Vec<Option<f64>>>,
    #[serde(default)] pub cloudcover:           Option<Vec<Option<f64>>>,
    #[serde(default)] pub shortwave_radiation:  Option<Vec<Option<f64>>>,
    #[serde(default)] pub direct_radiation:     Option<Vec<Option<f64>>>,
    #[serde(default)] pub relativehumidity_2m:  Option<Vec<Option<f64>>>,
    #[serde(default)] pub weathercode:          Option<Vec<Option<i64>>>,
}

/// Open-Meteo client configured with a generous timeout.
pub struct OpenMeteoClient {
    http: Client,
}

impl Default for OpenMeteoClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenMeteoClient {
    pub fn new() -> Self {
        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(60))
                .user_agent("RustWeatherML/1.0 (+https://github.com/)")
                .build()
                .expect("HTTP client"),
        }
    }

    /// Fetch `past_days` of past observations plus `forecast_days` of
    /// forecast for a single city.
    pub fn fetch_recent(
        &self,
        city: &CityConfig,
        past_days: u8,
        forecast_days: u8,
    ) -> Result<ForecastResponse> {
        let url = format!(
            "{base}?latitude={lat}&longitude={lon}\
             &hourly={vars}\
             &past_days={pd}&forecast_days={fd}\
             &timezone={tz}",
            base = FORECAST_URL,
            lat = city.latitude,
            lon = city.longitude,
            vars = HOURLY_PARAMS,
            pd = past_days,
            fd = forecast_days,
            tz = city.timezone,
        );

        let resp = self
            .http
            .get(&url)
            .send()
            .with_context(|| format!("HTTP request for {}", city.name))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Open-Meteo returned {} for {}",
                resp.status(),
                city.name
            ));
        }

        let parsed: ForecastResponse = resp
            .json()
            .with_context(|| format!("parse JSON for {}", city.name))?;

        if parsed.hourly.time.is_empty() {
            return Err(anyhow!("Empty hourly block for {}", city.name));
        }

        Ok(parsed)
    }
}
