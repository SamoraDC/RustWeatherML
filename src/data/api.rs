//! Open-Meteo API client for fetching weather data.

use anyhow::{Result, Context};
use chrono::NaiveDate;
use reqwest::blocking::Client;
use std::thread;
use std::time::Duration;

use super::schema::{City, WeatherRecord, OpenMeteoResponse};

/// Open-Meteo API client
pub struct OpenMeteoClient {
    client: Client,
    base_url: String,
    archive_url: String,
}

impl Default for OpenMeteoClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenMeteoClient {
    /// Create a new API client
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api.open-meteo.com/v1/forecast".to_string(),
            archive_url: "https://archive-api.open-meteo.com/v1/archive".to_string(),
        }
    }

    /// Hourly variables to request from the API
    fn hourly_params() -> &'static str {
        "temperature_2m,apparent_temperature,dewpoint_2m,\
         precipitation,rain,snowfall,\
         windspeed_10m,windgusts_10m,winddirection_10m,\
         pressure_msl,surface_pressure,cloudcover,visibility,\
         shortwave_radiation,direct_radiation,\
         relativehumidity_2m,weathercode"
    }

    /// Fetch historical weather data for a city and date range
    pub fn fetch_historical(
        &self,
        city: &City,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<WeatherRecord>> {
        let url = format!(
            "{}?latitude={}&longitude={}&start_date={}&end_date={}&hourly={}&timezone={}",
            self.archive_url,
            city.latitude,
            city.longitude,
            start_date.format("%Y-%m-%d"),
            end_date.format("%Y-%m-%d"),
            Self::hourly_params(),
            city.timezone
        );

        let response: OpenMeteoResponse = self.client
            .get(&url)
            .send()
            .context("Failed to send request to Open-Meteo API")?
            .json()
            .context("Failed to parse Open-Meteo response")?;

        self.parse_response(city, response)
    }

    /// Fetch historical data in chunks (to avoid API limits on large date ranges)
    pub fn fetch_historical_chunked(
        &self,
        city: &City,
        start_date: NaiveDate,
        end_date: NaiveDate,
        chunk_days: i64,
        delay_ms: u64,
    ) -> Result<Vec<WeatherRecord>> {
        let mut all_records = Vec::new();
        let mut current_start = start_date;

        while current_start < end_date {
            let current_end = std::cmp::min(
                current_start + chrono::Duration::days(chunk_days),
                end_date
            );

            println!(
                "  Fetching {} from {} to {}...",
                city.name,
                current_start.format("%Y-%m-%d"),
                current_end.format("%Y-%m-%d")
            );

            let records = self.fetch_historical(city, current_start, current_end)?;
            all_records.extend(records);

            current_start = current_end + chrono::Duration::days(1);

            // Rate limiting
            if current_start < end_date && delay_ms > 0 {
                thread::sleep(Duration::from_millis(delay_ms));
            }
        }

        Ok(all_records)
    }

    /// Fetch current forecast for a city
    pub fn fetch_forecast(&self, city: &City, forecast_days: u8) -> Result<Vec<WeatherRecord>> {
        let url = format!(
            "{}?latitude={}&longitude={}&hourly={}&forecast_days={}&timezone={}",
            self.base_url,
            city.latitude,
            city.longitude,
            Self::hourly_params(),
            forecast_days,
            city.timezone
        );

        let response: OpenMeteoResponse = self.client
            .get(&url)
            .send()
            .context("Failed to send request to Open-Meteo API")?
            .json()
            .context("Failed to parse Open-Meteo response")?;

        self.parse_response(city, response)
    }

    /// Parse API response into WeatherRecord structs
    fn parse_response(&self, city: &City, response: OpenMeteoResponse) -> Result<Vec<WeatherRecord>> {
        let hourly = response.hourly;
        let n = hourly.time.len();
        let mut records = Vec::with_capacity(n);

        for i in 0..n {
            let timestamp = chrono::NaiveDateTime::parse_from_str(
                &hourly.time[i],
                "%Y-%m-%dT%H:%M"
            ).context("Failed to parse timestamp")?;

            let record = WeatherRecord {
                city: city.name.clone(),
                country_code: city.country_code.clone(),
                latitude: city.latitude,
                longitude: city.longitude,
                timestamp,
                temperature_2m: hourly.temperature_2m.as_ref().and_then(|v| v.get(i).copied().flatten()),
                apparent_temperature: hourly.apparent_temperature.as_ref().and_then(|v| v.get(i).copied().flatten()),
                dewpoint_2m: hourly.dewpoint_2m.as_ref().and_then(|v| v.get(i).copied().flatten()),
                precipitation: hourly.precipitation.as_ref().and_then(|v| v.get(i).copied().flatten()),
                rain: hourly.rain.as_ref().and_then(|v| v.get(i).copied().flatten()),
                snowfall: hourly.snowfall.as_ref().and_then(|v| v.get(i).copied().flatten()),
                windspeed_10m: hourly.windspeed_10m.as_ref().and_then(|v| v.get(i).copied().flatten()),
                windgusts_10m: hourly.windgusts_10m.as_ref().and_then(|v| v.get(i).copied().flatten()),
                winddirection_10m: hourly.winddirection_10m.as_ref().and_then(|v| v.get(i).copied().flatten()),
                pressure_msl: hourly.pressure_msl.as_ref().and_then(|v| v.get(i).copied().flatten()),
                surface_pressure: hourly.surface_pressure.as_ref().and_then(|v| v.get(i).copied().flatten()),
                cloudcover: hourly.cloudcover.as_ref().and_then(|v| v.get(i).copied().flatten()),
                visibility: hourly.visibility.as_ref().and_then(|v| v.get(i).copied().flatten()),
                shortwave_radiation: hourly.shortwave_radiation.as_ref().and_then(|v| v.get(i).copied().flatten()),
                direct_radiation: hourly.direct_radiation.as_ref().and_then(|v| v.get(i).copied().flatten()),
                relativehumidity_2m: hourly.relativehumidity_2m.as_ref().and_then(|v| v.get(i).copied().flatten()),
                weathercode: hourly.weathercode.as_ref().and_then(|v| v.get(i).copied().flatten()),
            };

            records.push(record);
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_fetch_forecast() {
        let client = OpenMeteoClient::new();
        let city = City::new("São Paulo", "Brazil", "BR", -23.55, -46.63, "America/Sao_Paulo");
        let records = client.fetch_forecast(&city, 1).unwrap();
        assert!(!records.is_empty());
    }
}
