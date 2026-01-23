//! Data schema definitions for weather data.

use serde::{Deserialize, Serialize};
use chrono::NaiveDateTime;

/// Represents a city with its geographic coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct City {
    pub name: String,
    pub country: String,
    pub country_code: String,
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: String,
}

impl City {
    pub fn new(name: &str, country: &str, country_code: &str, lat: f64, lon: f64, tz: &str) -> Self {
        Self {
            name: name.to_string(),
            country: country.to_string(),
            country_code: country_code.to_string(),
            latitude: lat,
            longitude: lon,
            timezone: tz.to_string(),
        }
    }
}

/// All 14 cities used in this project
pub fn get_all_cities() -> Vec<City> {
    vec![
        // Brazil
        City::new("São Paulo", "Brazil", "BR", -23.55, -46.63, "America/Sao_Paulo"),
        City::new("Rio de Janeiro", "Brazil", "BR", -22.91, -43.17, "America/Sao_Paulo"),
        City::new("São José dos Campos", "Brazil", "BR", -23.18, -45.88, "America/Sao_Paulo"),
        City::new("Campinas", "Brazil", "BR", -22.91, -47.06, "America/Sao_Paulo"),
        // USA
        City::new("New York", "USA", "US", 40.71, -74.01, "America/New_York"),
        City::new("Los Angeles", "USA", "US", 34.05, -118.24, "America/Los_Angeles"),
        // Europe
        City::new("London", "United Kingdom", "GB", 51.51, -0.13, "Europe/London"),
        City::new("Berlin", "Germany", "DE", 52.52, 13.40, "Europe/Berlin"),
        City::new("Oslo", "Norway", "NO", 59.91, 10.75, "Europe/Oslo"),
        // Asia
        City::new("Tokyo", "Japan", "JP", 35.68, 139.69, "Asia/Tokyo"),
        City::new("Shanghai", "China", "CN", 31.23, 121.47, "Asia/Shanghai"),
        City::new("Chongqing", "China", "CN", 29.56, 106.55, "Asia/Shanghai"),
        City::new("Nanjing", "China", "CN", 32.06, 118.80, "Asia/Shanghai"),
        City::new("Dubai", "UAE", "AE", 25.27, 55.30, "Asia/Dubai"),
    ]
}

/// Weather condition classification based on WMO codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeatherCondition {
    Clear,   // WMO codes: 0, 1
    Cloudy,  // WMO codes: 2, 3
    Foggy,   // WMO codes: 45, 48
    Rainy,   // WMO codes: 51-67, 80-82
    Snowy,   // WMO codes: 71-77, 85-86
    Stormy,  // WMO codes: 95-99
}

impl WeatherCondition {
    /// Convert WMO weather code to WeatherCondition
    pub fn from_wmo_code(code: i32) -> Self {
        match code {
            0 | 1 => WeatherCondition::Clear,
            2 | 3 => WeatherCondition::Cloudy,
            45 | 48 => WeatherCondition::Foggy,
            51..=67 | 80..=82 => WeatherCondition::Rainy,
            71..=77 | 85 | 86 => WeatherCondition::Snowy,
            95..=99 => WeatherCondition::Stormy,
            _ => WeatherCondition::Clear, // Default to clear for unknown codes
        }
    }

    /// Get emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            WeatherCondition::Clear => "☀️",
            WeatherCondition::Cloudy => "⛅",
            WeatherCondition::Foggy => "🌫️",
            WeatherCondition::Rainy => "🌧️",
            WeatherCondition::Snowy => "❄️",
            WeatherCondition::Stormy => "⛈️",
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            WeatherCondition::Clear => "Clear",
            WeatherCondition::Cloudy => "Cloudy",
            WeatherCondition::Foggy => "Foggy",
            WeatherCondition::Rainy => "Rainy",
            WeatherCondition::Snowy => "Snowy",
            WeatherCondition::Stormy => "Stormy",
        }
    }

    /// Convert to numeric label for classification
    pub fn to_label(&self) -> i32 {
        match self {
            WeatherCondition::Clear => 0,
            WeatherCondition::Cloudy => 1,
            WeatherCondition::Foggy => 2,
            WeatherCondition::Rainy => 3,
            WeatherCondition::Snowy => 4,
            WeatherCondition::Stormy => 5,
        }
    }

    /// Convert from numeric label
    pub fn from_label(label: i32) -> Self {
        match label {
            0 => WeatherCondition::Clear,
            1 => WeatherCondition::Cloudy,
            2 => WeatherCondition::Foggy,
            3 => WeatherCondition::Rainy,
            4 => WeatherCondition::Snowy,
            5 => WeatherCondition::Stormy,
            _ => WeatherCondition::Clear,
        }
    }
}

/// A single weather observation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherRecord {
    // Identifiers
    pub city: String,
    pub country_code: String,
    pub latitude: f64,
    pub longitude: f64,
    pub timestamp: NaiveDateTime,

    // Temperature
    pub temperature_2m: Option<f64>,
    pub apparent_temperature: Option<f64>,
    pub dewpoint_2m: Option<f64>,

    // Precipitation
    pub precipitation: Option<f64>,
    pub rain: Option<f64>,
    pub snowfall: Option<f64>,

    // Wind
    pub windspeed_10m: Option<f64>,
    pub windgusts_10m: Option<f64>,
    pub winddirection_10m: Option<f64>,

    // Atmosphere
    pub pressure_msl: Option<f64>,
    pub surface_pressure: Option<f64>,
    pub cloudcover: Option<f64>,
    pub visibility: Option<f64>,

    // Radiation
    pub shortwave_radiation: Option<f64>,
    pub direct_radiation: Option<f64>,

    // Humidity
    pub relativehumidity_2m: Option<f64>,

    // Weather code
    pub weathercode: Option<i32>,
}

impl WeatherRecord {
    /// Check if it will rain (precipitation > 0)
    pub fn will_rain(&self) -> Option<bool> {
        self.precipitation.map(|p| p > 0.0)
    }

    /// Get weather condition from WMO code
    pub fn weather_condition(&self) -> Option<WeatherCondition> {
        self.weathercode.map(WeatherCondition::from_wmo_code)
    }
}

/// Response structure from Open-Meteo API
#[derive(Debug, Deserialize)]
pub struct OpenMeteoResponse {
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: String,
    pub hourly: HourlyData,
}

#[derive(Debug, Deserialize)]
pub struct HourlyData {
    pub time: Vec<String>,
    pub temperature_2m: Option<Vec<Option<f64>>>,
    pub apparent_temperature: Option<Vec<Option<f64>>>,
    pub dewpoint_2m: Option<Vec<Option<f64>>>,
    pub precipitation: Option<Vec<Option<f64>>>,
    pub rain: Option<Vec<Option<f64>>>,
    pub snowfall: Option<Vec<Option<f64>>>,
    pub windspeed_10m: Option<Vec<Option<f64>>>,
    pub windgusts_10m: Option<Vec<Option<f64>>>,
    pub winddirection_10m: Option<Vec<Option<f64>>>,
    pub pressure_msl: Option<Vec<Option<f64>>>,
    pub surface_pressure: Option<Vec<Option<f64>>>,
    pub cloudcover: Option<Vec<Option<f64>>>,
    pub visibility: Option<Vec<Option<f64>>>,
    pub shortwave_radiation: Option<Vec<Option<f64>>>,
    pub direct_radiation: Option<Vec<Option<f64>>>,
    pub relativehumidity_2m: Option<Vec<Option<f64>>>,
    pub weathercode: Option<Vec<Option<i32>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weather_condition_from_wmo() {
        assert_eq!(WeatherCondition::from_wmo_code(0), WeatherCondition::Clear);
        assert_eq!(WeatherCondition::from_wmo_code(3), WeatherCondition::Cloudy);
        assert_eq!(WeatherCondition::from_wmo_code(61), WeatherCondition::Rainy);
        assert_eq!(WeatherCondition::from_wmo_code(71), WeatherCondition::Snowy);
        assert_eq!(WeatherCondition::from_wmo_code(95), WeatherCondition::Stormy);
    }

    #[test]
    fn test_city_creation() {
        let cities = get_all_cities();
        assert_eq!(cities.len(), 14);
        assert_eq!(cities[0].name, "São Paulo");
    }
}
