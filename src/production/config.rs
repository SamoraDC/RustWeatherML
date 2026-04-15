//! Project-wide runtime configuration.
//!
//! The city list and Köppen climate-zone labels **must** match Notebook 01
//! and Notebook 02 exactly, because the climate-zone one-hot features are
//! part of the trained Ridge model. Any drift here fails the golden test.

use serde::{Deserialize, Serialize};

/// A city with the metadata required to (i) query the Open-Meteo API and
/// (ii) reproduce the climate-zone one-hot features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityConfig {
    pub name: String,
    pub country: String,
    pub country_code: String,
    pub latitude: f64,
    pub longitude: f64,
    pub elevation_m: f64,
    pub timezone: String,
    /// Köppen first letter + subtype (e.g. "Cfa", "BWh").
    pub climate_zone: &'static str,
    /// Flag emoji used in the README table.
    pub flag: &'static str,
}


/// The 14 cities in the order they appear in the README. The ORDER is
/// display-only — the model does not depend on it.
pub fn cities() -> Vec<CityConfig> {
    vec![
        CityConfig {
            name: "Sao Paulo".into(),
            country: "Brazil".into(),
            country_code: "BR".into(),
            latitude: -23.55, longitude: -46.63, elevation_m: 760.0,
            timezone: "America/Sao_Paulo".into(),
            climate_zone: "Cfa", flag: "🇧🇷",
        },
        CityConfig {
            name: "Rio de Janeiro".into(),
            country: "Brazil".into(),
            country_code: "BR".into(),
            latitude: -22.91, longitude: -43.17, elevation_m: 5.0,
            timezone: "America/Sao_Paulo".into(),
            climate_zone: "Aw", flag: "🇧🇷",
        },
        CityConfig {
            name: "Sao Jose dos Campos".into(),
            country: "Brazil".into(),
            country_code: "BR".into(),
            latitude: -23.18, longitude: -45.88, elevation_m: 600.0,
            timezone: "America/Sao_Paulo".into(),
            climate_zone: "Cfa", flag: "🇧🇷",
        },
        CityConfig {
            name: "Campinas".into(),
            country: "Brazil".into(),
            country_code: "BR".into(),
            latitude: -22.91, longitude: -47.06, elevation_m: 680.0,
            timezone: "America/Sao_Paulo".into(),
            climate_zone: "Cfa", flag: "🇧🇷",
        },
        CityConfig {
            name: "New York".into(),
            country: "USA".into(),
            country_code: "US".into(),
            latitude: 40.71, longitude: -74.01, elevation_m: 10.0,
            timezone: "America/New_York".into(),
            climate_zone: "Dfa", flag: "🇺🇸",
        },
        CityConfig {
            name: "Los Angeles".into(),
            country: "USA".into(),
            country_code: "US".into(),
            latitude: 34.05, longitude: -118.24, elevation_m: 89.0,
            timezone: "America/Los_Angeles".into(),
            climate_zone: "Csb", flag: "🇺🇸",
        },
        CityConfig {
            name: "London".into(),
            country: "United Kingdom".into(),
            country_code: "GB".into(),
            latitude: 51.51, longitude: -0.13, elevation_m: 35.0,
            timezone: "Europe/London".into(),
            climate_zone: "Cfb", flag: "🇬🇧",
        },
        CityConfig {
            name: "Berlin".into(),
            country: "Germany".into(),
            country_code: "DE".into(),
            latitude: 52.52, longitude: 13.40, elevation_m: 34.0,
            timezone: "Europe/Berlin".into(),
            climate_zone: "Cfb", flag: "🇩🇪",
        },
        CityConfig {
            name: "Oslo".into(),
            country: "Norway".into(),
            country_code: "NO".into(),
            latitude: 59.91, longitude: 10.75, elevation_m: 23.0,
            timezone: "Europe/Oslo".into(),
            climate_zone: "Dfb", flag: "🇳🇴",
        },
        CityConfig {
            name: "Tokyo".into(),
            country: "Japan".into(),
            country_code: "JP".into(),
            latitude: 35.68, longitude: 139.69, elevation_m: 40.0,
            timezone: "Asia/Tokyo".into(),
            climate_zone: "Cfa", flag: "🇯🇵",
        },
        CityConfig {
            name: "Shanghai".into(),
            country: "China".into(),
            country_code: "CN".into(),
            latitude: 31.23, longitude: 121.47, elevation_m: 4.0,
            timezone: "Asia/Shanghai".into(),
            climate_zone: "Cfa", flag: "🇨🇳",
        },
        CityConfig {
            name: "Chongqing".into(),
            country: "China".into(),
            country_code: "CN".into(),
            latitude: 29.56, longitude: 106.55, elevation_m: 244.0,
            timezone: "Asia/Shanghai".into(),
            climate_zone: "Cwa", flag: "🇨🇳",
        },
        CityConfig {
            name: "Nanjing".into(),
            country: "China".into(),
            country_code: "CN".into(),
            latitude: 32.06, longitude: 118.80, elevation_m: 20.0,
            timezone: "Asia/Shanghai".into(),
            climate_zone: "Cfa", flag: "🇨🇳",
        },
        CityConfig {
            name: "Dubai".into(),
            country: "UAE".into(),
            country_code: "AE".into(),
            latitude: 25.27, longitude: 55.30, elevation_m: 5.0,
            timezone: "Asia/Dubai".into(),
            climate_zone: "BWh", flag: "🇦🇪",
        },
    ]
}

/// The eight Köppen climate zones present in the training data. The ORDER
/// matters — it is the order used to build the `climate_*` one-hot features
/// in Notebook 02, which is baked into the Ridge coefficients.
pub const CLIMATE_ZONES: [&str; 8] =
    ["Aw", "BWh", "Cfa", "Cfb", "Csb", "Cwa", "Dfa", "Dfb"];

/// Standard paths (relative to the project root) for production artifacts.
pub mod paths {
    pub const FEATURES_SCHEMA: &str     = "models/features_schema.json";
    pub const SCALER: &str              = "models/scaler.json";
    pub const RIDGE_BIN: &str           = "models/ridge_model.bin";
    pub const RIDGE_MANUAL: &str        = "models/ridge_manual.json";
    pub const RIDGE_48H_BIN: &str       = "models/ridge_48h_model.bin";
    pub const RIDGE_48H_MANUAL: &str    = "models/ridge_48h_manual.json";
    pub const RIDGE_72H_BIN: &str       = "models/ridge_72h_model.bin";
    pub const RIDGE_72H_MANUAL: &str    = "models/ridge_72h_manual.json";
    pub const RAIN_RF_BIN: &str         = "models/rain_rf_model.bin";
    pub const RAIN_BAGGING_BIN: &str    = "models/rain_bagging_ensemble.bin";
    pub const RAIN_CALIBRATION: &str    = "models/rain_calibration.json";
    pub const CLIMATOLOGY: &str         = "models/climatology.json";
    pub const CONTRACT: &str            = "models/production_contract.json";
    pub const GOLDEN_TEST: &str         = "models/golden_test.json";
    pub const DRIFT_REFERENCE: &str     = "models/drift_reference.json";
    pub const PREDICTIONS_DIR: &str     = "data/predictions";
    pub const MONITORING_HISTORY_DIR: &str = "data/monitoring_history";
    pub const DASHBOARD_DIR: &str       = "docs/dashboard";
    pub const README: &str              = "README.md";
}
