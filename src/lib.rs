//! # RustWeatherML
//!
//! A production-grade machine learning system for weather prediction built entirely in Rust.
//!
//! ## Modules
//!
//! - `data`         - Data loading and API client for Open-Meteo
//! - `preprocessing`- Data cleaning and transformation utilities
//! - `features`     - Feature engineering and selection
//! - `models`       - ML model implementations
//! - `training`     - Training utilities and cross-validation
//! - `evaluation`   - Metrics and visualization
//! - `monitoring`   - Drift detection and performance tracking
//! - `production`   - Runtime consumer of the Notebook 05 artifacts
//!                    (model bundle, feature pipeline, daily predictor).

pub mod data;
pub mod preprocessing;
pub mod features;
pub mod models;
pub mod training;
pub mod evaluation;
pub mod monitoring;

// Production pipeline (Notebook 05 consumer).
pub mod production;

/// Re-export commonly used types
pub use data::schema::{City, WeatherRecord, WeatherCondition};
pub use data::api::OpenMeteoClient;
