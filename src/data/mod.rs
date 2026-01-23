//! Data module for weather data collection and management.
//!
//! This module provides:
//! - Open-Meteo API client
//! - Data loading utilities
//! - Data schema definitions

pub mod api;
pub mod loader;
pub mod schema;

pub use api::OpenMeteoClient;
pub use schema::{City, WeatherRecord, WeatherCondition};
