//! Production pipeline for daily weather predictions.
//!
//! This module is the runtime counterpart of Notebooks 01-05. It fetches
//! fresh weather data from the Open-Meteo API, applies the exact same
//! feature engineering the notebooks did, loads the Ridge model that was
//! serialized in Notebook 05, and emits both a JSON report and a README
//! update.
//!
//! The equivalence between notebook and production is guaranteed by the
//! integration test `tests/integration_loadmodel.rs`, which replays the
//! 50 golden samples persisted in `models/golden_test.json` within a
//! tolerance of `1e-9`.

pub mod config;
pub mod open_meteo;
pub mod pipeline;
pub mod artifacts;
pub mod predict;
pub mod drift_monitor;
pub mod report;
pub mod dashboard;
