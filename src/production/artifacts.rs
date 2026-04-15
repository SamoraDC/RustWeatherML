//! Loaders for every production artifact emitted by Notebook 05.
//!
//! We deliberately load the Ridge model from its **manual JSON form**
//! (coefficients + intercept), not from the bincode blob. The round-trip
//! verified in Notebook 05 shows the manual path is bit-exact to the
//! bincode path (max diff 4.97e-14 < 1e-9), and it removes any runtime
//! coupling to smartcore's internal type layout.
//!
//! The RandomForestClassifier has no simple coefficient form, so we load
//! it via bincode as shipped by smartcore's `serde` feature.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::config::paths;

/// z-score parameters persisted by Notebook 05.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scaler {
    pub version: String,
    pub n_features: usize,
    pub feature_names: Vec<String>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub formula: String,
}

impl Scaler {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read scaler from {:?}", path.as_ref()))?;
        let s: Scaler = serde_json::from_str(&raw).context("parse scaler.json")?;
        if s.means.len() != s.n_features || s.stds.len() != s.n_features {
            return Err(anyhow!(
                "scaler shape mismatch: means={} stds={} n={}",
                s.means.len(), s.stds.len(), s.n_features
            ));
        }
        Ok(s)
    }

    /// Apply z = (x - mean) / std, feature-wise.
    pub fn apply_row(&self, raw: &[f64]) -> Vec<f64> {
        debug_assert_eq!(raw.len(), self.n_features);
        raw.iter()
            .zip(self.means.iter().zip(self.stds.iter()))
            .map(|(x, (mu, sd))| (x - mu) / sd)
            .collect()
    }
}

/// Manual Ridge form (coefficients + intercept). Notebook 05 guarantees
/// that `y_hat = intercept + sum_j(coefs[j] * z[j])` equals the smartcore
/// prediction within 5e-14.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeManual {
    pub model_type: String,
    pub alpha: f64,
    pub intercept: f64,
    pub coefficients_shape: Vec<usize>,   // [1, n_features]
    pub coefficients_row_major: Vec<f64>,
    pub feature_names: Vec<String>,
    pub feature_order_is_standardized: bool,
    pub formula: String,
}

impl RidgeManual {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read Ridge manual JSON from {:?}", path.as_ref()))?;
        let m: RidgeManual = serde_json::from_str(&raw).context("parse ridge_manual.json")?;
        if m.model_type != "ridge_regression" {
            return Err(anyhow!("unexpected model_type: {}", m.model_type));
        }
        if m.coefficients_row_major.len() != m.feature_names.len() {
            // Smartcore sometimes stores coefs as shape [1, p]; we just need length == p.
            return Err(anyhow!(
                "coef length {} != feature count {}",
                m.coefficients_row_major.len(), m.feature_names.len()
            ));
        }
        Ok(m)
    }

    /// Predict a single row using ALREADY z-scored features.
    pub fn predict_z(&self, z: &[f64]) -> f64 {
        debug_assert_eq!(z.len(), self.coefficients_row_major.len());
        let mut y = self.intercept;
        for (c, x) in self.coefficients_row_major.iter().zip(z.iter()) {
            y += c * x;
        }
        y
    }
}

/// Rain classifier deserialized from `rain_rf_model.bin`.
pub type RainModel = RandomForestClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;

/// Load the RandomForestClassifier via bincode.
pub fn load_rain_model<P: AsRef<Path>>(path: P) -> Result<RainModel> {
    let bytes = fs::read(path.as_ref())
        .with_context(|| format!("read RFC from {:?}", path.as_ref()))?;
    let model: RainModel = bincode::deserialize(&bytes)
        .context("bincode deserialize RandomForestClassifier")?;
    Ok(model)
}

/// Per-(city, hour) temperature climatology fallback.
#[derive(Debug, Clone, Deserialize)]
pub struct Climatology {
    pub version: String,
    pub description: String,
    pub target: String,
    pub global_mean: f64,
    pub hours_per_row: usize,
    pub climatology: HashMap<String, Vec<f64>>,
}

impl Climatology {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read climatology from {:?}", path.as_ref()))?;
        Ok(serde_json::from_str(&raw).context("parse climatology.json")?)
    }

    /// Lookup `(city, hour)` returning the global mean if the key is absent.
    pub fn lookup(&self, city: &str, hour: u32) -> f64 {
        self.climatology
            .get(city)
            .and_then(|hours| hours.get(hour as usize))
            .copied()
            .unwrap_or(self.global_mean)
    }
}

/// Contract produced by Notebook 05 with expected metrics + invariants.
#[derive(Debug, Clone, Deserialize)]
pub struct ProductionContract {
    pub version: String,
    pub winner_model: String,
    pub n_features: usize,
    pub feature_names: Vec<String>,
    pub tolerance_abs_regression: f64,
    pub tolerance_abs_golden_path: f64,
    pub expected_regression_metrics: ContractRegMetrics,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContractRegMetrics {
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub mbe: f64,
    pub rmse_95_ci: (f64, f64),
    pub skill_vs_persistence_24h: f64,
}

impl ProductionContract {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read contract from {:?}", path.as_ref()))?;
        Ok(serde_json::from_str(&raw).context("parse production_contract.json")?)
    }
}

/// Convenience: load *every* artifact required by the daily binary. Pass
/// the project-root relative path prefix (usually `.`).
pub struct ModelBundle {
    pub scaler: Scaler,
    pub ridge: RidgeManual,
    pub rain: RainModel,
    pub climatology: Climatology,
    pub contract: ProductionContract,
}

impl ModelBundle {
    pub fn load_from_root<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref();
        let scaler      = Scaler::load(root.join(paths::SCALER))?;
        let ridge       = RidgeManual::load(root.join(paths::RIDGE_MANUAL))?;
        let rain        = load_rain_model(root.join(paths::RAIN_RF_BIN))?;
        let climatology = Climatology::load(root.join(paths::CLIMATOLOGY))?;
        let contract    = ProductionContract::load(root.join(paths::CONTRACT))?;

        // Sanity: feature ordering must be identical across scaler,
        // ridge, and contract. If this breaks, Ridge coefficients no
        // longer align with the input vector.
        if scaler.feature_names != ridge.feature_names {
            return Err(anyhow!(
                "feature_names mismatch between scaler and Ridge: {} vs {}",
                scaler.feature_names.len(), ridge.feature_names.len()
            ));
        }
        if contract.feature_names != scaler.feature_names {
            return Err(anyhow!(
                "feature_names mismatch between contract and scaler"
            ));
        }

        Ok(Self { scaler, ridge, rain, climatology, contract })
    }

    pub fn feature_count(&self) -> usize {
        self.scaler.n_features
    }

    pub fn feature_names(&self) -> &[String] {
        &self.scaler.feature_names
    }
}
