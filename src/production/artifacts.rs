//! Loaders for every production artifact emitted by Notebook 05 v2.0.
//!
//! Artifacts loaded by this module:
//!
//! | File | Loader |
//! |---|---|
//! | `models/ridge_manual.json`       | [`RidgeManual`] (24 h) |
//! | `models/ridge_48h_manual.json`   | [`RidgeManual`] (48 h) |
//! | `models/ridge_72h_manual.json`   | [`RidgeManual`] (72 h) |
//! | `models/rain_rf_model.bin`       | [`RainModel`] (RandomForestClassifier) |
//! | `models/rain_bagging_ensemble.bin` | [`RainBaggingEnsemble`] (30 DT trees) |
//! | `models/rain_calibration.json`   | [`RainCalibration`] (10-bin reliability curve) |
//! | `models/scaler.json`             | [`Scaler`] (means, stds, feature_names) |
//! | `models/climatology.json`        | [`Climatology`] (per-(city, hour) mean fallback) |
//! | `models/production_contract.json`| [`ProductionContract`] (expected metrics + hashes) |
//! | `models/drift_reference.json`    | [`DriftReference`] (per-feature quantile bins + probs) |
//!
//! Ridge models are loaded from the **manual JSON form** (coefficients +
//! intercept). Notebook 05 verifies the manual path is bit-exact to the
//! bincode path (< 5e-14), and JSON is robust against smartcore version
//! drift. The RFC and the bagging ensemble use bincode because their
//! internal tree structure is impractical to serialize by hand.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::config::paths;

// -------------------------------------------------------------------------
// Scaler
// -------------------------------------------------------------------------

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

// -------------------------------------------------------------------------
// Ridge (manual)
// -------------------------------------------------------------------------

/// Manual Ridge form (coefficients + intercept). Notebook 05 guarantees
/// that `y_hat = intercept + sum_j(coefs[j] * z[j])` equals the smartcore
/// prediction within 5e-14.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeManual {
    pub model_type: String,
    #[serde(default)]
    pub horizon: String,
    pub alpha: f64,
    pub intercept: f64,
    pub coefficients_shape: Vec<usize>,
    pub coefficients_row_major: Vec<f64>,
    pub feature_names: Vec<String>,
    pub feature_order_is_standardized: bool,
    pub formula: String,
}

impl RidgeManual {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read Ridge manual JSON from {:?}", path.as_ref()))?;
        let m: RidgeManual = serde_json::from_str(&raw).context("parse Ridge JSON")?;
        if m.model_type != "ridge_regression" {
            return Err(anyhow!("unexpected model_type: {}", m.model_type));
        }
        if m.coefficients_row_major.len() != m.feature_names.len() {
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

    /// Predict a batch of rows.
    pub fn predict_batch_z(&self, zs: &[Vec<f64>]) -> Vec<f64> {
        zs.iter().map(|z| self.predict_z(z)).collect()
    }
}

// -------------------------------------------------------------------------
// Rain RFC + bagging ensemble + calibration
// -------------------------------------------------------------------------

/// Rain classifier deserialized from `rain_rf_model.bin`.
pub type RainModel = RandomForestClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;

/// Single DT classifier from the bagging ensemble.
pub type RainBagTree = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;

/// 30-tree bagging ensemble deserialized from `rain_bagging_ensemble.bin`.
pub type RainBaggingEnsemble = Vec<RainBagTree>;

/// Load the RandomForestClassifier via bincode.
pub fn load_rain_model<P: AsRef<Path>>(path: P) -> Result<RainModel> {
    let bytes = fs::read(path.as_ref())
        .with_context(|| format!("read RFC from {:?}", path.as_ref()))?;
    let model: RainModel = bincode::deserialize(&bytes)
        .context("bincode deserialize RandomForestClassifier")?;
    Ok(model)
}

/// Load the bagging ensemble via bincode.
pub fn load_rain_bagging_ensemble<P: AsRef<Path>>(path: P) -> Result<RainBaggingEnsemble> {
    let bytes = fs::read(path.as_ref())
        .with_context(|| format!("read bagging ensemble from {:?}", path.as_ref()))?;
    let ens: RainBaggingEnsemble = bincode::deserialize(&bytes)
        .context("bincode deserialize RainBaggingEnsemble")?;
    if ens.is_empty() {
        return Err(anyhow!("empty bagging ensemble"));
    }
    Ok(ens)
}

/// Apply every tree in the ensemble to a single feature vector and
/// return `(vote_count, probability)`.
pub fn bagging_vote(
    ensemble: &RainBaggingEnsemble,
    raw_features: &[f64],
) -> Result<(usize, f64)> {
    let dm = DenseMatrix::from_2d_vec(&vec![raw_features.to_vec()]);
    let mut votes = 0usize;
    for tree in ensemble {
        let p: Vec<u32> = tree.predict(&dm)
            .map_err(|e| anyhow!("bagging predict: {e}"))?;
        if p.first().copied().unwrap_or(0) == 1 {
            votes += 1;
        }
    }
    let p = votes as f64 / ensemble.len() as f64;
    Ok((votes, p))
}

/// Apply the ensemble to a batch of feature vectors.
pub fn bagging_vote_batch(
    ensemble: &RainBaggingEnsemble,
    raw_features_batch: &[Vec<f64>],
) -> Result<Vec<f64>> {
    let rows: Vec<Vec<f64>> = raw_features_batch.iter().cloned().collect();
    let dm = DenseMatrix::from_2d_vec(&rows);
    let mut vote_sum = vec![0usize; rows.len()];
    for tree in ensemble {
        let p: Vec<u32> = tree.predict(&dm)
            .map_err(|e| anyhow!("bagging batch predict: {e}"))?;
        for (i, v) in p.iter().enumerate() {
            if *v == 1 {
                vote_sum[i] += 1;
            }
        }
    }
    let n = ensemble.len() as f64;
    Ok(vote_sum.into_iter().map(|v| v as f64 / n).collect())
}

/// Rain calibration map persisted by Notebook 05.
#[derive(Debug, Clone, Deserialize)]
pub struct RainCalibration {
    pub version: String,
    pub n_trees: usize,
    pub n_bins: usize,
    pub reliability_bins: Vec<RainBin>,
    pub brier_score: f64,
    pub brier_climatology: f64,
    pub brier_skill_score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RainBin {
    pub bin_lo: f64,
    pub bin_hi: f64,
    pub count: usize,
    pub pred_mean: f64,
    pub obs_freq: f64,
}

impl RainCalibration {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read rain_calibration from {:?}", path.as_ref()))?;
        Ok(serde_json::from_str(&raw).context("parse rain_calibration.json")?)
    }

    /// Map a raw vote probability `p` in [0, 1] to the observed
    /// frequency of its bin. If the bin has too few samples (< 5), we
    /// fall back to the raw value.
    pub fn calibrate(&self, raw: f64) -> f64 {
        let clamped = raw.clamp(0.0, 1.0);
        let idx = ((clamped * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);
        let bin = &self.reliability_bins[idx];
        if bin.count < 5 {
            clamped
        } else {
            bin.obs_freq.clamp(0.0, 1.0)
        }
    }
}

// -------------------------------------------------------------------------
// Climatology fallback
// -------------------------------------------------------------------------

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

    pub fn lookup(&self, city: &str, hour: u32) -> f64 {
        self.climatology
            .get(city)
            .and_then(|hours| hours.get(hour as usize))
            .copied()
            .unwrap_or(self.global_mean)
    }
}

// -------------------------------------------------------------------------
// Production contract
// -------------------------------------------------------------------------

/// Contract produced by Notebook 05 with expected metrics + invariants.
#[derive(Debug, Clone, Deserialize)]
pub struct ProductionContract {
    pub version: String,
    pub winner_model: String,
    pub n_features: usize,
    pub feature_names: Vec<String>,
    pub tolerance_abs_regression: f64,
    pub tolerance_abs_golden_path: f64,
    #[serde(default)]
    pub tolerance_abs_probability: f64,
    pub expected_regression_metrics: ContractRegMetrics,
    #[serde(default)]
    pub expected_regression_metrics_48h: Option<ContractRegMetrics>,
    #[serde(default)]
    pub expected_regression_metrics_72h: Option<ContractRegMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContractRegMetrics {
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub mbe: f64,
    pub rmse_95_ci: (f64, f64),
    #[serde(default)]
    pub skill_vs_persistence_24h: f64,
}

impl ProductionContract {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read contract from {:?}", path.as_ref()))?;
        Ok(serde_json::from_str(&raw).context("parse production_contract.json")?)
    }
}

// -------------------------------------------------------------------------
// Drift reference (for production drift monitoring)
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DriftReference {
    pub version: String,
    pub reference_dataset: String,
    pub n_features: usize,
    pub features: HashMap<String, DriftFeatureRef>,
    pub psi_thresholds: PsiThresholds,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DriftFeatureRef {
    pub n_bins: usize,
    pub edges: Vec<f64>,
    pub ref_counts: Vec<usize>,
    pub ref_probs: Vec<f64>,
    pub ref_mean: f64,
    pub ref_std: f64,
    pub ref_n: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PsiThresholds {
    pub moderate: f64,
    pub severe: f64,
}

impl DriftReference {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read drift reference from {:?}", path.as_ref()))?;
        Ok(serde_json::from_str(&raw).context("parse drift_reference.json")?)
    }
}

// -------------------------------------------------------------------------
// Unified bundle
// -------------------------------------------------------------------------

pub struct ModelBundle {
    pub scaler: Scaler,
    pub ridge_24h: RidgeManual,
    pub ridge_48h: RidgeManual,
    pub ridge_72h: RidgeManual,
    pub rain_rf: RainModel,
    pub rain_bagging: RainBaggingEnsemble,
    pub rain_calibration: RainCalibration,
    pub climatology: Climatology,
    pub contract: ProductionContract,
    pub drift_reference: DriftReference,
}

impl ModelBundle {
    pub fn load_from_root<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref();
        let scaler           = Scaler::load(root.join(paths::SCALER))?;
        let ridge_24h        = RidgeManual::load(root.join(paths::RIDGE_MANUAL))?;
        let ridge_48h        = RidgeManual::load(root.join(paths::RIDGE_48H_MANUAL))?;
        let ridge_72h        = RidgeManual::load(root.join(paths::RIDGE_72H_MANUAL))?;
        let rain_rf          = load_rain_model(root.join(paths::RAIN_RF_BIN))?;
        let rain_bagging     = load_rain_bagging_ensemble(root.join(paths::RAIN_BAGGING_BIN))?;
        let rain_calibration = RainCalibration::load(root.join(paths::RAIN_CALIBRATION))?;
        let climatology      = Climatology::load(root.join(paths::CLIMATOLOGY))?;
        let contract         = ProductionContract::load(root.join(paths::CONTRACT))?;
        let drift_reference  = DriftReference::load(root.join(paths::DRIFT_REFERENCE))?;

        // Sanity: feature ordering must be identical across every artifact.
        let canonical = scaler.feature_names.clone();
        for (label, names) in [
            ("ridge_24h", &ridge_24h.feature_names),
            ("ridge_48h", &ridge_48h.feature_names),
            ("ridge_72h", &ridge_72h.feature_names),
        ] {
            if names != &canonical {
                return Err(anyhow!(
                    "feature_names mismatch: scaler vs {}: {} != {}",
                    label, canonical.len(), names.len()
                ));
            }
        }
        if contract.feature_names != canonical {
            return Err(anyhow!("feature_names mismatch between contract and scaler"));
        }

        Ok(Self {
            scaler, ridge_24h, ridge_48h, ridge_72h, rain_rf,
            rain_bagging, rain_calibration, climatology, contract, drift_reference,
        })
    }

    pub fn feature_count(&self) -> usize {
        self.scaler.n_features
    }

    pub fn feature_names(&self) -> &[String] {
        &self.scaler.feature_names
    }

    /// Apply Ridge 24 h with bias correction.
    pub fn predict_ridge_24h_corrected(&self, z: &[f64]) -> f64 {
        self.ridge_24h.predict_z(z) - self.contract.expected_regression_metrics.mbe
    }

    /// Apply Ridge 48 h with the 48 h bias correction (falls back to 0 if missing).
    pub fn predict_ridge_48h_corrected(&self, z: &[f64]) -> f64 {
        let bias = self
            .contract
            .expected_regression_metrics_48h
            .as_ref()
            .map(|m| m.mbe)
            .unwrap_or(0.0);
        self.ridge_48h.predict_z(z) - bias
    }

    /// Apply Ridge 72 h with the 72 h bias correction.
    pub fn predict_ridge_72h_corrected(&self, z: &[f64]) -> f64 {
        let bias = self
            .contract
            .expected_regression_metrics_72h
            .as_ref()
            .map(|m| m.mbe)
            .unwrap_or(0.0);
        self.ridge_72h.predict_z(z) - bias
    }
}
