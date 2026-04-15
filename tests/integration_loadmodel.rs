//! Equivalence test: replay the 50 golden samples persisted by
//! Notebook 05 v2.0 and assert the production Rust code reproduces
//! every prediction within the tolerances declared in
//! `production_contract.json`.
//!
//! Verifies:
//! - Scaler (raw → z-score) bit-exact
//! - Ridge 24 h, 48 h, 72 h all bit-exact (< 1e-9)
//! - RandomForestClassifier class match (0 mismatches)
//! - Rain bagging ensemble probability bit-exact
//! - Rain calibration map bit-exact
//! - Production contract validates

use anyhow::Result;
use rust_weather_ml::production::artifacts::{
    bagging_vote_batch, load_rain_bagging_ensemble, load_rain_model,
    ProductionContract, RainCalibration, RidgeManual, Scaler,
};
use serde::Deserialize;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct GoldenFile {
    #[allow(dead_code)]
    version: String,
    tolerance_abs_regression: f64,
    #[allow(dead_code)]
    tolerance_abs_classification: f64,
    #[serde(default)]
    tolerance_abs_probability: f64,
    n_samples: usize,
    feature_names: Vec<String>,
    samples: Vec<GoldenSample>,
}

#[derive(Debug, Deserialize)]
struct GoldenSample {
    #[allow(dead_code)] row_index_in_test: usize,
    #[allow(dead_code)] city: String,
    #[allow(dead_code)] timestamp: String,
    raw_features: Vec<f64>,
    standardized_features: Vec<f64>,
    #[allow(dead_code)] y_true_temp_next_24h: f64,
    #[allow(dead_code)] y_true_temp_next_48h: Option<f64>,
    #[allow(dead_code)] y_true_temp_next_72h: Option<f64>,
    #[allow(dead_code)] y_true_will_rain: u32,
    pred_ridge: f64,
    #[serde(default)] pred_ridge_48h: f64,
    #[serde(default)] pred_ridge_72h: f64,
    #[allow(dead_code)] pred_lasso: f64,
    #[allow(dead_code)] pred_rf: f64,
    #[allow(dead_code)] pred_gb: f64,
    pred_rfc_class: u32,
    #[serde(default)] pred_rain_proba_raw: f64,
    #[serde(default)] pred_rain_proba_cal: f64,
    #[allow(dead_code)] pred_log_class: u32,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_golden(path: &Path) -> Result<GoldenFile> {
    let raw = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

#[test]
fn golden_scaler_matches_raw_to_standardized() -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let scaler = Scaler::load(root.join("models/scaler.json"))?;

    assert_eq!(golden.feature_names, scaler.feature_names);
    assert_eq!(golden.n_samples, golden.samples.len());

    let tol = 1e-12;
    let mut max_diff = 0.0_f64;
    for (k, s) in golden.samples.iter().enumerate() {
        let z = scaler.apply_row(&s.raw_features);
        for (a, b) in z.iter().zip(s.standardized_features.iter()) {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
            assert!(d < tol, "sample {} scaler diff {} > {}", k, d, tol);
        }
    }
    println!("Scaler golden: max diff = {:.2e}", max_diff);
    Ok(())
}

fn ridge_golden_equivalence(
    manual_path: &str,
    getter: fn(&GoldenSample) -> f64,
    label: &str,
) -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let ridge = RidgeManual::load(root.join(manual_path))?;
    assert_eq!(golden.feature_names, ridge.feature_names);

    let tol = golden.tolerance_abs_regression;
    let mut max_diff = 0.0_f64;
    for (k, s) in golden.samples.iter().enumerate() {
        let y_hat = ridge.predict_z(&s.standardized_features);
        let expected = getter(s);
        let d = (y_hat - expected).abs();
        if d > max_diff { max_diff = d; }
        assert!(
            d < tol,
            "{} sample {} diff {:.2e} > tol {:.0e} (got {:.6} vs {:.6})",
            label, k, d, tol, y_hat, expected
        );
    }
    println!("{} golden: max diff = {:.2e}", label, max_diff);
    Ok(())
}

#[test]
fn golden_ridge_24h_matches_pred_ridge() -> Result<()> {
    ridge_golden_equivalence(
        "models/ridge_manual.json",
        |s| s.pred_ridge,
        "Ridge 24h",
    )
}

#[test]
fn golden_ridge_48h_matches_pred_ridge_48h() -> Result<()> {
    ridge_golden_equivalence(
        "models/ridge_48h_manual.json",
        |s| s.pred_ridge_48h,
        "Ridge 48h",
    )
}

#[test]
fn golden_ridge_72h_matches_pred_ridge_72h() -> Result<()> {
    ridge_golden_equivalence(
        "models/ridge_72h_manual.json",
        |s| s.pred_ridge_72h,
        "Ridge 72h",
    )
}

#[test]
fn golden_rain_rf_matches_pred_class() -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let rfc = load_rain_model(root.join("models/rain_rf_model.bin"))?;

    let xs: Vec<Vec<f64>> = golden
        .samples
        .iter()
        .map(|s| s.raw_features.clone())
        .collect();
    let dm = DenseMatrix::from_2d_vec(&xs);
    let preds: Vec<u32> = rfc.predict(&dm)?;

    let mismatches = preds
        .iter()
        .zip(golden.samples.iter())
        .filter(|(p, s)| **p != s.pred_rfc_class)
        .count();
    assert_eq!(mismatches, 0, "RFC diverged on {} samples", mismatches);
    println!("RFC golden: 0 mismatches on {} samples", preds.len());
    Ok(())
}

#[test]
fn golden_rain_bagging_probability_matches_raw_proba() -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let ensemble = load_rain_bagging_ensemble(root.join("models/rain_bagging_ensemble.bin"))?;

    let xs: Vec<Vec<f64>> = golden
        .samples
        .iter()
        .map(|s| s.raw_features.clone())
        .collect();
    let probs = bagging_vote_batch(&ensemble, &xs)?;

    // The bagging proxy in Nb05 used 30 trees, so probabilities are
    // multiples of 1/30. Tolerance is 1e-9.
    let tol = golden.tolerance_abs_probability.max(1e-9);
    let mut max_diff = 0.0_f64;
    for (k, (p, s)) in probs.iter().zip(golden.samples.iter()).enumerate() {
        let d = (p - s.pred_rain_proba_raw).abs();
        if d > max_diff { max_diff = d; }
        assert!(
            d < tol,
            "sample {} bagging diff {:.2e} > tol {:.0e} (got {:.6} vs {:.6})",
            k, d, tol, p, s.pred_rain_proba_raw
        );
    }
    println!("Bagging proba golden: max diff = {:.2e}", max_diff);
    Ok(())
}

#[test]
fn golden_rain_calibration_matches_cal_proba() -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let calib = RainCalibration::load(root.join("models/rain_calibration.json"))?;

    let tol = 1e-9;
    let mut max_diff = 0.0_f64;
    for (k, s) in golden.samples.iter().enumerate() {
        let cal = calib.calibrate(s.pred_rain_proba_raw);
        let d = (cal - s.pred_rain_proba_cal).abs();
        if d > max_diff { max_diff = d; }
        assert!(
            d < tol,
            "sample {} calibration diff {:.2e} > {:.0e}",
            k, d, tol
        );
    }
    println!("Calibration golden: max diff = {:.2e}", max_diff);
    Ok(())
}

#[test]
fn production_contract_is_valid() -> Result<()> {
    let root = project_root();
    let contract = ProductionContract::load(root.join("models/production_contract.json"))?;
    assert!(contract.n_features > 0);
    assert_eq!(contract.n_features, contract.feature_names.len());
    assert!(contract.expected_regression_metrics.rmse > 0.0);
    assert!(contract.expected_regression_metrics.rmse < 10.0);
    assert!(contract.tolerance_abs_golden_path > 0.0);
    // v2.0 fields must be present
    assert!(contract.expected_regression_metrics_48h.is_some(), "48h metrics missing");
    assert!(contract.expected_regression_metrics_72h.is_some(), "72h metrics missing");
    println!(
        "Contract v{}: winner={}, 24h RMSE={:.3}, 48h RMSE={:.3}, 72h RMSE={:.3}",
        contract.version,
        contract.winner_model,
        contract.expected_regression_metrics.rmse,
        contract.expected_regression_metrics_48h.as_ref().unwrap().rmse,
        contract.expected_regression_metrics_72h.as_ref().unwrap().rmse,
    );
    Ok(())
}
