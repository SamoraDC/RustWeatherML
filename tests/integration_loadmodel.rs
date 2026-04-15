//! Equivalence test: replay the 50 golden samples persisted by
//! Notebook 05 and assert the production Rust code reproduces each
//! prediction within the tolerances declared in `production_contract.json`.
//!
//! This test is the single formal guarantee that the runtime binary
//! consumes the same model the notebook evaluated. If any downstream
//! change — feature order, scaler computation, Ridge reconstruction —
//! drifts by more than 1e-9, CI fails here.

use anyhow::Result;
use rust_weather_ml::production::artifacts::{
    load_rain_model, ProductionContract, RidgeManual, Scaler,
};
use serde::Deserialize;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct GoldenFile {
    #[allow(dead_code)]
    version: String,
    tolerance_abs_regression: f64,
    tolerance_abs_classification: f64,
    n_samples: usize,
    feature_names: Vec<String>,
    samples: Vec<GoldenSample>,
}

#[derive(Debug, Deserialize)]
struct GoldenSample {
    #[allow(dead_code)]
    row_index_in_test: usize,
    #[allow(dead_code)]
    city: String,
    #[allow(dead_code)]
    timestamp: String,
    raw_features: Vec<f64>,
    standardized_features: Vec<f64>,
    #[allow(dead_code)]
    y_true_temp_next_24h: f64,
    #[allow(dead_code)]
    y_true_will_rain: u32,
    pred_ridge: f64,
    #[allow(dead_code)]
    pred_lasso: f64,
    #[allow(dead_code)]
    pred_rf: f64,
    #[allow(dead_code)]
    pred_gb: f64,
    pred_rfc_class: u32,
    #[allow(dead_code)]
    pred_rfc_probability: f64,
    #[allow(dead_code)]
    pred_log_class: u32,
}

fn project_root() -> std::path::PathBuf {
    // Cargo runs tests from the crate root (CARGO_MANIFEST_DIR).
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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

    assert_eq!(golden.feature_names, scaler.feature_names,
        "feature names must be identical between golden and scaler");
    assert_eq!(golden.n_samples, golden.samples.len());

    let tol = 1e-12;
    let mut max_diff = 0.0_f64;
    for (k, s) in golden.samples.iter().enumerate() {
        let z = scaler.apply_row(&s.raw_features);
        assert_eq!(z.len(), s.standardized_features.len());
        for (a, b) in z.iter().zip(s.standardized_features.iter()) {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
            assert!(
                d < tol,
                "sample {} scaler divergence {} > {}", k, d, tol
            );
        }
    }
    println!("Scaler golden: max diff = {:.2e} (tol {:.0e})", max_diff, tol);
    Ok(())
}

#[test]
fn golden_ridge_manual_matches_pred_ridge() -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let ridge = RidgeManual::load(root.join("models/ridge_manual.json"))?;

    assert_eq!(golden.feature_names, ridge.feature_names);

    let tol = golden.tolerance_abs_regression;
    let mut max_diff = 0.0_f64;
    for (k, s) in golden.samples.iter().enumerate() {
        let y_hat = ridge.predict_z(&s.standardized_features);
        let d = (y_hat - s.pred_ridge).abs();
        if d > max_diff { max_diff = d; }
        assert!(
            d < tol,
            "sample {} Ridge divergence {:.2e} > {:.0e} \
             (predicted {:.6} vs notebook {:.6})",
            k, d, tol, y_hat, s.pred_ridge
        );
    }
    println!("Ridge golden: max diff = {:.2e} (tol {:.0e})", max_diff, tol);
    Ok(())
}

#[test]
fn golden_rain_rf_matches_pred_class() -> Result<()> {
    let root = project_root();
    let golden = load_golden(&root.join("models/golden_test.json"))?;
    let rfc = load_rain_model(root.join("models/rain_rf_model.bin"))?;

    // Build a (n_samples x n_features) matrix.
    let xs: Vec<Vec<f64>> = golden
        .samples
        .iter()
        .map(|s| s.raw_features.clone())
        .collect();
    let dm = DenseMatrix::from_2d_vec(&xs);
    let preds: Vec<u32> = rfc.predict(&dm)?;

    let tol = golden.tolerance_abs_classification as i64;
    let mut mismatches = 0usize;
    for (k, (p, s)) in preds.iter().zip(golden.samples.iter()).enumerate() {
        let d = (*p as i64 - s.pred_rfc_class as i64).abs();
        if d > tol {
            mismatches += 1;
            eprintln!("sample {}: rfc predicted {} expected {}", k, p, s.pred_rfc_class);
        }
    }
    assert_eq!(
        mismatches, 0,
        "RFC predictions diverged on {} of {} golden samples",
        mismatches, preds.len()
    );
    println!("RFC golden: 0 mismatches on {} samples", preds.len());
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
    println!(
        "Contract: winner={}, RMSE={:.3}, CI=[{:.2}, {:.2}]",
        contract.winner_model,
        contract.expected_regression_metrics.rmse,
        contract.expected_regression_metrics.rmse_95_ci.0,
        contract.expected_regression_metrics.rmse_95_ci.1,
    );
    Ok(())
}
