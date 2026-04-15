//! Production drift monitoring.
//!
//! At every run we compare the distribution of the **last 3 days of
//! fetched data** against the **training reference** persisted by
//! Notebook 06 in `models/drift_reference.json`. For each monitored
//! feature we compute:
//!
//! - **PSI** (Population Stability Index) with the same quantile edges
//!   the notebook used for the reference — this is what makes the two
//!   runs comparable, not just numerically similar.
//! - **KS** statistic between the reference empirical CDF and the
//!   current empirical CDF.
//! - Mean and std shift in reference-scale units (i.e. (current_mean -
//!   ref_mean) / ref_std).
//!
//! The snapshot is persisted under `data/monitoring_history/YYYY-MM-DD.json`
//! so the Grafana/HTML dashboard can plot PSI trends over time and spot
//! slow seasonal / concept drift.

use anyhow::{Context, Result};
use chrono::Utc;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::artifacts::{DriftFeatureRef, DriftReference};
use super::config::paths;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftSnapshot {
    pub timestamp_utc: String,
    pub n_observations: usize,
    pub n_cities: usize,
    pub max_psi: f64,
    pub any_drift: bool,
    pub features: BTreeMap<String, FeatureDrift>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDrift {
    pub psi: f64,
    pub ks: f64,
    pub current_mean: f64,
    pub current_std: f64,
    pub mean_shift_in_ref_std: f64,
    pub n_observations: usize,
    pub status: String,
}

fn interpret(psi: f64, thresholds_moderate: f64, thresholds_severe: f64) -> String {
    if psi >= thresholds_severe {
        "severe".to_string()
    } else if psi >= thresholds_moderate {
        "moderate".to_string()
    } else {
        "ok".to_string()
    }
}

/// Bucket a value into one of `edges.len() - 1` bins.
fn bin_index(value: f64, edges: &[f64]) -> usize {
    let n_bins = edges.len().saturating_sub(1);
    if n_bins == 0 {
        return 0;
    }
    for (i, pair) in edges.windows(2).enumerate() {
        if value >= pair[0] && value < pair[1] {
            return i;
        }
    }
    n_bins - 1
}

/// Compute PSI between ref_probs (expected) and current empirical probs.
fn psi(ref_probs: &[f64], cur_counts: &[usize], n_cur: usize) -> f64 {
    let eps = 1e-6;
    let n_cur_f = n_cur.max(1) as f64;
    let mut sum = 0.0;
    for (rp, cc) in ref_probs.iter().zip(cur_counts.iter()) {
        let rp = rp.max(eps);
        let cp = (*cc as f64 / n_cur_f).max(eps);
        sum += (cp - rp) * (cp / rp).ln();
    }
    sum
}

/// Compute the KS statistic between two empirical CDFs using the ref
/// bin edges as evaluation points.
fn ks_stat_from_bins(ref_probs: &[f64], cur_counts: &[usize], n_cur: usize) -> f64 {
    let n_cur_f = n_cur.max(1) as f64;
    let mut ref_cdf = 0.0;
    let mut cur_cdf = 0.0;
    let mut max_d = 0.0_f64;
    for (rp, cc) in ref_probs.iter().zip(cur_counts.iter()) {
        ref_cdf += rp;
        cur_cdf += *cc as f64 / n_cur_f;
        let d = (ref_cdf - cur_cdf).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

fn feature_drift(
    _feature_name: &str,
    feature_ref: &DriftFeatureRef,
    values: &[f64],
    thresh_mod: f64,
    thresh_sev: f64,
) -> FeatureDrift {
    let n_cur = values.len();
    let mut counts = vec![0usize; feature_ref.ref_probs.len()];
    for &v in values {
        if v.is_finite() {
            let b = bin_index(v, &feature_ref.edges);
            if b < counts.len() {
                counts[b] += 1;
            }
        }
    }
    let psi_val = psi(&feature_ref.ref_probs, &counts, n_cur);
    let ks_val = ks_stat_from_bins(&feature_ref.ref_probs, &counts, n_cur);

    let n_f = n_cur.max(1) as f64;
    let mean: f64 = values.iter().filter(|v| v.is_finite()).sum::<f64>() / n_f;
    let var: f64 = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / n_f.max(1.0);
    let std = var.sqrt();
    let ref_std = feature_ref.ref_std.max(1e-9);
    let shift = (mean - feature_ref.ref_mean) / ref_std;

    FeatureDrift {
        psi: psi_val,
        ks: ks_val,
        current_mean: (mean * 1000.0).round() / 1000.0,
        current_std: (std * 1000.0).round() / 1000.0,
        mean_shift_in_ref_std: (shift * 1000.0).round() / 1000.0,
        n_observations: n_cur,
        status: interpret(psi_val, thresh_mod, thresh_sev),
    }
}

/// Compute drift statistics from an engineered DataFrame.
pub fn compute_drift(df: &DataFrame, reference: &DriftReference) -> Result<DriftSnapshot> {
    let mut features: BTreeMap<String, FeatureDrift> = BTreeMap::new();
    for (feat_name, feat_ref) in reference.features.iter() {
        let col = df
            .column(feat_name.as_str())
            .with_context(|| format!("feature {feat_name} missing"))?;
        let f = col
            .cast(&DataType::Float64)
            .context("cast to f64 for drift")?;
        let values: Vec<f64> = f
            .f64()?
            .into_iter()
            .filter_map(|x| x)
            .collect();
        let drift = feature_drift(
            feat_name,
            feat_ref,
            &values,
            reference.psi_thresholds.moderate,
            reference.psi_thresholds.severe,
        );
        features.insert(feat_name.clone(), drift);
    }
    let max_psi = features.values().map(|f| f.psi).fold(0.0_f64, f64::max);
    let n_obs = features.values().map(|f| f.n_observations).max().unwrap_or(0);
    let n_cities_in_df = unique_cities(df)?;
    Ok(DriftSnapshot {
        timestamp_utc: Utc::now().to_rfc3339(),
        n_observations: n_obs,
        n_cities: n_cities_in_df,
        max_psi,
        any_drift: max_psi >= reference.psi_thresholds.moderate,
        features,
    })
}

/// Persist the snapshot to `data/monitoring_history/YYYY-MM-DDTHHZ.json`
/// and also refresh `latest.json` + `index.json` so the dashboard and
/// Grafana can enumerate the history deterministically without a
/// directory listing.
pub fn save_snapshot<P: AsRef<Path>>(root: P, snapshot: &DriftSnapshot) -> Result<PathBuf> {
    let dir = root.as_ref().join(paths::MONITORING_HISTORY_DIR);
    fs::create_dir_all(&dir)
        .with_context(|| format!("create {}", dir.display()))?;

    // Hour-stamped filename from the snapshot timestamp (UTC, ISO 8601).
    let stem = hour_stamp_from_iso(&snapshot.timestamp_utc);
    let filename = format!("{stem}.json");
    let path = dir.join(&filename);
    let body = serde_json::to_string_pretty(snapshot)?;
    fs::write(&path, &body)
        .with_context(|| format!("write {}", path.display()))?;

    // latest.json — overwritten every run.
    let latest_path = root.as_ref().join(paths::MONITORING_LATEST);
    fs::write(&latest_path, &body)
        .with_context(|| format!("write {}", latest_path.display()))?;

    // index.json — sorted list of every history file for the dashboard.
    refresh_index(&dir, &root.as_ref().join(paths::MONITORING_INDEX))?;

    Ok(path)
}

/// Convert an ISO-8601 timestamp like "2026-04-15T12:34:56+00:00" into
/// the hour-stamped stem "2026-04-15T12Z".
fn hour_stamp_from_iso(iso: &str) -> String {
    // We just extract the date + hour from the first 13 characters and
    // tag with Z.
    let s: String = iso.chars().take(13).collect(); // "2026-04-15T12"
    format!("{}Z", s)
}

/// Build a sorted list of all monitoring JSON files and write it to
/// `index.json`. The dashboard uses this to iterate history without
/// needing a file-system listing capability.
fn refresh_index(dir: &Path, index_path: &Path) -> Result<()> {
    let mut files: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".json") && n != "latest.json" && n != "index.json")
        .collect();
    files.sort();
    let body = serde_json::to_string_pretty(&serde_json::json!({
        "dir": dir.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "count": files.len(),
        "files": files,
    }))?;
    fs::write(index_path, body)
        .with_context(|| format!("write {}", index_path.display()))?;
    Ok(())
}

fn unique_cities(df: &DataFrame) -> Result<usize> {
    let cities_col = df.column("city")?.str()?;
    let mut set = std::collections::HashSet::new();
    for i in 0..df.height() {
        if let Some(c) = cities_col.get(i) {
            set.insert(c.to_string());
        }
    }
    Ok(set.len())
}

/// Load the last N days of monitoring history, ordered oldest-first.
pub fn load_recent_history<P: AsRef<Path>>(root: P, n_days: usize) -> Result<Vec<DriftSnapshot>> {
    let dir = root.as_ref().join(paths::MONITORING_HISTORY_DIR);
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut files: Vec<_> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("json"))
        .collect();
    files.sort();
    let start = files.len().saturating_sub(n_days);
    let mut out = Vec::new();
    for p in &files[start..] {
        let raw = fs::read_to_string(p)?;
        if let Ok(s) = serde_json::from_str::<DriftSnapshot>(&raw) {
            out.push(s);
        }
    }
    Ok(out)
}

/// Ensure the monitoring module is part of the `production` submodule tree.
pub fn module_marker() {}
