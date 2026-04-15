//! Static HTML dashboard generator.
//!
//! Reads every daily prediction JSON in `data/predictions/` and every
//! drift snapshot in `data/monitoring_history/` to produce a single
//! self-contained HTML file in `docs/dashboard/index.html`. The file
//! renders with Chart.js (loaded from a CDN) and is served by GitHub
//! Pages out of the `docs/` folder.
//!
//! This is a best-effort, zero-dependency-at-runtime approach: the
//! HTML only needs a browser, no Grafana, no backend. For users who
//! want Grafana, we also emit `grafana_dashboard.json` which can be
//! imported directly into a Grafana instance that points at the
//! prediction JSON files via the Infinity plugin.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use super::config::paths;

const INDEX_HTML: &str = include_str!("dashboard_assets/index_template.html");
const GRAFANA_JSON: &str = include_str!("dashboard_assets/grafana_dashboard.json");
const DASHBOARD_README: &str = include_str!("dashboard_assets/README.md");

/// Ensure `docs/dashboard/` is populated with the static HTML, the
/// Grafana JSON, and a small README explaining both.
pub fn ensure_dashboard_assets<P: AsRef<Path>>(dash_dir: P) -> Result<()> {
    let dir = dash_dir.as_ref();
    fs::create_dir_all(dir)
        .with_context(|| format!("create dashboard dir {}", dir.display()))?;

    fs::write(dir.join("index.html"), INDEX_HTML)?;
    fs::write(dir.join("grafana_dashboard.json"), GRAFANA_JSON)?;
    fs::write(dir.join("README.md"), DASHBOARD_README)?;
    Ok(())
}

/// Regenerate the dashboard from the given project root.
pub fn regenerate<P: AsRef<Path>>(root: P) -> Result<()> {
    let dir = root.as_ref().join(paths::DASHBOARD_DIR);
    ensure_dashboard_assets(dir)
}
