//! Daily predictions binary for GitHub Actions (v2.0).
//!
//! Steps per run:
//!
//! 1. Load every artifact produced by Notebook 05 + Notebook 06.
//! 2. For each of the 14 cities, fetch fresh Open-Meteo data, engineer
//!    the features, and call the three Ridge models + the bagging
//!    rain ensemble.
//! 3. Compute a drift snapshot against the Notebook 06 reference.
//! 4. Emit `data/predictions/YYYY-MM-DD.json` (rich daily report).
//! 5. Emit `data/monitoring_history/YYYY-MM-DD.json` (drift snapshot).
//! 6. Inject both into `README.md`.
//! 7. Regenerate the static HTML dashboard at `docs/dashboard/index.html`.

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use std::fs;
use std::path::PathBuf;

use rust_weather_ml::production::{
    artifacts::ModelBundle,
    config::{cities, paths},
    drift_monitor::{self, DriftSnapshot},
    open_meteo::OpenMeteoClient,
    pipeline,
    predict::{self, DailyReport},
    report,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "RustWeatherML daily predictions (v2.0)", long_about = None)]
struct Args {
    /// Project root directory. Defaults to the current working directory.
    #[arg(long)]
    project_root: Option<PathBuf>,

    /// Write the report JSON but do NOT update README.md or the dashboard.
    #[arg(long)]
    dry_run: bool,

    /// Rate-limit between API calls (milliseconds).
    #[arg(long, default_value = "400")]
    rate_limit_ms: u64,

    /// Skip the network and load the latest report from disk instead.
    #[arg(long)]
    from_cache: Option<PathBuf>,

    /// Skip drift monitoring (useful for offline reruns).
    #[arg(long)]
    skip_drift: bool,

    /// Skip the HTML dashboard regeneration step.
    #[arg(long)]
    skip_dashboard: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args
        .project_root
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    println!("RustWeatherML - Daily Predictions v2.0");
    println!("=======================================");
    println!("Project root: {}", root.display());

    // Load or compute the daily report.
    let (report_data, drift_snapshot) = if let Some(cache) = args.from_cache.as_ref() {
        println!("Loading cached report from {}", cache.display());
        let raw = fs::read_to_string(cache)?;
        let report: DailyReport = serde_json::from_str(&raw)?;
        (report, None)
    } else {
        println!("Loading model bundle ...");
        let bundle = ModelBundle::load_from_root(&root)
            .context("load model bundle")?;
        println!(
            "  - {} features, ridge 24h RMSE = {:.3} C",
            bundle.feature_count(),
            bundle.contract.expected_regression_metrics.rmse
        );
        if let Some(m) = &bundle.contract.expected_regression_metrics_48h {
            println!("  - ridge 48h RMSE = {:.3} C", m.rmse);
        }
        if let Some(m) = &bundle.contract.expected_regression_metrics_72h {
            println!("  - ridge 72h RMSE = {:.3} C", m.rmse);
        }
        println!(
            "  - rain bagging ensemble: {} trees, Brier skill = {:.3}",
            bundle.rain_bagging.len(),
            bundle.rain_calibration.brier_skill_score
        );

        let cfg = cities();
        println!("\nRunning predictions for {} cities ...", cfg.len());
        let daily = predict::run_daily(&cfg, &bundle, args.rate_limit_ms)?;
        println!(
            "  -> {} / {} cities succeeded",
            daily.n_successful, daily.n_cities
        );

        // Compute drift snapshot on the same engineered DataFrame we
        // produced during prediction — but since `run_daily` owns that
        // object, we re-run the pipeline here on the cached response
        // data. For efficiency, the drift computation uses the same
        // per-city fetches by calling the pipeline once more.
        let drift = if args.skip_drift {
            None
        } else {
            println!("\nComputing drift snapshot ...");
            compute_drift_snapshot(&cfg, &bundle, args.rate_limit_ms).ok()
        };

        (daily, drift)
    };

    println!(
        "  -> {} / {} cities succeeded",
        report_data.n_successful, report_data.n_cities
    );

    // Persist the rich JSON report.
    let predictions_dir = root.join(paths::PREDICTIONS_DIR);
    fs::create_dir_all(&predictions_dir)?;
    let date = Utc::now().format("%Y-%m-%d").to_string();
    let report_path = predictions_dir.join(format!("{date}.json"));
    fs::write(&report_path, serde_json::to_string_pretty(&report_data)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    println!("Saved report {}", report_path.display());

    // Persist drift snapshot if we have one.
    if let Some(snap) = drift_snapshot.as_ref() {
        let path = drift_monitor::save_snapshot(&root, snap)?;
        println!(
            "Saved drift snapshot {} (max_psi={:.3}, {} features)",
            path.display(),
            snap.max_psi,
            snap.features.len()
        );
    }

    // Update README.
    if args.dry_run {
        println!("Dry run: README.md not touched.");
    } else {
        let readme = root.join(paths::README);
        report::update_readme(&readme, &report_data, drift_snapshot.as_ref())?;
        println!("README.md updated.");
    }

    // Regenerate dashboard.
    if args.dry_run || args.skip_dashboard {
        println!("Dashboard regeneration skipped.");
    } else {
        match rust_weather_ml::production::dashboard::regenerate(&root) {
            Ok(()) => println!("Dashboard assets written to {}", root.join(paths::DASHBOARD_DIR).display()),
            Err(e) => println!("Dashboard generation failed: {e}"),
        }
    }

    println!("\nDone.");
    Ok(())
}

/// Re-run the pipeline once for the sole purpose of computing drift.
/// This adds 14 extra API calls, but it keeps the drift snapshot
/// comparable to the predictions (same fetch window, same feature
/// engineering, same "now" alignment).
///
/// For production deployments where API quota is a concern, the
/// drift snapshot can be skipped via `--skip-drift`.
fn compute_drift_snapshot(
    cities: &[rust_weather_ml::production::config::CityConfig],
    bundle: &ModelBundle,
    rate_limit_ms: u64,
) -> Result<DriftSnapshot> {
    use std::thread;
    use std::time::Duration;

    let api = OpenMeteoClient::new();
    let mut dfs = Vec::new();
    for city in cities {
        if let Ok(resp) = api.fetch_recent(city, 3, 4) {
            if let Ok(df) = pipeline::response_to_dataframe(city, &resp) {
                dfs.push(df);
            }
        }
        thread::sleep(Duration::from_millis(rate_limit_ms));
    }
    if dfs.is_empty() {
        return Err(anyhow::anyhow!("no dataframes for drift"));
    }
    let stacked = pipeline::stack_dataframes(dfs)?;
    let engineered = pipeline::run(stacked)?;
    drift_monitor::compute_drift(&engineered, &bundle.drift_reference)
}
