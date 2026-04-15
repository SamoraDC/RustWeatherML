//! Daily predictions binary for GitHub Actions.
//!
//! Steps:
//! 1. Load every artifact produced by Notebook 05.
//! 2. For each of the 14 cities, fetch fresh Open-Meteo data, engineer
//!    the features, and call the Ridge + RandomForest models.
//! 3. Emit a JSON report to `data/predictions/YYYY-MM-DD.json`.
//! 4. Inject the forecast table into `README.md`.
//!
//! The binary is deterministic: given the same artifacts and the same
//! API responses it will always produce the same output. The equivalence
//! to the notebooks is guarded by `tests/integration_loadmodel.rs`.

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use std::fs;
use std::path::PathBuf;

use rust_weather_ml::production::{
    artifacts::ModelBundle,
    config::{cities, paths},
    predict,
    report,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "RustWeatherML daily predictions", long_about = None)]
struct Args {
    /// Project root directory. Defaults to the current working directory.
    #[arg(long)]
    project_root: Option<PathBuf>,

    /// Write the report JSON but do NOT update README.md.
    #[arg(long)]
    dry_run: bool,

    /// Rate-limit between API calls (milliseconds).
    #[arg(long, default_value = "400")]
    rate_limit_ms: u64,

    /// Skip the network and load the latest report from disk instead.
    /// Useful for offline README-only reruns.
    #[arg(long)]
    from_cache: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args
        .project_root
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    println!("RustWeatherML - Daily Predictions");
    println!("==================================");
    println!("Project root: {}", root.display());

    // Load or compute the daily report.
    let report_data = if let Some(cache) = args.from_cache.as_ref() {
        println!("Loading cached report from {}", cache.display());
        let raw = fs::read_to_string(cache)?;
        serde_json::from_str::<predict::DailyReport>(&raw)?
    } else {
        println!("Loading model bundle ...");
        let bundle = ModelBundle::load_from_root(&root)
            .context("load model bundle")?;
        println!(
            "  - {} features, expected RMSE = {:.3} C",
            bundle.feature_count(),
            bundle.contract.expected_regression_metrics.rmse
        );

        let cfg = cities();
        println!("Running predictions for {} cities ...", cfg.len());
        predict::run_daily(&cfg, &bundle, args.rate_limit_ms)?
    };

    println!(
        "  -> {} / {} cities succeeded",
        report_data.n_successful, report_data.n_cities
    );

    // Persist the JSON report.
    let predictions_dir = root.join(paths::PREDICTIONS_DIR);
    fs::create_dir_all(&predictions_dir).with_context(|| {
        format!("create {}", predictions_dir.display())
    })?;
    let filename = format!("{}.json", Utc::now().format("%Y-%m-%d"));
    let report_path = predictions_dir.join(&filename);
    fs::write(&report_path, serde_json::to_string_pretty(&report_data)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    println!("Saved report {}", report_path.display());

    // Update README.
    if args.dry_run {
        println!("Dry run: README.md not touched.");
    } else {
        let readme = root.join(paths::README);
        report::update_readme(&readme, &report_data)?;
        println!("README.md updated.");
    }

    println!("Done.");
    Ok(())
}
