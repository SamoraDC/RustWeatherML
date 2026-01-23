//! Daily predictions CLI tool for GitHub Actions.
//!
//! This binary fetches current weather data, runs predictions,
//! and updates the README.md with the latest forecasts.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output format (markdown, json)
    #[arg(short, long, default_value = "markdown")]
    format: String,

    /// Update README.md in place
    #[arg(short, long)]
    update_readme: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("RustWeatherML - Daily Predictions");
    println!("==================================");
    println!("Format: {}", args.format);
    println!("Update README: {}", args.update_readme);

    // TODO: Implement prediction logic
    // 1. Load trained models
    // 2. Fetch current weather data for all cities
    // 3. Generate predictions (24h, 48h, 72h)
    // 4. Format output
    // 5. Optionally update README.md

    println!("\nPrediction logic not yet implemented.");
    println!("This will be completed after model training.");

    Ok(())
}
