//! README injector.
//!
//! Finds the fenced section
//! ```markdown
//! ### 24-Hour, 48-Hour & 72-Hour Forecast
//! ...table...
//! ```
//! and rewrites it with fresh predictions. The replacement is bounded by
//! the `### 24-Hour, 48-Hour & 72-Hour Forecast` heading and the next
//! heading of the same level (`###`).

use anyhow::{Context, Result};
use chrono::Utc;
use std::fs;
use std::path::Path;

use super::predict::DailyReport;

const FORECAST_HEADER: &str = "### 24-Hour, 48-Hour & 72-Hour Forecast";
const LAST_RUN_MARK: &str   = "> Auto-updated daily at 06:00 UTC";

/// Build the markdown table for the forecast section.
pub fn build_forecast_table(report: &DailyReport) -> String {
    let mut out = String::new();
    out.push_str(FORECAST_HEADER);
    out.push_str("\n\n");
    out.push_str("| City | Country | Current | +24h | +48h | +72h | Rain % | Confidence |\n");
    out.push_str("|------|---------|---------|------|------|------|--------|------------|\n");

    // Preserve README ordering: the `cities` list is already in display order
    // because `run_daily` iterates `config::cities()`.
    for cf in &report.cities {
        // "Confidence" = ±RMSE, the 1-sigma uncertainty of the 24 h
        // prediction from Notebook 05. Reporting ±3.4 C is honest: a
        // single prediction is expected to land within that band ~68%
        // of the time.
        let sigma = cf.meta.expected_rmse_c;
        out.push_str(&format!(
            "| {name} | {flag} | {cur:+.1}°C | {p24:+.1}°C | {p48:+.1}°C | {p72:+.1}°C | {rain:>3}% | ±{sigma:.1}°C |\n",
            name = cf.city,
            flag = cf.flag,
            cur = cf.current_temp_c,
            p24 = cf.forecast.t_plus_24h_c,
            p48 = cf.forecast.t_plus_48h_c,
            p72 = cf.forecast.t_plus_72h_c,
            rain = (cf.rain.probability * 100.0).round() as i32,
            sigma = sigma,
        ));
    }
    out.push_str("\n> **Source of each horizon.** `+24h` comes from the Ridge (alpha=10) ");
    out.push_str("model we trained in Notebook 05 (post-hoc bias-corrected). ");
    out.push_str("`+48h` and `+72h` are taken directly from Open-Meteo's own NWP forecast, ");
    out.push_str("since we did not train dedicated models for those horizons. ");
    out.push_str("`Rain %` is the rain probability of the RandomForest classifier ");
    out.push_str("(high band 85% / low band 15% via the reliability curve in Nb05). ");
    out.push_str("`Confidence` is ±1 sigma = ±RMSE on the held-out test.\n");
    out
}

/// Build the performance block that sits below the forecast table.
pub fn build_performance_block(report: &DailyReport) -> String {
    let any_meta = report.cities.first().map(|c| c.meta.clone());
    let mut out = String::new();
    out.push_str("\n### Model Performance (held-out test set)\n\n");
    out.push_str("| Metric | Value |\n|--------|-------|\n");
    if let Some(m) = any_meta {
        out.push_str(&format!("| Model | {} |\n", m.model));
        out.push_str(&format!("| Test RMSE (24 h temperature) | {:.2} °C |\n", m.expected_rmse_c));
        out.push_str(&format!(
            "| 95 % CI of RMSE | [{:.2}, {:.2}] °C |\n",
            m.rmse_95_ci_c.0, m.rmse_95_ci_c.1
        ));
        out.push_str(&format!("| Skill vs persistence-24 h | {:+.3} |\n", m.skill_vs_persistence_24h));
        out.push_str(&format!("| Post-hoc bias correction  | {:+.2} °C |\n", m.bias_correction_c));
    }
    out.push_str(&format!(
        "| Successful cities         | {} / {} |\n",
        report.n_successful, report.n_cities
    ));
    out
}

/// Replace the forecast section of the README with fresh content.
/// Returns `Ok(true)` if the file was rewritten.
pub fn update_readme<P: AsRef<Path>>(path: P, report: &DailyReport) -> Result<bool> {
    let readme = fs::read_to_string(path.as_ref())
        .with_context(|| format!("read README from {:?}", path.as_ref()))?;

    // 1) Replace the "Auto-updated daily at..." line with the actual timestamp.
    let ts = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let mut updated = replace_line(&readme, LAST_RUN_MARK, &format!(
        "> Auto-updated daily at 06:00 UTC | Last run: {ts}"
    ));

    // 2) Find the forecast header and the next `###` heading and splice in
    //    our fresh block.
    if let Some(start) = updated.find(FORECAST_HEADER) {
        // Find the start of the **next** ### heading after the table.
        let tail = &updated[start + FORECAST_HEADER.len()..];
        let end_offset = tail
            .find("\n### ")
            .map(|off| start + FORECAST_HEADER.len() + off)
            .unwrap_or_else(|| updated.len());
        let before = &updated[..start];
        let after = &updated[end_offset..];

        let mut new_section = build_forecast_table(report);
        new_section.push_str(&build_performance_block(report));

        updated = format!("{before}{new_section}\n{after}");
    } else {
        return Err(anyhow::anyhow!(
            "README does not contain the '{}' marker",
            FORECAST_HEADER
        ));
    }

    fs::write(path.as_ref(), updated)
        .with_context(|| format!("write README to {:?}", path.as_ref()))?;
    Ok(true)
}

/// Replace lines beginning with `marker` in `src` with `new_line`.
fn replace_line(src: &str, marker: &str, new_line: &str) -> String {
    src.lines()
        .map(|l| if l.starts_with(marker) { new_line.to_string() } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
}
