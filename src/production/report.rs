//! README injector (v2.0 — richer layout).
//!
//! The README now contains three blocks that this module rewrites on
//! every run:
//!
//! 1. **Summary table** — one row per city with "Now" + +24/48/72 h
//!    point predictions + calibrated rain probability over the next
//!    24 h.
//! 2. **Hourly detail** — a collapsible `<details>` block per city with
//!    the 24 hourly predictions (local time, temp, rain %,
//!    precipitation, clouds, wind, weather condition).
//! 3. **Model performance + drift banner** — stored metrics from
//!    Notebook 05 plus the most recent drift snapshot.
//!
//! Section boundaries in the README are anchored by the existing
//! `### 24-Hour, 48-Hour & 72-Hour Forecast` heading and end at the
//! next `###` heading.

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use std::fs;
use std::path::Path;

use super::drift_monitor::DriftSnapshot;
use super::predict::DailyReport;

const FORECAST_HEADER: &str = "### 24-Hour, 48-Hour & 72-Hour Forecast";
const LAST_RUN_MARK: &str   = "> Auto-updated daily at 06:00 UTC";

// -------------------------------------------------------------------------
// Table builders
// -------------------------------------------------------------------------

pub fn build_summary_table(report: &DailyReport) -> String {
    let mut out = String::new();
    out.push_str(FORECAST_HEADER);
    out.push_str("\n\n");
    out.push_str("| City | Country | Now (local) | Current | +24h | +48h | +72h | Rain 24h | Confidence |\n");
    out.push_str("|------|---------|-------------|---------|------|------|------|----------|------------|\n");
    for cf in &report.cities {
        let local_hour = local_hour_short(&cf.reference_local_time);
        out.push_str(&format!(
            "| {name} | {flag} | {hour} | {cur:+.1}°C | {p24:+.1}°C | {p48:+.1}°C | {p72:+.1}°C | {rain:>3}% | ±{sigma:.1}°C |\n",
            name = cf.city,
            flag = cf.flag,
            hour = local_hour,
            cur = cf.current.temperature_c,
            p24 = cf.multi_horizon.t_plus_24h.temperature_c,
            p48 = cf.multi_horizon.t_plus_48h.temperature_c,
            p72 = cf.multi_horizon.t_plus_72h.temperature_c,
            rain = (cf.rain_next_24h.probability * 100.0).round() as i32,
            sigma = cf.meta.expected_rmse_24h_c,
        ));
    }
    out.push_str("\n");
    out.push_str("> **Source of each horizon.** `+24h`, `+48h`, and `+72h` all come from dedicated Ridge (alpha=10) ");
    out.push_str("models trained in Notebook 05 on the `temp_next_{24,48,72}h` targets. ");
    out.push_str("All three models share the same feature set and scaler; the RMSE grows from ~3.4 °C at 24 h to ~5.1 °C at 72 h ");
    out.push_str("because weather decorrelates over time. `Rain 24h` is the aggregate calibrated probability from a 30-tree ");
    out.push_str("DecisionTree bagging ensemble, mapped through the Notebook 05 reliability curve. ");
    out.push_str("`Confidence` is ±1 sigma = ±RMSE on the held-out test. `Now (local)` is the city's local time at the reference row.\n");
    out
}

pub fn build_hourly_details(report: &DailyReport) -> String {
    let mut out = String::new();
    out.push_str("\n### Hourly Predictions (next 24 h, per city)\n\n");
    out.push_str("Click a city to expand. Each row is one hour; `temp` is our Ridge 24 h model rolled across the past 24 hours, ");
    out.push_str("`rain %` is the calibrated bagging probability, `precip` and `clouds` and `wind` come from Open-Meteo's own NWP forecast.\n\n");
    for cf in &report.cities {
        out.push_str(&format!(
            "<details><summary><strong>{flag} {name}</strong> — {zone}, {tz}</summary>\n\n",
            flag = cf.flag,
            name = cf.city,
            zone = cf.climate_zone,
            tz = cf.timezone,
        ));
        out.push_str("| +h | local time | temp | rain % | precip | clouds | wind | conditions |\n");
        out.push_str("|----|------------|------|--------|--------|--------|------|------------|\n");
        for h in &cf.hourly {
            out.push_str(&format!(
                "| +{off} | {t} | {temp:+.1}°C | {rain:>3}% | {precip:.1} mm | {cloud:>3}% | {wind:>4.1} km/h | {cond} |\n",
                off = h.hour_offset,
                t = h.local_time.replace('T', " "),
                temp = h.temperature_c,
                rain = (h.rain_probability * 100.0).round() as i32,
                precip = h.precipitation_mm,
                cloud = h.cloudcover_pct.round() as i32,
                wind = h.windspeed_kmh,
                cond = h.weather_condition,
            ));
        }
        out.push_str("\n</details>\n\n");
    }
    out
}

pub fn build_performance_block(report: &DailyReport, drift: Option<&DriftSnapshot>) -> String {
    let any_meta = report.cities.first().map(|c| c.meta.clone());
    let mut out = String::new();
    out.push_str("\n### Model Performance (held-out test set)\n\n");
    out.push_str("| Metric | Value |\n|--------|-------|\n");
    if let Some(m) = any_meta {
        out.push_str(&format!("| 24 h model | {} |\n", m.model_24h));
        out.push_str(&format!("| 48 h model | {} |\n", m.model_48h));
        out.push_str(&format!("| 72 h model | {} |\n", m.model_72h));
        out.push_str(&format!("| Rain model | {} |\n", m.rain_model));
        out.push_str(&format!("| Test RMSE (24 h) | {:.2} °C |\n", m.expected_rmse_24h_c));
        out.push_str(&format!("| Test RMSE (48 h) | {:.2} °C |\n", m.expected_rmse_48h_c));
        out.push_str(&format!("| Test RMSE (72 h) | {:.2} °C |\n", m.expected_rmse_72h_c));
        out.push_str(&format!("| Bias correction (24 h) | {:+.2} °C |\n", m.bias_correction_24h_c));
        out.push_str(&format!("| Contract version | {} |\n", m.model_version));
    }
    out.push_str(&format!(
        "| Successful cities | {} / {} |\n",
        report.n_successful, report.n_cities
    ));

    if let Some(d) = drift {
        out.push_str("\n### Drift Monitor (last run)\n\n");
        let status_icon = if d.any_drift { "⚠" } else { "✓" };
        out.push_str(&format!(
            "{icon} Max PSI = {psi:.3}  |  observations = {n}  |  cities = {c}  |  status = {st}\n\n",
            icon = status_icon,
            psi = d.max_psi,
            n = d.n_observations,
            c = d.n_cities,
            st = if d.any_drift { "drift detected" } else { "stable" }
        ));
        out.push_str("| Feature | PSI | KS | Mean shift (σ_ref) | Status |\n");
        out.push_str("|---------|-----|----|--------------------|--------|\n");
        for (name, f) in &d.features {
            out.push_str(&format!(
                "| `{}` | {:.3} | {:.3} | {:+.2} | {} |\n",
                name, f.psi, f.ks, f.mean_shift_in_ref_std, f.status
            ));
        }
    }
    out
}

// -------------------------------------------------------------------------
// README injection
// -------------------------------------------------------------------------

pub fn update_readme<P: AsRef<Path>>(
    path: P,
    report: &DailyReport,
    drift: Option<&DriftSnapshot>,
) -> Result<bool> {
    let readme = fs::read_to_string(path.as_ref())
        .with_context(|| format!("read README from {:?}", path.as_ref()))?;

    let ts = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let mut updated = replace_line(
        &readme,
        LAST_RUN_MARK,
        &format!("> Auto-updated daily at 06:00 UTC | Last run: {ts}"),
    );

    if let Some(start) = updated.find(FORECAST_HEADER) {
        let tail = &updated[start + FORECAST_HEADER.len()..];
        let end_offset = tail
            .find("\n### ")
            .map(|off| start + FORECAST_HEADER.len() + off)
            .unwrap_or_else(|| updated.len());
        let before = &updated[..start];
        let after = &updated[end_offset..];

        let mut new_section = build_summary_table(report);
        new_section.push_str(&build_hourly_details(report));
        new_section.push_str(&build_performance_block(report, drift));

        updated = format!("{before}{new_section}\n{after}");
    } else {
        return Err(anyhow!(
            "README does not contain the '{}' marker",
            FORECAST_HEADER
        ));
    }

    fs::write(path.as_ref(), updated)
        .with_context(|| format!("write README to {:?}", path.as_ref()))?;
    Ok(true)
}

fn replace_line(src: &str, marker: &str, new_line: &str) -> String {
    src.lines()
        .map(|l| {
            if l.starts_with(marker) {
                new_line.to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract `HH:MM` from an ISO local-time string.
fn local_hour_short(iso: &str) -> String {
    iso.split('T').nth(1)
        .unwrap_or(iso)
        .chars().take(5).collect()
}
