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
// Match the original marker ("Auto-updated daily at 06:00 UTC") AND the new
// marker ("Auto-updated every 3 h") so reruns keep working regardless of
// which version of the README we started from.
const LAST_RUN_MARK_OLD: &str = "> Auto-updated daily at 06:00 UTC";
const LAST_RUN_MARK_NEW: &str = "> Auto-updated every 3 h";

// -------------------------------------------------------------------------
// Table builders
// -------------------------------------------------------------------------

pub fn build_summary_table(report: &DailyReport) -> String {
    let mut out = String::new();
    out.push_str(FORECAST_HEADER);
    out.push_str("\n\n");
    out.push_str("| City | Country | Now (local) | Current | +24h | +48h | +72h | Rain 24h | NWP precip | ML only | Confidence |\n");
    out.push_str("|------|---------|-------------|---------|------|------|------|----------|------------|---------|------------|\n");
    for cf in &report.cities {
        let local_hour = local_hour_short(&cf.reference_local_time);
        let rain = &cf.rain_next_24h;
        out.push_str(&format!(
            "| {name} | {flag} | {hour} | {cur:+.1}°C | {p24:+.1}°C | {p48:+.1}°C | {p72:+.1}°C | {pct:>3}% | {mm:.1} mm ({hrs}h) | {ml_pct:>3}% | ±{sigma:.1}°C |\n",
            name = cf.city,
            flag = cf.flag,
            hour = local_hour,
            cur = cf.current.temperature_c,
            p24 = cf.multi_horizon.t_plus_24h.temperature_c,
            p48 = cf.multi_horizon.t_plus_48h.temperature_c,
            p72 = cf.multi_horizon.t_plus_72h.temperature_c,
            pct = (rain.probability * 100.0).round() as i32,
            mm = rain.total_precip_mm,
            hrs = rain.n_hours_with_rain,
            ml_pct = (rain.model_probability * 100.0).round() as i32,
            sigma = cf.meta.expected_rmse_24h_c,
        ));
    }
    out.push_str("\n");
    out.push_str("> **How to read the rain columns.** `Rain 24h` is the **blended** probability ");
    out.push_str("that there will be any rain at all in the next 24 h, computed as ");
    out.push_str("`α·p_NWP + (1−α)·p_ML` with `α = 0.9`. The physical NWP signal dominates because ");
    out.push_str("the ML classifier was trained on a target (`precip_sum_next_24h > 0 mm`) that was ");
    out.push_str("73.6 % positive in the training set and therefore has a positive prior bias. ");
    out.push_str("`NWP precip` shows the *actual* predicted rainfall in millimetres from Open-Meteo's ");
    out.push_str("numerical weather model, plus how many hours will have any precipitation. ");
    out.push_str("`ML only` shows the bagging-ensemble probability in isolation so you can see the drift. ");
    out.push_str("`+24h`, `+48h`, `+72h` come from dedicated Ridge (α=10) regressors trained in ");
    out.push_str("Notebook 05 on `temp_next_{24,48,72}h` with RMSE 3.5 / 4.5 / 5.1 °C. ");
    out.push_str("`Confidence` is ±1σ = ±RMSE of the 24 h test. `Now (local)` is the city's local time.\n");
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
        out.push_str("| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |\n");
        out.push_str("|----|------------|------|--------|-------|------|--------|--------|------|------------|\n");
        for h in &cf.hourly {
            out.push_str(&format!(
                "| +{off} | {t} | {temp:+.1}°C | {rain:>3}% | {nwp:>3}% | {ml:>3}% | {precip:.1} mm | {cloud:>3}% | {wind:>4.1} km/h | {cond} |\n",
                off = h.hour_offset,
                t = h.local_time.replace('T', " "),
                temp = h.temperature_c,
                rain = (h.rain_probability * 100.0).round() as i32,
                nwp = (h.rain_probability_nwp * 100.0).round() as i32,
                ml = (h.rain_probability_model * 100.0).round() as i32,
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
    let new_line = format!(
        "> Auto-updated every 3 h via GitHub Actions | Last run: {ts}"
    );
    let mut updated = replace_line_multi(
        &readme,
        &[LAST_RUN_MARK_OLD, LAST_RUN_MARK_NEW],
        &new_line,
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

fn replace_line_multi(src: &str, markers: &[&str], new_line: &str) -> String {
    src.lines()
        .map(|l| {
            if markers.iter().any(|m| l.starts_with(m)) {
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
