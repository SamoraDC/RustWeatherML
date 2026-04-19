//! README injector (v2.1 — marker-anchored).
//!
//! The README now contains three blocks that this module rewrites on
//! every run:
//!
//! 1. **Summary table** — one row per city with the reference timestamp,
//!    the +24 / +48 / +72 h point predictions, and the calibrated rain
//!    probability over the next 24 h.
//!
//! 2. **Hourly detail** — a collapsible `<details>` block per city with
//!    the 24 hourly predictions (local time, temp, rain %, precipitation,
//!    clouds, wind, weather condition).
//!
//! 3. **Model performance + drift banner** — stored metrics from
//!    Notebook 05 plus the most recent drift snapshot.
//!
//! The live block is delimited by two HTML-comment markers
//! (`BEGIN_MARK` / `END_MARK`). Every invocation replaces the bytes
//! between those markers in full, which makes the operation idempotent
//! — running the injector twice with the same report produces the same
//! file.
//!
//! The previous version used "next `### ` heading" as the end sentinel,
//! but the new section itself contained several `### ` sub-headings, so
//! each run only rewrote the summary table and appended a fresh copy of
//! the hourly / performance / drift blocks in front of the ones written
//! by the previous run. This produced the 22-way duplication observed
//! in README.md. The marker-based approach eliminates that class of bug
//! entirely.

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use std::fs;
use std::path::Path;

use super::drift_monitor::DriftSnapshot;
use super::predict::DailyReport;

const FORECAST_HEADER: &str = "### 24-Hour, 48-Hour & 72-Hour Forecast";
/// Opening sentinel for the live-prediction block. Must live on its own
/// line so it is invisible in rendered markdown.
const BEGIN_MARK: &str = "<!-- BEGIN:LIVE_PREDICTIONS -->";
/// Closing sentinel for the live-prediction block.
const END_MARK: &str = "<!-- END:LIVE_PREDICTIONS -->";
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
    out.push_str("| City | Country | As of (local) | Current | +24h | +48h | +72h | Rain 24h | NWP precip | ML only | Confidence |\n");
    out.push_str("|------|---------|---------------|---------|------|------|------|----------|------------|---------|------------|\n");
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
    out.push('\n');
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
    out.push_str("`Confidence` is ±1σ = ±RMSE of the 24 h test. ");
    out.push_str("`As of (local)` is the city's local timestamp of the last observed hour that anchors ");
    out.push_str("the forecast (end of the Open-Meteo past window, typically 23:00 of the previous day).\n");
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

/// Build the full live-predictions block wrapped between the BEGIN / END
/// sentinels. Exposed so that tests can exercise it in isolation.
pub fn build_live_block(report: &DailyReport, drift: Option<&DriftSnapshot>) -> String {
    let mut out = String::new();
    out.push_str(BEGIN_MARK);
    out.push('\n');
    out.push_str(&build_summary_table(report));
    out.push_str(&build_hourly_details(report));
    out.push_str(&build_performance_block(report, drift));
    out.push('\n');
    out.push_str(END_MARK);
    out
}

pub fn update_readme<P: AsRef<Path>>(
    path: P,
    report: &DailyReport,
    drift: Option<&DriftSnapshot>,
) -> Result<bool> {
    let readme = fs::read_to_string(path.as_ref())
        .with_context(|| format!("read README from {:?}", path.as_ref()))?;
    let updated = rewrite_readme(&readme, report, drift)?;
    fs::write(path.as_ref(), updated)
        .with_context(|| format!("write README to {:?}", path.as_ref()))?;
    Ok(true)
}

/// Pure rewrite step — takes the current README body and returns the new
/// one. Factored out of `update_readme` so unit tests can exercise the
/// logic without hitting the filesystem.
pub fn rewrite_readme(
    source: &str,
    report: &DailyReport,
    drift: Option<&DriftSnapshot>,
) -> Result<String> {
    let ts = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let new_line = format!(
        "> Auto-updated every 3 h via GitHub Actions | Last run: {ts}"
    );
    let mut updated = replace_line_multi(
        source,
        &[LAST_RUN_MARK_OLD, LAST_RUN_MARK_NEW],
        &new_line,
    );

    let new_section = build_live_block(report, drift);

    match (updated.find(BEGIN_MARK), updated.find(END_MARK)) {
        (Some(b), Some(e)) if e > b => {
            // Fast path: both markers present and in the correct order.
            let before = &updated[..b];
            let after = &updated[e + END_MARK.len()..];
            updated = format!("{before}{new_section}{after}");
        }
        (Some(_), Some(_)) => {
            return Err(anyhow!(
                "README has '{}' appearing before '{}'; refusing to rewrite",
                END_MARK,
                BEGIN_MARK
            ));
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(anyhow!(
                "README has only one of '{}' / '{}'; refusing to rewrite",
                BEGIN_MARK,
                END_MARK
            ));
        }
        (None, None) => {
            // Migration path: anchor markers missing. This catches both
            // the pristine v1 README and the duplicate-bloated README we
            // inherited from the v2.0 injector. Cut from FORECAST_HEADER
            // up to (but not including) the next level-2 heading, then
            // splice the fresh marker-wrapped block in its place with a
            // horizontal-rule separator.
            let start = updated.find(FORECAST_HEADER).ok_or_else(|| {
                anyhow!(
                    "README contains neither the marker pair nor '{}'; nothing to anchor on",
                    FORECAST_HEADER
                )
            })?;
            let tail = &updated[start..];
            let level2_offset = tail
                .find("\n## ")
                .ok_or_else(|| anyhow!(
                    "README contains '{}' but no following '## ' heading; aborting",
                    FORECAST_HEADER
                ))?;
            let before = &updated[..start];
            // `level2_offset` points at the `\n` that precedes the next
            // level-2 heading. Keep that newline as the start of `after`,
            // and emit exactly one horizontal rule + one blank line so
            // the transition into `## 📋 Project Overview` renders the
            // same way the hand-authored source did.
            let after = &updated[start + level2_offset..];
            updated = format!("{before}{new_section}\n\n---\n{after}");
        }
    }

    Ok(updated)
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

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::production::drift_monitor::{DriftSnapshot, FeatureDrift};
    use crate::production::predict::{
        CityForecast, CurrentObservation, DailyReport, ForecastMeta, HorizonPoint,
        HourlyPrediction, MultiHorizon, RainSummary,
    };
    use std::collections::BTreeMap;

    fn stub_report() -> DailyReport {
        let meta = ForecastMeta {
            model_24h: "Ridge".into(),
            model_48h: "Ridge 48h".into(),
            model_72h: "Ridge 72h".into(),
            model_version: "2.0.0".into(),
            hourly_model: "Ridge rolled".into(),
            rain_model: "Bagging".into(),
            expected_rmse_24h_c: 3.5,
            expected_rmse_48h_c: 4.5,
            expected_rmse_72h_c: 5.0,
            bias_correction_24h_c: 0.5,
        };
        let hp = HorizonPoint {
            temperature_c: 20.0,
            local_time: "2026-04-18T23:00".into(),
            source: "ridge_24h".into(),
            expected_rmse_c: 3.5,
            ci95_low_c: 3.3,
            ci95_high_c: 3.7,
        };
        let hourly: Vec<HourlyPrediction> = (1..=24)
            .map(|k: i32| HourlyPrediction {
                local_time: format!("2026-04-18T{:02}:00", (k as u32) % 24),
                hour_offset: k,
                temperature_c: 20.0,
                temp_source: "ridge_24h".into(),
                precipitation_mm: 0.0,
                rain_probability: 0.1,
                rain_probability_model: 0.1,
                rain_probability_nwp: 0.1,
                cloudcover_pct: 10.0,
                windspeed_kmh: 5.0,
                weathercode: 0,
                weather_condition: "Clear".into(),
            })
            .collect();
        let cf = CityForecast {
            city: "Test City".into(),
            country_code: "TC".into(),
            flag: "🏳".into(),
            latitude: 0.0,
            longitude: 0.0,
            elevation_m: 0.0,
            timezone: "UTC".into(),
            climate_zone: "Cfa".into(),
            reference_local_time: "2026-04-17T23:00".into(),
            reference_utc_time: "2026-04-17T23:00Z".into(),
            current: CurrentObservation {
                temperature_c: 19.0,
                dewpoint_c: 12.0,
                humidity_pct: 70.0,
                windspeed_kmh: 5.0,
                winddir_deg: 180.0,
                pressure_hpa: 1012.0,
                cloudcover_pct: 10.0,
                precipitation_mm: 0.0,
                weathercode: 0,
                weather_condition: "Clear".into(),
            },
            hourly,
            multi_horizon: MultiHorizon {
                t_plus_24h: hp.clone(),
                t_plus_48h: hp.clone(),
                t_plus_72h: hp,
            },
            rain_next_24h: RainSummary {
                any_rain: false,
                probability: 0.1,
                model_probability: 0.1,
                nwp_probability: 0.1,
                blend_alpha_nwp: 0.9,
                rfc_raw_class: 0,
                n_hours_with_rain: 0,
                total_precip_mm: 0.0,
            },
            meta,
        };
        DailyReport {
            generated_at_utc: "2026-04-18T21:46Z".into(),
            n_cities: 1,
            n_successful: 1,
            n_failed: 0,
            model_contract_version: "2.0.0".into(),
            cities: vec![cf],
            failures: vec![],
        }
    }

    fn stub_drift() -> DriftSnapshot {
        let mut features = BTreeMap::new();
        features.insert(
            "temperature_2m".to_string(),
            FeatureDrift {
                psi: 0.5,
                ks: 0.1,
                current_mean: 20.0,
                current_std: 4.0,
                mean_shift_in_ref_std: 0.2,
                n_observations: 2000,
                status: "moderate".into(),
            },
        );
        DriftSnapshot {
            timestamp_utc: "2026-04-18T21:46Z".into(),
            n_observations: 2000,
            n_cities: 14,
            max_psi: 0.5,
            any_drift: false,
            features,
        }
    }

    const MIGRATION_FIXTURE: &str = "# Title\n\n## 🌍 Live Weather Predictions\n\n> Auto-updated every 3 h via GitHub Actions | Last run: OLD\n\n### 24-Hour, 48-Hour & 72-Hour Forecast\n\nold summary table garbage\n\n### Hourly Predictions (next 24 h, per city)\n\nold hourly\n\n### Model Performance (held-out test set)\n\nold perf\n\n### Drift Monitor (last run)\n\nold drift\n\n### Hourly Predictions (next 24 h, per city)\n\nduplicate copy\n\n### Model Performance (held-out test set)\n\nduplicate copy\n\n---\n\n## 📋 Project Overview\n\nProject body stays.\n";

    #[test]
    fn rewrite_is_idempotent() {
        let report = stub_report();
        let drift = stub_drift();
        let first = rewrite_readme(MIGRATION_FIXTURE, &report, Some(&drift)).unwrap();
        let second = rewrite_readme(&first, &report, Some(&drift)).unwrap();
        // The only legitimate difference between consecutive runs is the
        // `Last run:` timestamp. We strip it before comparing so the test
        // is not flaky when the second call happens in a different UTC
        // minute.
        let strip_ts = |s: &str| {
            s.lines()
                .map(|l| {
                    if l.starts_with("> Auto-updated every 3 h") {
                        "> Auto-updated every 3 h via GitHub Actions | Last run: <TS>"
                            .to_string()
                    } else {
                        l.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(strip_ts(&first), strip_ts(&second));
    }

    #[test]
    fn rewrite_cleans_up_duplicates() {
        let report = stub_report();
        let drift = stub_drift();
        let out = rewrite_readme(MIGRATION_FIXTURE, &report, Some(&drift)).unwrap();
        // Exactly one of each live sub-heading must remain.
        assert_eq!(out.matches("### Hourly Predictions (next 24 h, per city)").count(), 1);
        assert_eq!(out.matches("### Model Performance (held-out test set)").count(), 1);
        assert_eq!(out.matches("### Drift Monitor (last run)").count(), 1);
        // Markers must be present and in the right order.
        let b = out.find(BEGIN_MARK).expect("BEGIN marker inserted");
        let e = out.find(END_MARK).expect("END marker inserted");
        assert!(b < e, "BEGIN must precede END");
        // The project overview must still be there, untouched.
        assert!(out.contains("## 📋 Project Overview"));
        assert!(out.contains("Project body stays."));
    }

    #[test]
    fn rewrite_accepts_marker_wrapped_input() {
        let report = stub_report();
        let drift = stub_drift();
        // Start from a marker-wrapped README so we exercise the fast path.
        let seeded = rewrite_readme(MIGRATION_FIXTURE, &report, Some(&drift)).unwrap();
        let again = rewrite_readme(&seeded, &report, Some(&drift)).unwrap();
        assert!(again.contains(BEGIN_MARK));
        assert!(again.contains(END_MARK));
        assert_eq!(again.matches(BEGIN_MARK).count(), 1);
        assert_eq!(again.matches(END_MARK).count(), 1);
    }

    #[test]
    fn summary_header_uses_as_of_label() {
        let out = build_summary_table(&stub_report());
        assert!(out.contains("| As of (local) |"));
        assert!(!out.contains("| Now (local) |"));
    }
}
