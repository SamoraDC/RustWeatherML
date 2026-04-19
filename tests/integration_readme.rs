//! Structural audit for `README.md` + idempotency smoke-test for the
//! live-prediction injector.
//!
//! Previous versions of `update_readme` used "next `### ` heading" as
//! the end sentinel. Because the section it generated itself contained
//! multiple `### ` sub-headings, every run appended a fresh copy of the
//! hourly + performance + drift blocks in front of the ones from the
//! previous run — README.md grew to 10 600 lines with 22 duplicate
//! cycles. This test locks down the invariants of the v2.1 injector so
//! that regression cannot come back silently:
//!
//! * The live block is delimited by `<!-- BEGIN:LIVE_PREDICTIONS -->`
//!   and `<!-- END:LIVE_PREDICTIONS -->` and they appear exactly once.
//! * Each of the three `###` sub-headings appears exactly once inside
//!   that block.
//! * The `As of (local)` column label is used, not `Now (local)`.
//! * `rewrite_readme` is a fixed point (idempotent modulo the
//!   timestamp line) when called on the real README with a stub report.

use rust_weather_ml::production::drift_monitor::{DriftSnapshot, FeatureDrift};
use rust_weather_ml::production::predict::{
    CityForecast, CurrentObservation, DailyReport, ForecastMeta, HorizonPoint,
    HourlyPrediction, MultiHorizon, RainSummary,
};
use rust_weather_ml::production::report::rewrite_readme;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn readme_path() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest).join("README.md")
}

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
    let hourly: Vec<HourlyPrediction> = (1..=24_i32)
        .map(|k| HourlyPrediction {
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

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

/// Normalize the `Last run:` line so two consecutive rewrites can be
/// compared byte-wise.
fn strip_ts(text: &str) -> String {
    text.lines()
        .map(|l| {
            if l.starts_with("> Auto-updated every 3 h") {
                "> Auto-updated every 3 h via GitHub Actions | Last run: <TS>".to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn readme_has_exactly_one_live_block() {
    let body = fs::read_to_string(readme_path()).expect("read README.md");
    assert_eq!(
        count(&body, "<!-- BEGIN:LIVE_PREDICTIONS -->"),
        1,
        "README should contain exactly one BEGIN marker"
    );
    assert_eq!(
        count(&body, "<!-- END:LIVE_PREDICTIONS -->"),
        1,
        "README should contain exactly one END marker"
    );
    let begin = body.find("<!-- BEGIN:LIVE_PREDICTIONS -->").unwrap();
    let end = body.find("<!-- END:LIVE_PREDICTIONS -->").unwrap();
    assert!(begin < end, "BEGIN must precede END in README");
}

#[test]
fn readme_has_one_of_each_live_heading() {
    let body = fs::read_to_string(readme_path()).expect("read README.md");
    assert_eq!(
        count(&body, "### 24-Hour, 48-Hour & 72-Hour Forecast"),
        1,
        "forecast heading must appear once"
    );
    assert_eq!(
        count(&body, "### Hourly Predictions (next 24 h, per city)"),
        1,
        "hourly heading must appear once"
    );
    assert_eq!(
        count(&body, "### Model Performance (held-out test set)"),
        1,
        "model performance heading must appear once"
    );
    assert_eq!(
        count(&body, "### Drift Monitor (last run)"),
        1,
        "drift monitor heading must appear once"
    );
}

#[test]
fn readme_uses_as_of_label_not_now() {
    let body = fs::read_to_string(readme_path()).expect("read README.md");
    assert!(
        !body.contains("| Now (local) |"),
        "the obsolete 'Now (local)' column must not appear anywhere"
    );
    assert!(
        body.contains("| As of (local) |"),
        "the 'As of (local)' column must be present in the summary table"
    );
}

#[test]
fn readme_preserves_project_overview_tail() {
    let body = fs::read_to_string(readme_path()).expect("read README.md");
    assert!(
        body.contains("## 📋 Project Overview"),
        "project overview section must not be wiped out by the injector"
    );
    // The project overview must come after the END marker, not before.
    let end_pos = body.find("<!-- END:LIVE_PREDICTIONS -->").unwrap();
    let overview_pos = body.find("## 📋 Project Overview").unwrap();
    assert!(
        overview_pos > end_pos,
        "project overview must live below the live-prediction block"
    );
}

#[test]
fn rewrite_readme_is_idempotent_on_real_file() {
    let body = fs::read_to_string(readme_path()).expect("read README.md");
    let report = stub_report();
    let drift = stub_drift();
    let once = rewrite_readme(&body, &report, Some(&drift)).expect("first rewrite");
    let twice = rewrite_readme(&once, &report, Some(&drift)).expect("second rewrite");
    assert_eq!(
        strip_ts(&once),
        strip_ts(&twice),
        "rewrite_readme must be a fixed point (modulo the Last run timestamp)"
    );
    // And the output must still only contain one of each live heading
    // — even with the stub report there must be no accidental duplicate
    // introduced by the rewrite.
    assert_eq!(count(&twice, "### Hourly Predictions (next 24 h, per city)"), 1);
    assert_eq!(count(&twice, "### Model Performance (held-out test set)"), 1);
    assert_eq!(count(&twice, "### Drift Monitor (last run)"), 1);
    assert_eq!(count(&twice, "<!-- BEGIN:LIVE_PREDICTIONS -->"), 1);
    assert_eq!(count(&twice, "<!-- END:LIVE_PREDICTIONS -->"), 1);
}

#[test]
fn rewrite_readme_preserves_footer() {
    let body = fs::read_to_string(readme_path()).expect("read README.md");
    let report = stub_report();
    let drift = stub_drift();
    let out = rewrite_readme(&body, &report, Some(&drift)).expect("rewrite");
    for anchor in &[
        "## 📋 Project Overview",
        "## 🏗️ Project Structure",
        "## 🚀 Getting Started",
        "## 📚 Documentation",
        "## 📄 License",
    ] {
        assert!(
            out.contains(anchor),
            "rewrite dropped README anchor '{}'",
            anchor
        );
    }
}
