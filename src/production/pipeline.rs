//! Runtime port of the Notebook 02 feature pipeline.
//!
//! Every transformation here is a direct Polars translation of the
//! expressions in `build_nb02.py`. The equivalence contract lives in
//! `models/features_schema.json` and is enforced by two mechanisms:
//!
//! 1. `tests/integration_loadmodel.rs` replays the 50 golden samples
//!    from `models/golden_test.json` and asserts `max|diff| < 1e-9`.
//! 2. The column order produced by [`build_feature_row`] is asserted at
//!    runtime to equal `scaler.feature_names` (see `predict.rs`).
//!
//! A critical invariant: **the order and naming of every derived column
//! must exactly match Notebook 02.** Adding a new feature here without
//! touching the notebook is a silent backwards-incompatibility.

use anyhow::{anyhow, Context, Result};
use chrono::{Datelike, NaiveDateTime, Timelike};
use polars::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::config::{CityConfig, CLIMATE_ZONES};
use super::open_meteo::ForecastResponse;

/// Build the raw DataFrame matching the Notebook 01/02 input schema.
///
/// Columns (exact order, exact dtypes):
/// `city, country_code, climate_zone, latitude, longitude, elevation_m,
///  timestamp, temperature_2m, dewpoint_2m, precipitation, rain,
///  snowfall, windspeed_10m, windgusts_10m, winddirection_10m,
///  pressure_msl, surface_pressure, cloudcover, shortwave_radiation,
///  relativehumidity_2m, weathercode`.
///
/// Note: `apparent_temperature` and `direct_radiation` are deliberately
/// **dropped** here to match the Nb02 structural cleanup step.
pub fn response_to_dataframe(
    city: &CityConfig,
    resp: &ForecastResponse,
) -> Result<DataFrame> {
    let n = resp.hourly.time.len();
    let extract = |o: &Option<Vec<Option<f64>>>| -> Vec<Option<f64>> {
        o.as_ref().map(|v| v.clone()).unwrap_or_else(|| vec![None; n])
    };
    let extract_i = |o: &Option<Vec<Option<i64>>>| -> Vec<Option<i64>> {
        o.as_ref().map(|v| v.clone()).unwrap_or_else(|| vec![None; n])
    };

    let city_names: Vec<&str>   = vec![city.name.as_str(); n];
    let country:    Vec<&str>   = vec![city.country_code.as_str(); n];
    let climate:    Vec<&str>   = vec![city.climate_zone; n];
    let lats:       Vec<f64>    = vec![city.latitude; n];
    let lons:       Vec<f64>    = vec![city.longitude; n];
    let elevs:      Vec<f64>    = vec![city.elevation_m; n];
    let ts:         Vec<&str>   = resp.hourly.time.iter().map(|s| s.as_str()).collect();

    // Apply the two structural drops from Nb02:
    // - we never even populate apparent_temperature / direct_radiation columns.
    // Plus the gust clipping: gust = max(gust, windspeed) applied later in
    // `apply_cleanup_and_clip` so nulls still go through fill.

    let df = df![
        "city"                => city_names,
        "country_code"        => country,
        "climate_zone"        => climate,
        "latitude"            => lats,
        "longitude"           => lons,
        "elevation_m"         => elevs,
        "timestamp"           => ts,
        "temperature_2m"      => extract(&resp.hourly.temperature_2m),
        "dewpoint_2m"         => extract(&resp.hourly.dewpoint_2m),
        "precipitation"       => extract(&resp.hourly.precipitation),
        "rain"                => extract(&resp.hourly.rain),
        "snowfall"            => extract(&resp.hourly.snowfall),
        "windspeed_10m"       => extract(&resp.hourly.windspeed_10m),
        "windgusts_10m"       => extract(&resp.hourly.windgusts_10m),
        "winddirection_10m"   => extract(&resp.hourly.winddirection_10m),
        "pressure_msl"        => extract(&resp.hourly.pressure_msl),
        "surface_pressure"    => extract(&resp.hourly.surface_pressure),
        "cloudcover"          => extract(&resp.hourly.cloudcover),
        "shortwave_radiation" => extract(&resp.hourly.shortwave_radiation),
        "relativehumidity_2m" => extract(&resp.hourly.relativehumidity_2m),
        "weathercode"         => extract_i(&resp.hourly.weathercode),
    ].context("build raw DataFrame")?;

    Ok(df)
}

/// Stack multiple per-city DataFrames into one.
pub fn stack_dataframes(mut parts: Vec<DataFrame>) -> Result<DataFrame> {
    if parts.is_empty() {
        return Err(anyhow!("no DataFrames to stack"));
    }
    let mut acc = parts.remove(0);
    for df in parts {
        acc = acc.vstack(&df).context("vstack")?;
    }
    acc.rechunk_mut();
    Ok(acc)
}

// --------------------------------------------------------------------------
// Section 1: cleanup + physical clips (Nb02 sections 2-4)
// --------------------------------------------------------------------------

fn clip_expr(name: &str, lo: f64, hi: f64) -> Expr {
    when(col(name).lt(lit(lo)))
        .then(lit(lo))
        .when(col(name).gt(lit(hi)))
        .then(lit(hi))
        .otherwise(col(name))
        .alias(name)
}

pub fn apply_cleanup_and_clip(df: DataFrame) -> Result<DataFrame> {
    // 1) Gust clipping: gust = max(gust, |U|)
    let df = df
        .lazy()
        .with_column(
            when(col("windgusts_10m").lt(col("windspeed_10m")))
                .then(col("windspeed_10m"))
                .otherwise(col("windgusts_10m"))
                .alias("windgusts_10m"),
        )
        .collect()
        .context("gust clipping")?;

    // 2) Per-city forward + backward fill on every numeric column.
    let numeric_cols = [
        "temperature_2m", "dewpoint_2m",
        "precipitation", "rain", "snowfall",
        "windspeed_10m", "windgusts_10m", "winddirection_10m",
        "pressure_msl", "surface_pressure", "cloudcover",
        "shortwave_radiation", "relativehumidity_2m",
    ];

    let df = df
        .lazy()
        .sort(["city", "timestamp"], Default::default())
        .with_columns(
            numeric_cols
                .iter()
                .map(|c| col(*c).forward_fill(None).backward_fill(None).over([col("city")]))
                .collect::<Vec<_>>(),
        )
        .with_columns([
            col("precipitation").fill_null(lit(0.0)),
            col("rain").fill_null(lit(0.0)),
            col("snowfall").fill_null(lit(0.0)),
            col("shortwave_radiation").fill_null(lit(0.0)),
            col("weathercode").fill_null(lit(0i64)),
        ])
        .collect()
        .context("per-city fill")?;

    // 3) Physical clips.
    let df = df
        .lazy()
        .with_columns([
            clip_expr("temperature_2m",      -60.0,   60.0),
            clip_expr("dewpoint_2m",         -60.0,   40.0),
            clip_expr("relativehumidity_2m",   0.0,  100.0),
            clip_expr("windspeed_10m",         0.0,  300.0),
            clip_expr("windgusts_10m",         0.0,  400.0),
            clip_expr("pressure_msl",        870.0, 1084.0),
            clip_expr("cloudcover",            0.0,  100.0),
            clip_expr("precipitation",         0.0,  500.0),
            clip_expr("shortwave_radiation",   0.0, 1500.0),
        ])
        .collect()
        .context("physical clips")?;

    Ok(df)
}

// --------------------------------------------------------------------------
// Section 2: temporal parsing via chrono + cyclic encoding
// --------------------------------------------------------------------------

pub fn apply_temporal_and_cyclic(df: DataFrame) -> Result<DataFrame> {
    // Parse timestamps once with chrono, build i32 columns.
    let ts_col = df.column("timestamp").unwrap().str().unwrap();
    let n = ts_col.len();
    let mut years = Vec::<i32>::with_capacity(n);
    let mut months = Vec::<i32>::with_capacity(n);
    let mut days = Vec::<i32>::with_capacity(n);
    let mut hours = Vec::<i32>::with_capacity(n);
    let mut dows = Vec::<i32>::with_capacity(n);
    let mut doys = Vec::<i32>::with_capacity(n);

    let fallback =
        NaiveDateTime::parse_from_str("1970-01-01T00:00", "%Y-%m-%dT%H:%M").unwrap();

    for opt in ts_col.into_iter() {
        let dt = opt
            .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M").ok())
            .unwrap_or(fallback);
        years.push(dt.year());
        months.push(dt.month() as i32);
        days.push(dt.day() as i32);
        hours.push(dt.hour() as i32);
        dows.push(dt.weekday().num_days_from_monday() as i32);
        doys.push(dt.ordinal() as i32);
    }

    let mut df = df;
    df.with_column(Series::new("year".into(),        years))?;
    df.with_column(Series::new("month".into(),       months))?;
    df.with_column(Series::new("day".into(),         days))?;
    df.with_column(Series::new("hour".into(),        hours))?;
    df.with_column(Series::new("day_of_week".into(), dows))?;
    df.with_column(Series::new("day_of_year".into(), doys))?;

    // Cyclic encodings — exactly as in Nb02.
    let df = df
        .lazy()
        .with_columns([
            ((col("hour").cast(DataType::Float64) * lit(2.0 * PI / 24.0)).sin()).alias("hour_sin"),
            ((col("hour").cast(DataType::Float64) * lit(2.0 * PI / 24.0)).cos()).alias("hour_cos"),
            ((col("day_of_week").cast(DataType::Float64) * lit(2.0 * PI / 7.0)).sin()).alias("dow_sin"),
            ((col("day_of_week").cast(DataType::Float64) * lit(2.0 * PI / 7.0)).cos()).alias("dow_cos"),
            (((col("month").cast(DataType::Float64) - lit(1.0)) * lit(2.0 * PI / 12.0)).sin()).alias("month_sin"),
            (((col("month").cast(DataType::Float64) - lit(1.0)) * lit(2.0 * PI / 12.0)).cos()).alias("month_cos"),
            (((col("day_of_year").cast(DataType::Float64) - lit(1.0)) * lit(2.0 * PI / 366.0)).sin()).alias("doy_sin"),
            (((col("day_of_year").cast(DataType::Float64) - lit(1.0)) * lit(2.0 * PI / 366.0)).cos()).alias("doy_cos"),
        ])
        .collect()
        .context("cyclic encoding")?;

    Ok(df)
}

// --------------------------------------------------------------------------
// Section 3: thermodynamic + wind + solar features
// --------------------------------------------------------------------------

pub fn apply_physical_features(df: DataFrame) -> Result<DataFrame> {
    let df = df
        .lazy()
        // Magnus-Tetens saturation vapor pressure (Bolton 1980).
        .with_columns([
            (lit(6.112)
                * (lit(17.67) * col("temperature_2m")
                    / (col("temperature_2m") + lit(243.5)))
                .exp())
            .alias("e_sat_T"),
            (lit(6.112)
                * (lit(17.67) * col("dewpoint_2m")
                    / (col("dewpoint_2m") + lit(243.5)))
                .exp())
            .alias("e_actual"),
        ])
        .with_columns([
            (col("e_sat_T") - col("e_actual")).alias("vpd"),
            (lit(0.622) * col("e_actual") / (col("pressure_msl") - col("e_actual")))
                .alias("mixing_ratio"),
        ])
        .with_columns([
            (col("mixing_ratio") / (lit(1.0) + col("mixing_ratio"))).alias("specific_humidity"),
            (col("temperature_2m") - col("dewpoint_2m")).alias("dewpoint_depression"),
        ])
        .with_columns([
            (lit(125.0) * col("dewpoint_depression")).alias("lcl_height_m"),
        ])
        // Wind vector + log transforms.
        .with_columns([
            (lit(-1.0)
                * col("windspeed_10m")
                * (col("winddirection_10m") * lit(PI / 180.0)).sin())
            .alias("u_wind"),
            (lit(-1.0)
                * col("windspeed_10m")
                * (col("winddirection_10m") * lit(PI / 180.0)).cos())
            .alias("v_wind"),
            (col("windgusts_10m") - col("windspeed_10m")).alias("gust_excess"),
            ((col("windspeed_10m") + lit(1.0)).log(std::f64::consts::E))
                .alias("log_windspeed"),
        ])
        .collect()
        .context("thermo + wind")?;

    // Solar geometry + clearness index.
    let lat = col("latitude") * lit(PI / 180.0);
    let doy = col("day_of_year").cast(DataType::Float64);
    let hr  = col("hour").cast(DataType::Float64);
    let decl = (lit(2.0 * PI) * (lit(284.0) + doy.clone()) / lit(365.0)).sin()
        * lit(23.45 * PI / 180.0);
    let hour_angle = (hr - lit(12.0)) * lit(15.0 * PI / 180.0);
    let cos_zenith = (lat.clone().sin() * decl.clone().sin())
        + (lat.cos() * decl.cos() * hour_angle.cos());
    let s0 = 1361.0_f64;

    let df = df
        .lazy()
        .with_columns([cos_zenith.clone().alias("cos_solar_zenith")])
        .with_columns([
            (when(col("cos_solar_zenith").lt(lit(0.0)))
                .then(lit(0.0))
                .otherwise(col("cos_solar_zenith") * lit(s0)))
            .alias("clear_sky_radiation"),
            (col("cos_solar_zenith").gt(lit(0.0))).alias("is_daytime"),
        ])
        .with_columns([
            (col("shortwave_radiation")
                / when(col("clear_sky_radiation").lt(lit(1.0)))
                    .then(lit(1.0))
                    .otherwise(col("clear_sky_radiation")))
            .alias("clearness_index_raw"),
        ])
        .with_columns([
            when(col("clearness_index_raw").lt(lit(0.0)))
                .then(lit(0.0))
                .when(col("clearness_index_raw").gt(lit(1.5)))
                .then(lit(1.5))
                .otherwise(col("clearness_index_raw"))
                .alias("clearness_index"),
        ])
        .drop(["clearness_index_raw"])
        .collect()
        .context("solar")?;

    Ok(df)
}

// --------------------------------------------------------------------------
// Section 4: lags + gradients + rolling + interactions + geo
// --------------------------------------------------------------------------

fn short_name(v: &str) -> &'static str {
    match v {
        "temperature_2m"      => "temp",
        "dewpoint_2m"         => "td",
        "pressure_msl"        => "pres",
        "windspeed_10m"       => "wind",
        "relativehumidity_2m" => "rh",
        "precipitation"       => "precip",
        _ => "x",
    }
}

pub fn apply_lags_gradients_rolling(df: DataFrame) -> Result<DataFrame> {
    // Lags — the hour offsets must match Notebook 02 exactly.
    let lag_specs: Vec<(&str, &[i64])> = vec![
        ("temperature_2m",     &[1, 3, 6, 12, 24, 48]),
        ("dewpoint_2m",        &[1, 6, 24]),
        ("pressure_msl",       &[1, 3, 6, 12, 24]),
        ("windspeed_10m",      &[1, 6, 24]),
        ("relativehumidity_2m",&[1, 6, 24]),
        ("precipitation",      &[1, 3, 6]),
    ];

    let mut lag_exprs: Vec<Expr> = Vec::new();
    for (var, ks) in &lag_specs {
        for k in *ks {
            let alias = format!("{}_lag{}h", short_name(var), k);
            lag_exprs.push(
                col(*var)
                    .shift(lit(*k as i64))
                    .over([col("city")])
                    .alias(alias),
            );
        }
    }

    let df = df
        .lazy()
        .sort(["city", "timestamp"], Default::default())
        .with_columns(lag_exprs)
        .collect()
        .context("lags")?;

    // Gradients.
    let df = df
        .lazy()
        .with_columns([
            (col("temperature_2m")      - col("temp_lag1h")).alias("temp_change_1h"),
            (col("temperature_2m")      - col("temp_lag3h")).alias("temp_change_3h"),
            (col("temperature_2m")      - col("temp_lag6h")).alias("temp_change_6h"),
            (col("temperature_2m")      - col("temp_lag24h")).alias("temp_change_24h"),
            (col("pressure_msl")        - col("pres_lag1h")).alias("pressure_change_1h"),
            (col("pressure_msl")        - col("pres_lag3h")).alias("pressure_change_3h"),
            (col("pressure_msl")        - col("pres_lag6h")).alias("pressure_change_6h"),
            (col("pressure_msl")        - col("pres_lag24h")).alias("pressure_change_24h"),
            (col("dewpoint_2m")         - col("td_lag1h")).alias("dewpoint_change_1h"),
            (col("dewpoint_2m")         - col("td_lag6h")).alias("dewpoint_change_6h"),
            (col("relativehumidity_2m") - col("rh_lag1h")).alias("rh_change_1h"),
            (col("windspeed_10m")       - col("wind_lag1h")).alias("wind_change_1h"),
        ])
        .collect()
        .context("gradients")?;

    // Rolling 24h statistics.
    let opts = RollingOptionsFixedWindow {
        window_size: 24,
        min_periods: 1,
        weights: None,
        center: false,
        fn_params: None,
    };
    let df = df
        .lazy()
        .sort(["city", "timestamp"], Default::default())
        .with_columns([
            col("temperature_2m").rolling_mean(opts.clone()).over([col("city")]).alias("temp_roll24_mean"),
            col("temperature_2m").rolling_std(opts.clone()).over([col("city")]).alias("temp_roll24_std"),
            col("temperature_2m").rolling_min(opts.clone()).over([col("city")]).alias("temp_roll24_min"),
            col("temperature_2m").rolling_max(opts.clone()).over([col("city")]).alias("temp_roll24_max"),
            col("pressure_msl").rolling_min(opts.clone()).over([col("city")]).alias("pres_roll24_min"),
            col("pressure_msl").rolling_max(opts.clone()).over([col("city")]).alias("pres_roll24_max"),
            col("precipitation").rolling_sum(opts.clone()).over([col("city")]).alias("precip_roll24_sum"),
        ])
        .with_column(
            (col("temp_roll24_max") - col("temp_roll24_min")).alias("temp_diurnal_range_24h"),
        )
        .collect()
        .context("rolling")?;

    Ok(df)
}

pub fn apply_interactions_and_geo(df: DataFrame) -> Result<DataFrame> {
    let df = df
        .lazy()
        .with_columns([
            (col("temperature_2m") * col("relativehumidity_2m") / lit(100.0)).alias("interact_t_rh"),
            (col("windspeed_10m") * col("dewpoint_depression")).alias("interact_wind_ddep"),
            (col("windspeed_10m") * col("vpd")).alias("interact_wind_vpd"),
        ])
        .collect()
        .context("interactions")?;

    // Geography + Köppen one-hot.
    let mut zone_exprs: Vec<Expr> = vec![col("latitude").abs().alias("abs_latitude")];
    for z in CLIMATE_ZONES.iter() {
        zone_exprs.push(
            when(col("climate_zone").eq(lit(*z)))
                .then(lit(1i32))
                .otherwise(lit(0i32))
                .alias(format!("climate_{}", z)),
        );
    }

    let df = df
        .lazy()
        .with_columns(zone_exprs)
        .collect()
        .context("geo + one-hot")?;

    // Log transforms used as features by the Ridge model (only
    // log1p_precipitation and log1p_windspeed — the third log feature in
    // Nb02 is derived from a future target and must NOT be referenced
    // from the features).
    let df = df
        .lazy()
        .with_columns([
            (col("precipitation") + lit(1.0)).log(std::f64::consts::E).alias("log1p_precipitation"),
            (col("windspeed_10m") + lit(1.0)).log(std::f64::consts::E).alias("log1p_windspeed"),
        ])
        .collect()
        .context("log transforms")?;

    Ok(df)
}

/// Orchestrator: apply the full pipeline end-to-end.
///
/// The input is expected to already be stacked across all cities (one
/// `DataFrame` with a `city` column). The output is a `DataFrame` with
/// every derived column — the Ridge feature vector can then be read by
/// selecting the columns named in `scaler.feature_names`.
pub fn run(df: DataFrame) -> Result<DataFrame> {
    let df = apply_cleanup_and_clip(df)?;
    let df = apply_temporal_and_cyclic(df)?;
    let df = apply_physical_features(df)?;
    let df = apply_lags_gradients_rolling(df)?;
    let df = apply_interactions_and_geo(df)?;
    Ok(df)
}

/// Build a single Ridge feature vector (in the exact order of
/// `feature_names`) from row `row_idx` of an already-engineered
/// `DataFrame`. Returns `None` if any referenced column is missing or any
/// value is null at that row.
pub fn feature_row(
    df: &DataFrame,
    row_idx: usize,
    feature_names: &[String],
) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(feature_names.len());
    for name in feature_names {
        let col = df
            .column(name.as_str())
            .with_context(|| format!("missing feature column {}", name))?;
        let f = col.cast(&DataType::Float64).context("cast to f64")?;
        let ca = f.f64().context("downcast f64")?;
        match ca.get(row_idx) {
            Some(v) if v.is_finite() => out.push(v),
            _ => return Err(anyhow!(
                "null or NaN in feature {} at row {}",
                name, row_idx
            )),
        }
    }
    Ok(out)
}

/// Lookup helper: the index of the most recent row whose lag-48h feature
/// is finite. This is the latest timestamp for which a full feature
/// vector can be constructed.
pub fn last_valid_row_for_city(df: &DataFrame, city: &str) -> Result<Option<usize>> {
    let cities_col = df.column("city")?.str()?;
    let lag48_col = df.column("temp_lag48h")?.f64()?;
    let timestamps = df.column("timestamp")?.str()?;
    // We want the greatest `timestamp` with non-null lag48 and city matches.
    // Assumes the DataFrame is already sorted by (city, timestamp).
    let mut best: Option<(usize, &str)> = None;
    for i in 0..df.height() {
        if cities_col.get(i) == Some(city) {
            if let Some(v) = lag48_col.get(i) {
                if v.is_finite() {
                    let ts = timestamps.get(i).unwrap_or("");
                    match best {
                        None => best = Some((i, ts)),
                        Some((_, best_ts)) if ts > best_ts => best = Some((i, ts)),
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(best.map(|(i, _)| i))
}

/// Lookup helper: map each city to a single `HashMap<String, usize>` of
/// column-name -> column-index. Used by the predictor to avoid repeated
/// linear lookups.
pub fn name_index(df: &DataFrame) -> HashMap<String, usize> {
    df.get_column_names()
        .iter()
        .enumerate()
        .map(|(i, n)| (n.to_string(), i))
        .collect()
}
