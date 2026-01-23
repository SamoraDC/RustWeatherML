//! Data loading and saving utilities.

use anyhow::{Result, Context};
use polars::prelude::*;
use std::path::Path;

use super::schema::WeatherRecord;

/// Save weather records to a Parquet file
pub fn save_to_parquet(records: &[WeatherRecord], path: &Path) -> Result<()> {
    let df = records_to_dataframe(records)?;

    let file = std::fs::File::create(path)
        .context("Failed to create parquet file")?;

    ParquetWriter::new(file)
        .finish(&mut df.clone())
        .context("Failed to write parquet file")?;

    Ok(())
}

/// Load weather records from a Parquet file
pub fn load_from_parquet(path: &Path) -> Result<DataFrame> {
    let file = std::fs::File::open(path)
        .context("Failed to open parquet file")?;

    ParquetReader::new(file)
        .finish()
        .context("Failed to read parquet file")
}

/// Save weather records to a CSV file
pub fn save_to_csv(records: &[WeatherRecord], path: &Path) -> Result<()> {
    let df = records_to_dataframe(records)?;

    let file = std::fs::File::create(path)
        .context("Failed to create CSV file")?;

    CsvWriter::new(file)
        .finish(&mut df.clone())
        .context("Failed to write CSV file")?;

    Ok(())
}

/// Load weather records from a CSV file
pub fn load_from_csv(path: &Path) -> Result<DataFrame> {
    let file = std::fs::File::open(path)
        .context("Failed to open CSV file")?;

    CsvReader::new(file)
        .finish()
        .context("Failed to read CSV file")
}

/// Convert WeatherRecord vector to Polars DataFrame
pub fn records_to_dataframe(records: &[WeatherRecord]) -> Result<DataFrame> {
    // Collect all fields into vectors
    let cities: Vec<&str> = records.iter().map(|r| r.city.as_str()).collect();
    let country_codes: Vec<&str> = records.iter().map(|r| r.country_code.as_str()).collect();
    let latitudes: Vec<f64> = records.iter().map(|r| r.latitude).collect();
    let longitudes: Vec<f64> = records.iter().map(|r| r.longitude).collect();
    let timestamps: Vec<i64> = records.iter()
        .map(|r| r.timestamp.and_utc().timestamp_millis())
        .collect();

    let temperature_2m: Vec<Option<f64>> = records.iter().map(|r| r.temperature_2m).collect();
    let apparent_temperature: Vec<Option<f64>> = records.iter().map(|r| r.apparent_temperature).collect();
    let dewpoint_2m: Vec<Option<f64>> = records.iter().map(|r| r.dewpoint_2m).collect();
    let precipitation: Vec<Option<f64>> = records.iter().map(|r| r.precipitation).collect();
    let rain: Vec<Option<f64>> = records.iter().map(|r| r.rain).collect();
    let snowfall: Vec<Option<f64>> = records.iter().map(|r| r.snowfall).collect();
    let windspeed_10m: Vec<Option<f64>> = records.iter().map(|r| r.windspeed_10m).collect();
    let windgusts_10m: Vec<Option<f64>> = records.iter().map(|r| r.windgusts_10m).collect();
    let winddirection_10m: Vec<Option<f64>> = records.iter().map(|r| r.winddirection_10m).collect();
    let pressure_msl: Vec<Option<f64>> = records.iter().map(|r| r.pressure_msl).collect();
    let surface_pressure: Vec<Option<f64>> = records.iter().map(|r| r.surface_pressure).collect();
    let cloudcover: Vec<Option<f64>> = records.iter().map(|r| r.cloudcover).collect();
    let visibility: Vec<Option<f64>> = records.iter().map(|r| r.visibility).collect();
    let shortwave_radiation: Vec<Option<f64>> = records.iter().map(|r| r.shortwave_radiation).collect();
    let direct_radiation: Vec<Option<f64>> = records.iter().map(|r| r.direct_radiation).collect();
    let relativehumidity_2m: Vec<Option<f64>> = records.iter().map(|r| r.relativehumidity_2m).collect();
    let weathercode: Vec<Option<i32>> = records.iter().map(|r| r.weathercode).collect();

    let df = df![
        "city" => cities,
        "country_code" => country_codes,
        "latitude" => latitudes,
        "longitude" => longitudes,
        "timestamp" => timestamps,
        "temperature_2m" => temperature_2m,
        "apparent_temperature" => apparent_temperature,
        "dewpoint_2m" => dewpoint_2m,
        "precipitation" => precipitation,
        "rain" => rain,
        "snowfall" => snowfall,
        "windspeed_10m" => windspeed_10m,
        "windgusts_10m" => windgusts_10m,
        "winddirection_10m" => winddirection_10m,
        "pressure_msl" => pressure_msl,
        "surface_pressure" => surface_pressure,
        "cloudcover" => cloudcover,
        "visibility" => visibility,
        "shortwave_radiation" => shortwave_radiation,
        "direct_radiation" => direct_radiation,
        "relativehumidity_2m" => relativehumidity_2m,
        "weathercode" => weathercode
    ].context("Failed to create DataFrame")?;

    Ok(df)
}

/// Get summary statistics for a DataFrame (basic implementation)
/// Returns a DataFrame with count, null_count for each column
pub fn describe_dataframe(df: &DataFrame) -> Result<String> {
    let mut result = String::new();
    result.push_str("DataFrame Summary:\n");
    result.push_str(&format!("  Rows: {}\n", df.height()));
    result.push_str(&format!("  Columns: {}\n", df.width()));
    result.push_str("\nColumn Info:\n");

    for col in df.get_columns() {
        let null_count = col.null_count();
        let dtype = col.dtype();
        result.push_str(&format!(
            "  {} - {:?} ({} nulls)\n",
            col.name(),
            dtype,
            null_count
        ));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDateTime;
    use tempfile::tempdir;

    fn create_test_records() -> Vec<WeatherRecord> {
        vec![
            WeatherRecord {
                city: "Test City".to_string(),
                country_code: "TC".to_string(),
                latitude: 0.0,
                longitude: 0.0,
                timestamp: NaiveDateTime::parse_from_str("2024-01-01T00:00", "%Y-%m-%dT%H:%M").unwrap(),
                temperature_2m: Some(25.0),
                apparent_temperature: Some(26.0),
                dewpoint_2m: Some(20.0),
                precipitation: Some(0.0),
                rain: Some(0.0),
                snowfall: Some(0.0),
                windspeed_10m: Some(10.0),
                windgusts_10m: Some(15.0),
                winddirection_10m: Some(180.0),
                pressure_msl: Some(1013.0),
                surface_pressure: Some(1010.0),
                cloudcover: Some(50.0),
                visibility: Some(10000.0),
                shortwave_radiation: Some(500.0),
                direct_radiation: Some(300.0),
                relativehumidity_2m: Some(70.0),
                weathercode: Some(1),
            }
        ]
    }

    #[test]
    fn test_records_to_dataframe() {
        let records = create_test_records();
        let df = records_to_dataframe(&records).unwrap();
        assert_eq!(df.height(), 1);
        assert!(df.column("temperature_2m").is_ok());
    }

    #[test]
    fn test_save_and_load_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");

        let records = create_test_records();
        save_to_parquet(&records, &path).unwrap();

        let loaded = load_from_parquet(&path).unwrap();
        assert_eq!(loaded.height(), 1);
    }
}
