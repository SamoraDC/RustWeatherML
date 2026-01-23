# RustWeatherML - Complete Weather Prediction System in Rust

## Design Document
**Date**: 2026-01-23
**Status**: Approved

---

## 1. Project Overview

A production-grade machine learning system built entirely in Rust for comprehensive weather prediction, featuring live daily predictions displayed on GitHub README. Uses Evcxr Jupyter kernel for interactive exploration and experimentation.

### Goals
- Demonstrate full ML lifecycle in Rust (from data collection to model monitoring)
- Compare multiple Rust ML libraries (linfa, smartcore, rustyml, Burn)
- Build production-ready weather prediction models
- Create live auto-updating predictions dashboard

---

## 2. Data Specification

### Source
- **API**: Open-Meteo (free, unlimited, no API key required)
- **Historical endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Forecast endpoint**: `https://api.open-meteo.com/v1/forecast`

### Dataset Details
| Attribute | Value |
|-----------|-------|
| Time Range | 2016-2025 (10 years) |
| Granularity | Hourly |
| Cities | 14 |
| Estimated Records | ~1.2 million |

### Cities (14 locations across 4 continents)

| City | Country | Latitude | Longitude | Climate |
|------|---------|----------|-----------|---------|
| São Paulo | Brazil | -23.55 | -46.63 | Subtropical |
| Rio de Janeiro | Brazil | -22.91 | -43.17 | Tropical |
| São José dos Campos | Brazil | -23.18 | -45.88 | Subtropical |
| Campinas | Brazil | -22.91 | -47.06 | Subtropical |
| New York | USA | 40.71 | -74.01 | Continental |
| Los Angeles | USA | 34.05 | -118.24 | Mediterranean |
| London | UK | 51.51 | -0.13 | Oceanic |
| Berlin | Germany | 52.52 | 13.40 | Continental |
| Oslo | Norway | 59.91 | 10.75 | Continental/Cold |
| Tokyo | Japan | 35.68 | 139.69 | Humid subtropical |
| Shanghai | China | 31.23 | 121.47 | Humid subtropical |
| Chongqing | China | 29.56 | 106.55 | Humid subtropical |
| Nanjing | China | 32.06 | 118.80 | Humid subtropical |
| Dubai | UAE | 25.27 | 55.30 | Hot desert |

### Features to Collect

| Category | Variables |
|----------|-----------|
| Temperature | temperature_2m, apparent_temperature, dewpoint_2m |
| Precipitation | precipitation, rain, snowfall, precipitation_probability |
| Wind | windspeed_10m, windgusts_10m, winddirection_10m |
| Atmosphere | pressure_msl, surface_pressure, cloudcover, visibility |
| Radiation | shortwave_radiation, direct_radiation, uv_index |
| Humidity | relativehumidity_2m |
| Weather Code | weathercode (WMO standard) |

### Data Split Strategy
- **Training**: 2016-2023 (8 years) - ~980K records
- **Validation**: 2024 (1 year) - ~122K records
- **Test**: 2025 (1 year) - ~122K records

*Temporal split (not random) to prevent data leakage*

---

## 3. ML Tasks

### Task 1: Rain Prediction (Binary Classification)
- **Output**: Yes/No + probability
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC

### Task 2: Weather Condition (Multi-class Classification)

| Class | WMO Codes | Description |
|-------|-----------|-------------|
| Clear | 0, 1 | Clear sky, mainly clear |
| Cloudy | 2, 3 | Partly cloudy, overcast |
| Foggy | 45, 48 | Fog, depositing rime fog |
| Rainy | 51-67, 80-82 | Drizzle, rain, showers |
| Snowy | 71-77, 85-86 | Snow, snow grains, showers |
| Stormy | 95-99 | Thunderstorm, hail |

- **Metrics**: Accuracy, Macro/Weighted F1, Confusion Matrix

### Task 3: Temperature Forecasting (Regression)

| Horizon | Target Variable |
|---------|-----------------|
| 24h | temp_next_24h |
| 48h | temp_next_48h |
| 72h | temp_next_72h |

- **Metrics**: RMSE, MAE, R², MAPE

### Task 4: Multi-target Forecasting

| Target | Unit | Range |
|--------|------|-------|
| Temperature | °C | -40 to 50 |
| Humidity | % | 0 to 100 |
| Wind Speed | km/h | 0 to 200 |

- **Metrics**: Per-target RMSE/MAE, Combined weighted score

---

## 4. Model Comparison

### Libraries to Compare

| Library | Focus | Algorithms |
|---------|-------|------------|
| linfa | Classical ML | Linear/Logistic Regression, Decision Tree, Random Forest |
| smartcore | Classical ML | All above + Gradient Boosting |
| rustyml | Classical ML | All classical algorithms |
| Burn | Deep Learning | Neural Networks (MLP) |

### Model Comparison Matrix

| Model | linfa | smartcore | rustyml | Burn |
|-------|-------|-----------|---------|------|
| Linear/Logistic Regression | ✓ | ✓ | ✓ | ✓ |
| Decision Tree | ✓ | ✓ | ✓ | - |
| Random Forest | ✓ | ✓ | ✓ | - |
| Gradient Boosting | - | ✓ | ✓ | - |
| Neural Network | - | - | - | ✓ |

---

## 5. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RUST WEATHER ML PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 1. DATA      │───▶│ 2. PREPROC   │───▶│ 3. FEATURE   │                  │
│  │ COLLECTION   │    │ & WRANGLING  │    │ ENGINEERING  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│        │                    │                    │                          │
│        ▼                    ▼                    ▼                          │
│  - Open-Meteo API    - Missing values      - Lag features (1h-24h)         │
│  - 14 cities         - Outlier detection   - Rolling statistics            │
│  - 10 years data     - Type conversion     - Cyclical encoding             │
│  - Hourly granular   - Normalization       - City embeddings               │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 4. FEATURE   │───▶│ 5. MODEL     │───▶│ 6. TRAINING  │                  │
│  │ SELECTION    │    │ COMPARISON   │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│        │                    │                    │                          │
│        ▼                    ▼                    ▼                          │
│  - Correlation anal  - linfa             - Train/Val/Test split            │
│  - Mutual information- smartcore         - Cross-validation                │
│  - Recursive elim    - rustyml           - Early stopping                  │
│  - Importance scores - Burn              - Learning curves                 │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 7. HYPER-    │───▶│ 8. EVAL &    │───▶│ 9. MONITOR   │                  │
│  │ PARAM TUNING │    │ VALIDATION   │    │ & DRIFT      │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│        │                    │                    │                          │
│        ▼                    ▼                    ▼                          │
│  - Grid search       - Accuracy/F1       - Data drift detection            │
│  - Random search     - RMSE/MAE          - Concept drift                   │
│  - Cross-validation  - Confusion matrix  - Performance decay               │
│                      - ROC/AUC curves    - Retraining triggers             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Project Structure

```
RustForMachineLearning/
├── Cargo.toml
├── README.md (with live dashboard)
├── .github/
│   └── workflows/
│       └── daily_predictions.yml
├── notebooks/
│   ├── 01_data_collection_and_exploration.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_feature_selection_and_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_evaluation_and_validation.ipynb
│   └── 06_drift_detection_and_monitoring.ipynb
├── src/
│   ├── lib.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── api.rs
│   │   ├── loader.rs
│   │   └── schema.rs
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── missing.rs
│   │   ├── outliers.rs
│   │   ├── normalization.rs
│   │   └── encoding.rs
│   ├── features/
│   │   ├── mod.rs
│   │   ├── temporal.rs
│   │   ├── cyclical.rs
│   │   ├── selection.rs
│   │   └── engineering.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── traits.rs
│   │   ├── linear.rs
│   │   ├── tree.rs
│   │   ├── forest.rs
│   │   ├── boosting.rs
│   │   ├── neural.rs
│   │   └── ensemble.rs
│   ├── training/
│   │   ├── mod.rs
│   │   ├── split.rs
│   │   ├── cross_val.rs
│   │   └── tuning.rs
│   ├── evaluation/
│   │   ├── mod.rs
│   │   ├── classification.rs
│   │   ├── regression.rs
│   │   └── visualization.rs
│   ├── monitoring/
│   │   ├── mod.rs
│   │   ├── drift.rs
│   │   └── performance.rs
│   └── bin/
│       └── daily_predictions.rs
├── models/
│   ├── rain_classifier.bin
│   ├── condition_classifier.bin
│   ├── temperature_regressor.bin
│   └── multi_target_regressor.bin
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
└── tests/
    ├── integration/
    └── unit/
```

---

## 7. Drift Detection & Monitoring

### Data Drift Detection Methods

| Method | Description |
|--------|-------------|
| KL Divergence | Measures distribution difference |
| Kolmogorov-Smirnov Test | Statistical test for distribution shift |
| Population Stability Index (PSI) | Binned distribution comparison |
| Feature Mean/Std Tracking | Rolling window comparison |

### Concept Drift Detection Methods

| Method | Description |
|--------|-------------|
| ADWIN | Adaptive windowing |
| Page-Hinkley Test | Mean shift detection |
| Performance Decay | Track accuracy over time |
| Prediction Confidence | Monitor model certainty |

### Automated Retraining Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Accuracy drop | > 5% from baseline | Alert + schedule retrain |
| PSI score | > 0.2 any feature | Investigate + possible retrain |
| Consecutive misses | > 10 days wrong predictions | Force retrain |
| Time-based | Every 90 days | Scheduled maintenance retrain |

---

## 8. Live Dashboard

### GitHub Actions Workflow
- **Schedule**: Daily at 06:00 UTC
- **Trigger**: Also supports manual dispatch

### Dashboard Features
- 14 cities predictions
- 24h, 48h, 72h forecast horizons
- Weather condition emojis
- Rain probability
- Model confidence scores
- 7-day rolling performance metrics

### Weather Emojis

| Condition | Emoji |
|-----------|-------|
| Clear | ☀️ |
| Partly Cloudy | 🌤️ |
| Cloudy | ⛅ |
| Foggy | 🌫️ |
| Rainy | 🌧️ |
| Snowy | 🌨️ |
| Heavy Snow | ❄️ |
| Stormy | ⛈️ |

---

## 9. Dependencies

### Cargo.toml

```toml
[dependencies]
# ML Libraries
linfa = "0.7"
smartcore = "0.3"
burn = "0.13"

# Data handling
polars = { version = "0.35", features = ["lazy", "parquet"] }
ndarray = "0.15"

# API & Serialization
reqwest = { version = "1.0", features = ["json", "blocking"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Utilities
chrono = "0.4"
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
plotters = "0.3"
statrs = "0.16"
```

---

## 10. Implementation Phases

### Phase 1: Foundation Setup
- Initialize Cargo project
- Setup Evcxr Jupyter kernel
- Configure dependencies
- Create directory structure

### Phase 2: Data Pipeline (Notebooks 01-02)
- Implement Open-Meteo API client
- Fetch historical data
- EDA and preprocessing
- Feature engineering

### Phase 3: Modeling (Notebooks 03-04)
- Implement model traits
- Train all algorithms
- Hyperparameter tuning
- Model comparison

### Phase 4: Evaluation (Notebook 05)
- Metrics implementation
- Visualization
- Error analysis
- Model selection

### Phase 5: Production (Notebook 06 + src/)
- Drift detection
- Monitoring framework
- GitHub Actions
- Live dashboard
