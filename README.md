# RustWeatherML

A production-grade machine learning system for weather prediction built entirely in Rust. This project demonstrates the complete ML lifecycle from data collection to model monitoring, using Evcxr Jupyter kernel for interactive exploration.

---

## 🌍 Live Weather Predictions

> Auto-updated every 3 h via GitHub Actions | Last run: 2026-11-18 11:23 UTC



<!-- BEGIN:LIVE_PREDICTIONS -->
### 24-Hour, 48-Hour & 72-Hour Forecast

| City | Country | As of (local) | Current | +24h | +48h | +72h | Rain 24h | NWP precip | ML only | Confidence |
|------|---------|---------------|---------|------|------|------|----------|------------|---------|------------|
| Dubai | 🇦🇪 | 23:00 | +26.0°C | +26.0°C | +26.1°C | +26.7°C |  14% | 0.0 mm (0h) |  96% | ±3.5°C |

> **How to read the rain columns.** `Rain 24h` is the **blended** probability that there will be any rain at all in the next 24 h, computed as `α·p_NWP + (1−α)·p_ML` with `α = 0.9`. The physical NWP signal dominates because the ML classifier was trained on a target (`precip_sum_next_24h > 0 mm`) that was 73.6 % positive in the training set and therefore has a positive prior bias. `NWP precip` shows the *actual* predicted rainfall in millimetres from Open-Meteo's numerical weather model, plus how many hours will have any precipitation. `ML only` shows the bagging-ensemble probability in isolation so you can see the drift. `+24h`, `+48h`, `+72h` come from dedicated Ridge (α=10) regressors trained in Notebook 05 on `temp_next_{24,48,72}h` with RMSE 3.5 / 4.5 / 5.1 °C. `Confidence` is ±1σ = ±RMSE of the 24 h test. `As of (local)` is the city's local timestamp of the last observed hour that anchors the forecast (end of the Open-Meteo past window, typically 23:00 of the previous day).

### Hourly Predictions (next 24 h, per city)

Click a city to expand. Each row is one hour; `temp` is our Ridge 24 h model rolled across the past 24 hours, `rain %` is the calibrated bagging probability, `precip` and `clouds` and `wind` come from Open-Meteo's own NWP forecast.

<details><summary><strong>🇦🇪 Dubai</strong> — BWh, Asia/Dubai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-18 00:00 | +25.8°C |  11% |   2% |  96% | 0.0 mm |  55% |  1.3 km/h | Clear |
| +2 | 2026-11-18 01:00 | +25.3°C |  11% |   2% |  96% | 0.0 mm |  51% |  1.5 km/h | Clear |
| +3 | 2026-11-18 02:00 | +25.0°C |  11% |   2% |  96% | 0.0 mm |  45% |  1.9 km/h | Clear |
| +4 | 2026-11-18 03:00 | +24.4°C |  11% |   2% |  96% | 0.0 mm |  48% |  2.6 km/h | Clear |
| +5 | 2026-11-18 04:00 | +23.6°C |  11% |   2% |  96% | 0.0 mm |  61% |  2.8 km/h | Clear |
| +6 | 2026-11-18 05:00 | +23.1°C |  11% |   2% |  96% | 0.0 mm |  61% |  3.6 km/h | Clear |
| +7 | 2026-11-18 06:00 | +22.4°C |  11% |   2% |  96% | 0.0 mm |  46% |  4.1 km/h | Clear |
| +8 | 2026-11-18 07:00 | +22.8°C |  10% |   2% |  81% | 0.0 mm |  44% |  3.9 km/h | Clear |
| +9 | 2026-11-18 08:00 | +23.9°C |  10% |   2% |  81% | 0.0 mm |  36% |  3.9 km/h | Clear |
| +10 | 2026-11-18 09:00 | +25.9°C |  11% |   2% |  96% | 0.0 mm |  13% |  4.4 km/h | Clear |
| +11 | 2026-11-18 10:00 | +28.1°C |  10% |   2% |  81% | 0.0 mm |   0% |  4.5 km/h | Clear |
| +12 | 2026-11-18 11:00 | +29.1°C |  11% |   2% |  96% | 0.0 mm |   3% |  4.6 km/h | Clear |
| +13 | 2026-11-18 12:00 | +29.9°C |  11% |   2% |  96% | 0.0 mm |  22% |  2.8 km/h | Clear |
| +14 | 2026-11-18 13:00 | +30.1°C |  11% |   2% |  96% | 0.0 mm |  65% |  1.9 km/h | Clear |
| +15 | 2026-11-18 14:00 | +29.9°C |  11% |   2% |  96% | 0.0 mm |  58% |  8.2 km/h | Clear |
| +16 | 2026-11-18 15:00 | +29.4°C |  11% |   2% |  96% | 0.0 mm |  92% | 12.1 km/h | Clear |
| +17 | 2026-11-18 16:00 | +28.7°C |  11% |   2% |  96% | 0.0 mm |  39% | 12.2 km/h | Clear |
| +18 | 2026-11-18 17:00 | +28.0°C |  11% |   2% |  96% | 0.0 mm |  48% | 10.0 km/h | Clear |
| +19 | 2026-11-18 18:00 | +27.3°C |  10% |   2% |  81% | 0.0 mm |  57% |  7.7 km/h | Clear |
| +20 | 2026-11-18 19:00 | +26.7°C |  11% |   2% |  96% | 0.0 mm |  79% |  4.7 km/h | Clear |
| +21 | 2026-11-18 20:00 | +26.3°C |  11% |   2% |  96% | 0.0 mm |  64% |  2.9 km/h | Clear |
| +22 | 2026-11-18 21:00 | +26.1°C |  11% |   2% |  96% | 0.0 mm |  67% |  1.8 km/h | Clear |
| +23 | 2026-11-18 22:00 | +26.0°C |  11% |   2% |  96% | 0.0 mm |  67% |  2.8 km/h | Clear |
| +24 | 2026-11-18 23:00 | +26.0°C |  11% |   2% |  96% | 0.0 mm |  63% |  3.4 km/h | Clear |

</details>


### Model Performance (held-out test set)

| Metric | Value |
|--------|-------|
| 24 h model | Ridge (alpha=10) |
| 48 h model | Ridge 48h (alpha=10) |
| 72 h model | Ridge 72h (alpha=10) |
| Rain model | Bagging ensemble (30 DT trees) + histogram calibration |
| Test RMSE (24 h) | 3.53 °C |
| Test RMSE (48 h) | 4.46 °C |
| Test RMSE (72 h) | 5.07 °C |
| Bias correction (24 h) | +0.69 °C |
| Contract version | 2.0.0 |
| Successful cities | 1 / 14 |

### Drift Monitor (last run)

⚠ Max PSI = 0.453  |  observations = 2352  |  cities = 14  |  status = drift detected

| Feature | PSI | KS | Mean shift (σ_ref) | Status |
|---------|-----|----|--------------------|--------|
| `clearness_index` | 0.075 | 0.056 | -0.11 | ok |
| `cloudcover` | 0.185 | 0.176 | -0.16 | moderate |
| `dewpoint_2m` | 0.335 | 0.174 | -0.24 | severe |
| `precipitation` | 0.000 | 0.004 | -0.04 | ok |
| `pressure_msl` | 0.263 | 0.110 | -0.14 | severe |
| `relativehumidity_2m` | 0.023 | 0.071 | -0.12 | ok |
| `temp_lag24h` | 0.218 | 0.128 | -0.17 | severe |
| `temperature_2m` | 0.204 | 0.142 | -0.19 | severe |
| `vpd` | 0.063 | 0.044 | -0.13 | ok |
| `windspeed_10m` | 0.453 | 0.247 | -0.54 | severe |

<!-- END:LIVE_PREDICTIONS -->

---

## 📋 Project Overview

### Features

- **Complete ML Pipeline**: Data collection → Preprocessing → Feature Engineering → Model Training → Evaluation → Monitoring
- **Multiple ML Libraries**: Side-by-side comparison of linfa, smartcore, rustyml, and Burn
- **10 Years of Data**: Historical weather data from 2016-2025 for 14 cities worldwide
- **4 Prediction Tasks**:
  - Rain prediction (binary classification)
  - Weather condition classification (6 classes)
  - Temperature forecasting (24h, 48h, 72h)
  - Multi-target forecasting (temp + humidity + wind)
- **Drift Detection**: Automated monitoring for data and concept drift
- **Live Dashboard**: Daily auto-updated predictions via GitHub Actions

### Data Source

- **API**: [Open-Meteo](https://open-meteo.com/) (free, unlimited, no API key required)
- **Historical Data**: 2016-2025 (10 years)
- **Granularity**: Hourly observations
- **Total Records**: ~1.2 million

### Cities Covered


| Region              | Cities                                                   |
| ------------------- | -------------------------------------------------------- |
| Brazil 🇧🇷         | São Paulo, Rio de Janeiro, São José dos Campos, Campinas |
| USA 🇺🇸            | New York, Los Angeles                                    |
| Europe 🇬🇧🇩🇪🇳🇴 | London, Berlin, Oslo                                     |
| Asia 🇯🇵🇨🇳🇦🇪   | Tokyo, Shanghai, Chongqing, Nanjing, Dubai               |


---

## 🏗️ Project Structure

```
RustForMachineLearning/
├── notebooks/                    # Jupyter notebooks (Evcxr)
│   ├── 01_data_collection_and_exploration.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_feature_selection_and_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_evaluation_and_validation.ipynb
│   └── 06_drift_detection_and_monitoring.ipynb
├── src/                          # Production Rust code
│   ├── lib.rs
│   ├── data/                     # Data loading & API client
│   ├── preprocessing/            # Data cleaning & transformation
│   ├── features/                 # Feature engineering & selection
│   ├── models/                   # ML model implementations
│   ├── training/                 # Training utilities
│   ├── evaluation/               # Metrics & visualization
│   ├── monitoring/               # Drift detection
│   └── bin/                      # CLI tools
├── data/                         # Data storage
│   ├── raw/                      # Raw API data
│   ├── processed/                # Cleaned data
│   └── features/                 # Engineered features
├── models/                       # Trained model artifacts
├── docs/                         # Documentation
└── .github/workflows/            # GitHub Actions
```

---

## 🚀 Getting Started

### Prerequisites

- Rust (1.70+)
- Jupyter with Evcxr kernel
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/SamoraDC/RustForMachineLearning.git
cd RustForMachineLearning

# Build the project
cargo build --release

# Install Evcxr Jupyter kernel (if not already installed)
cargo install evcxr_jupyter
evcxr_jupyter --install
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter lab

# Navigate to notebooks/ and open the notebooks in order
```

### Running Daily Predictions

```bash
cargo run --release --bin daily_predictions
```

---

## 📊 ML Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    DATA     │───▶│  PREPROC    │───▶│  FEATURES   │
│ COLLECTION  │    │ & WRANGLING │    │ ENGINEERING │
└─────────────┘    └─────────────┘    └─────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  Open-Meteo API   Missing values      Lag features
  14 cities        Outliers            Rolling stats
  10 years         Normalization       Cyclical encoding

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  FEATURE    │───▶│   MODEL     │───▶│  TRAINING   │
│ SELECTION   │    │ COMPARISON  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  Correlation       linfa            Train/Val/Test
  Importance        smartcore        Cross-validation
  Recursive elim    Burn             Early stopping

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   TUNING    │───▶│ EVALUATION  │───▶│ MONITORING  │
│             │    │             │    │ & DRIFT     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  Grid search       Accuracy/F1       Data drift
  Random search     RMSE/MAE          Concept drift
  Cross-val         ROC/AUC           Auto-retrain
```

---

## 📈 Model Performance

### Classification Tasks


| Model               | Rain (Acc) | Rain (F1) | Condition (Acc) | Condition (F1) |
| ------------------- | ---------- | --------- | --------------- | -------------- |
| Logistic Regression | --         | --        | --              | --             |
| Decision Tree       | --         | --        | --              | --             |
| Random Forest       | --         | --        | --              | --             |
| Gradient Boosting   | --         | --        | --              | --             |
| Neural Network      | --         | --        | --              | --             |
| **Ensemble**        | --         | --        | --              | --             |


### Regression Tasks


| Model             | Temp 24h (RMSE) | Temp 48h (RMSE) | Temp 72h (RMSE) |
| ----------------- | --------------- | --------------- | --------------- |
| Linear Regression | --              | --              | --              |
| Decision Tree     | --              | --              | --              |
| Random Forest     | --              | --              | --              |
| Gradient Boosting | --              | --              | --              |
| Neural Network    | --              | --              | --              |
| **Ensemble**      | --              | --              | --              |


---

## 🔧 Technologies

### ML Libraries


| Library                                                | Purpose       | Status |
| ------------------------------------------------------ | ------------- | ------ |
| [linfa](https://github.com/rust-ml/linfa)              | Classical ML  | ✅      |
| [smartcore](https://github.com/smartcorelib/smartcore) | Classical ML  | ✅      |
| [rustyml](https://github.com/rustyml/rustyml)          | Classical ML  | 🔄     |
| [Burn](https://github.com/tracel-ai/burn)              | Deep Learning | 🔄     |


### Data & Utilities

- **polars**: DataFrame operations
- **ndarray**: N-dimensional arrays
- **reqwest**: HTTP client
- **chrono**: Date/time handling
- **plotters**: Visualization
- **statrs**: Statistical functions

---

## 📚 Documentation

- [Design Document](docs/plans/2026-01-23-rust-weather-ml-design.md)
- [API Reference](docs/api.md) *(coming soon)*
- [Contributing Guide](CONTRIBUTING.md) *(coming soon)*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for providing free weather data
- [Rust ML Community](https://github.com/rust-ml) for the excellent ML libraries
- [Evcxr](https://github.com/evcxr/evcxr) for the Rust Jupyter kernel