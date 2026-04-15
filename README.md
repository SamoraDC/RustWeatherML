# RustWeatherML

A production-grade machine learning system for weather prediction built entirely in Rust. This project demonstrates the complete ML lifecycle from data collection to model monitoring, using Evcxr Jupyter kernel for interactive exploration.

---

## 🌍 Live Weather Predictions

> Auto-updated daily at 06:00 UTC | Last run: 2026-04-15 11:55 UTC

### 24-Hour, 48-Hour & 72-Hour Forecast

| City | Country | Current | +24h | +48h | +72h | Rain % | Confidence |
|------|---------|---------|------|------|------|--------|------------|
| Sao Paulo | 🇧🇷 | +18.3°C | +19.0°C | +19.6°C | +21.9°C |  85% | ±3.4°C |
| Rio de Janeiro | 🇧🇷 | +22.1°C | +22.8°C | +23.0°C | +22.7°C |  85% | ±3.4°C |
| Sao Jose dos Campos | 🇧🇷 | +18.0°C | +18.4°C | +18.5°C | +17.6°C |  15% | ±3.4°C |
| Campinas | 🇧🇷 | +18.8°C | +19.5°C | +17.6°C | +19.6°C |  85% | ±3.4°C |
| New York | 🇺🇸 | +24.5°C | +21.2°C | +24.1°C | +19.6°C |  85% | ±3.4°C |
| Los Angeles | 🇺🇸 | +14.6°C | +16.1°C | +16.7°C | +20.8°C |  15% | ±3.4°C |
| London | 🇬🇧 | +14.5°C | +14.4°C | +12.5°C | +13.2°C |  85% | ±3.4°C |
| Berlin | 🇩🇪 | +9.4°C | +12.0°C | +15.3°C | +12.4°C |  85% | ±3.4°C |
| Oslo | 🇳🇴 | +8.7°C | +9.5°C | +8.9°C | +7.2°C |  85% | ±3.4°C |
| Tokyo | 🇯🇵 | +14.3°C | +15.7°C | +10.4°C | +11.5°C |  85% | ±3.4°C |
| Shanghai | 🇨🇳 | +15.3°C | +15.8°C | +16.0°C | +15.4°C |  85% | ±3.4°C |
| Chongqing | 🇨🇳 | +17.3°C | +18.5°C | +18.6°C | +20.8°C |  85% | ±3.4°C |
| Nanjing | 🇨🇳 | +16.3°C | +16.5°C | +16.6°C | +19.0°C |  85% | ±3.4°C |
| Dubai | 🇦🇪 | +24.3°C | +24.9°C | +23.3°C | +24.1°C |  85% | ±3.4°C |

> **Source of each horizon.** `+24h` comes from the Ridge (alpha=10) model we trained in Notebook 05 (post-hoc bias-corrected). `+48h` and `+72h` are taken directly from Open-Meteo's own NWP forecast, since we did not train dedicated models for those horizons. `Rain %` is the rain probability of the RandomForest classifier (high band 85% / low band 15% via the reliability curve in Nb05). `Confidence` is ±1 sigma = ±RMSE on the held-out test.

### Model Performance (held-out test set)

| Metric | Value |
|--------|-------|
| Model | Ridge (alpha=10) |
| Test RMSE (24 h temperature) | 3.41 °C |
| 95 % CI of RMSE | [3.32, 3.50] °C |
| Skill vs persistence-24 h | +0.264 |
| Post-hoc bias correction  | +0.68 °C |
| Successful cities         | 14 / 14 |


### Model Performance (held-out test set)

| Metric | Value |
|--------|-------|
| Model | Ridge (alpha=10) |
| Test RMSE (24 h temperature) | 3.41 °C |
| 95 % CI of RMSE | [3.32, 3.50] °C |
| Skill vs persistence-24 h | +0.264 |
| Post-hoc bias correction  | +0.68 °C |
| Successful cities         | 14 / 14 |


### Model Performance (Last 7 Days)

| Metric | Rain Prediction | Condition | Temp 24h | Temp 48h | Temp 72h |
|--------|-----------------|-----------|----------|----------|----------|
| Accuracy/RMSE | --% | --% | --°C | --°C | --°C |
| vs Baseline | -- | -- | -- | -- | -- |

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

| Region | Cities |
|--------|--------|
| Brazil 🇧🇷 | São Paulo, Rio de Janeiro, São José dos Campos, Campinas |
| USA 🇺🇸 | New York, Los Angeles |
| Europe 🇬🇧🇩🇪🇳🇴 | London, Berlin, Oslo |
| Asia 🇯🇵🇨🇳🇦🇪 | Tokyo, Shanghai, Chongqing, Nanjing, Dubai |

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
git clone https://github.com/yourusername/RustForMachineLearning.git
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

| Model | Rain (Acc) | Rain (F1) | Condition (Acc) | Condition (F1) |
|-------|------------|-----------|-----------------|----------------|
| Logistic Regression | -- | -- | -- | -- |
| Decision Tree | -- | -- | -- | -- |
| Random Forest | -- | -- | -- | -- |
| Gradient Boosting | -- | -- | -- | -- |
| Neural Network | -- | -- | -- | -- |
| **Ensemble** | -- | -- | -- | -- |

### Regression Tasks

| Model | Temp 24h (RMSE) | Temp 48h (RMSE) | Temp 72h (RMSE) |
|-------|-----------------|-----------------|-----------------|
| Linear Regression | -- | -- | -- |
| Decision Tree | -- | -- | -- |
| Random Forest | -- | -- | -- |
| Gradient Boosting | -- | -- | -- |
| Neural Network | -- | -- | -- |
| **Ensemble** | -- | -- | -- |

---

## 🔧 Technologies

### ML Libraries

| Library | Purpose | Status |
|---------|---------|--------|
| [linfa](https://github.com/rust-ml/linfa) | Classical ML | ✅ |
| [smartcore](https://github.com/smartcorelib/smartcore) | Classical ML | ✅ |
| [rustyml](https://github.com/rustyml/rustyml) | Classical ML | 🔄 |
| [Burn](https://github.com/tracel-ai/burn) | Deep Learning | 🔄 |

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