# RustWeatherML

A production-grade machine learning system for weather prediction built entirely in Rust. This project demonstrates the complete ML lifecycle from data collection to model monitoring, using Evcxr Jupyter kernel for interactive exploration.

---

## 🌍 Live Weather Predictions

> Auto-updated daily at 06:00 UTC | Last run: *Pending first deployment*

### 24-Hour, 48-Hour & 72-Hour Forecast

| City | Country | Current | +24h | +48h | +72h | Rain % | Confidence |
|------|---------|---------|------|------|------|--------|------------|
| São Paulo | 🇧🇷 | --°C | --°C | --°C | --°C | --% | -- |
| Rio de Janeiro | 🇧🇷 | --°C | --°C | --°C | --°C | --% | -- |
| São José dos Campos | 🇧🇷 | --°C | --°C | --°C | --°C | --% | -- |
| Campinas | 🇧🇷 | --°C | --°C | --°C | --°C | --% | -- |
| New York | 🇺🇸 | --°C | --°C | --°C | --°C | --% | -- |
| Los Angeles | 🇺🇸 | --°C | --°C | --°C | --°C | --% | -- |
| London | 🇬🇧 | --°C | --°C | --°C | --°C | --% | -- |
| Berlin | 🇩🇪 | --°C | --°C | --°C | --°C | --% | -- |
| Oslo | 🇳🇴 | --°C | --°C | --°C | --°C | --% | -- |
| Tokyo | 🇯🇵 | --°C | --°C | --°C | --°C | --% | -- |
| Shanghai | 🇨🇳 | --°C | --°C | --°C | --°C | --% | -- |
| Chongqing | 🇨🇳 | --°C | --°C | --°C | --°C | --% | -- |
| Nanjing | 🇨🇳 | --°C | --°C | --°C | --°C | --% | -- |
| Dubai | 🇦🇪 | --°C | --°C | --°C | --°C | --% | -- |

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
