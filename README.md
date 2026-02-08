# German Electricity Price Forecasting: An End-to-End MLOps Project

![Project Status: In Progress](https://img.shields.io/badge/status-in%20progress-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.17-blue.svg)

This repository documents the development of a production-ready MLOps pipeline for forecasting day-ahead wholesale electricity prices in the German market. The goal of this project is to build a complete, automated system that demonstrates practical skills in MLOps and cloud engineering, moving beyond academic concepts to a deployable solution.

**Note:** This is an active project currently in its initial development phase. The features and architecture described below represent the final goal of the 8-milestone development plan.

## 🎯 The Business Problem

The volatility of electricity prices is a major challenge for energy traders, grid operators, and industrial consumers. Accurate short-term price forecasting is critical for optimizing energy consumption, managing costs, and making strategic resource allocation decisions. This project will address this real-world need by building an automated system to predict future prices.

The data is sourced from **SMARD.de**, the official market data platform of the German Federal Network Agency (Bundesnetzagentur), ensuring relevance and credibility within the German market.

---

## 🚧 Project Roadmap

This roadmap outlines the key milestones and will be updated as the project progresses.

* [x] **Milestone 1:** Project Scaffolding, Data Ingestion & Complete EDA with Feature Engineering
* [x] **Milestone 2:** Baseline & Tuned Modeling with MLflow Experiment Tracking
* [x] **Milestone 3:** FastAPI Inference Service + Containerization (Docker)
* [ ] **Milestone 4:** Automation with CI/CT Pipeline (GitHub Actions)
* [ ] **Milestone 5:** Advanced Modeling & Feature Enrichment (XGBoost)
* [ ] **Milestone 6:** Cloud Deployment on Microsoft Azure (Azure App Service)
* [ ] **Milestone 7:** Project Showcase & Narrative Crafting
* [ ] **Milestone 8:** Strategic Application & Interview Mastery

---

## 🔧 Planned Technical Architecture

The planned architecture is designed for automation, reproducibility, and scalability, incorporating modern MLOps best practices.

**Target Tech Stack:**

* **Language & Libraries**: Python, Pandas, Scikit-learn, XGBoost, LightGBM
* **Experiment Tracking**: MLflow
* **Containerization**: Docker & Docker Compose
* **CI/CT Automation**: GitHub Actions (planned)
* **API Development**: FastAPI with Uvicorn
* **Cloud Platform**: Microsoft Azure (App Service, Container Registry)

### Implemented MLOps Features

* **📊 Experiment Tracking**: ✅ All model training runs logged with **MLflow**, capturing code versions, parameters, and metrics for full auditability.
* **📦 Reproducibility**: ✅ The entire application containerized with **Docker** to guarantee consistent environments for training and deployment.
* **🐳 Orchestration**: ✅ **Docker Compose** service defined for coordinating train, MLflow UI, and inference API containers.
* **🚀 API Inference**: ✅ **FastAPI** REST API implemented with health checks, Pydantic validation, and interactive documentation.
* **⚙️ Automation (CI/CT)**: GitHub Actions workflow planned for automated testing and weekly model retraining (Milestone 4).
* **☁️ Cloud Deployment**: Azure deployment planned for Milestone 6 (currently runs locally via Docker).

---

## 📁 Project Structure

```text
├── LICENSE
├── README.md
├── requirements.txt
├── docker-compose.yml           # Docker Compose orchestration (train, mlflow, api services)
├── Dockerfile.train             # Training image for model pipeline
├── Dockerfile.inference         # Lightweight inference image for FastAPI
├── data/
│   ├── interim/             # Clean intermediate datasets with metadata
│   │   ├── actual_generation_clean.metadata.json
│   │   ├── actual_generation_clean.parquet
│   │   ├── day_ahead_prices_clean.metadata.json
│   │   └── day_ahead_prices_clean.parquet
│   ├── processed/           # Model-ready feature datasets with train/test splits and artifacts
│   │   ├── best_model_random_forest_tuned_v1_20251009.joblib
│   │   ├── best_model_xgboost_v1_20251009.joblib
│   │   ├── best_model_xgboost_tuned_v1_20251009.joblib
│   │   ├── features_v1_20251009.metadata.json
│   │   ├── features_v1_20251009.parquet
│   │   ├── model_results_v1_20251009.csv
│   │   ├── test_v1_20251009.metadata.json
│   │   ├── test_v1_20251009.parquet
│   │   ├── train_v1_20251009.metadata.json
│   │   └── train_v1_20251009.parquet
│   └── raw/                 # Raw data files from SMARD.de
│       ├── Actual_generation_202101010000_202509180000_Hour.csv
│       └── Day-ahead_prices_202101010000_202509180000_Hour.csv
├── images/                      # Visualizations and plots (model diagnostics, MLflow screenshots)
│   ├── model_predictions_*.png
│   ├── model_timeseries_*.png
│   ├── MLflow-Experiment-Tracking.png
│   └── FastAPI-deployment.png
├── mlruns/                      # MLflow experiment tracking directory
├── notebooks/               # Jupyter notebooks (01-EDA.ipynb for EDA)
├── scripts/                 # Utility and pipeline scripts (currently empty)
├── src/                     # Source code for the project
│   ├── train.py            # Model training pipeline with visualization generation
│   └── api.py              # FastAPI inference service with Pydantic validation
└── tests/                   # Unit and integration tests (currently empty)
```

---

## 🛠️ Setup

**Milestone 1-3 Complete:** Comprehensive EDA, baseline modeling with MLflow tracking, and containerized inference API fully implemented. The remaining pipeline (CI/CT automation, Azure cloud deployment) is planned for upcoming weeks.

### Quick Start with Docker Compose

The easiest way to run the full pipeline (training, MLflow UI, and API server) is with Docker Compose:

```bash
# Start all services (training, MLflow UI, FastAPI)
docker compose up -d

# View MLflow UI at http://localhost:5000
# Access API documentation at http://localhost:8000/docs
# Access API at http://localhost:8000

# Stop all services
docker compose down
```

### Local Setup (Without Docker)

1. **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Train models:**

    ```bash
    python src/train.py
    ```

5. **Launch the API:**

    ```bash
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    ```

6. **View MLflow results:**

    ```bash
    mlflow ui --backend-store-uri file:./mlruns
    ```

---

## 🚩 Challenges Encountered & Solutions

### Lagged Time Series Features ✅ **SOLVED**

**Challenge:** When I started visualizing the data in my EDA, I realized I needed to use lagged features (previous values in the time series) for forecasting. I didn't fully understand how to create and use these features properly.

**Solution:** Successfully implemented comprehensive lag feature engineering including:

* Previous day prices (24h lag) and weekly prices (168h lag)
* Rolling statistics (3-day moving averages, 24-hour standard deviations)
* Cross-border price spreads and relative spreads
* Renewable energy ratio dynamics with temporal interactions
* Autocorrelation analysis validated 24h and 168h cycles

## 📊 Milestone 1 Accomplishments

### Data Processing Pipeline

* ✅ **Clean Data Layer**: Processed 41K+ hourly records (2021-2025) with proper datetime indexing
* ✅ **Feature Engineering**: Created 20+ engineered features including temporal, lag, and derived features
* ✅ **Data Quality**: Handled missing values, German nuclear phase-out gaps, and data type conversions
* ✅ **Train/Test Split**: Prepared model-ready datasets with temporal split (70/30 around June 2024)

### Key Insights Discovered

* **Market Integration**: Neighboring country prices show strongest correlation (~0.9) with German prices
* **Renewable Impact**: Higher renewable generation correlates with lower prices (correlation ~-0.4)
* **Temporal Patterns**: Clear 24-hour and weekly cycles validated through autocorrelation analysis
* **Seasonality**: Winter and autumn months show significantly higher price medians
* **Negative Prices**: Identified rare but important supply-excess periods in the data

### Technical Implementation

* **Metadata Tracking**: JSON metadata files accompany all processed datasets
* **Parquet Storage**: Efficient columnar storage for all intermediate and final datasets
* **Temporal Features**: Cyclic encoding for hour, day, month, and seasonal patterns
* **Validation**: Autocorrelation analysis confirmed lag feature selection strategy

---

## 📊 Milestone 2 Accomplishments

### Baseline & Tuned Models

* ✅ **Multi-Model Comparison**: 6 models trained (Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM)
* ✅ **Standardized Pipeline**: Scikit-learn pipelines with scaling to prevent leakage across splits
* ✅ **Hyperparameter Tuning**: RandomizedSearchCV applied to XGBoost (best validator); tuned model saved
* ✅ **MLflow Experiment Tracking**: All 7 runs (6 baseline + 1 tuned) logged with metrics, params, and models to `mlruns/`
  * Run names: `ridge_baseline`, `lasso_baseline`, `decision_tree_baseline`, `random_forest_baseline`, `xgboost_baseline`, `lightgbm_baseline`, `xgboost_tuned`
  * Metrics tracked: val_mse, val_mae, val_r2, test_mse, test_mae, test_r2, improvement_pct
  * Models serialized and stored as MLflow artifacts

### Best Model Performance

* **Best Pre-Tuning**: **XGBoost** with Test R² = **0.5729** (57.3% of price variance explained)
  * Test MAE: 24.60 EUR/MWh (typical prediction error)
  * Test MSE: 1416.11
* **Best Tuned**: **XGBoost (tuned)** via RandomizedSearchCV
  * Further gains achieved through hyperparameter optimization

### Key Findings

* **Top Features**: Neighboring country prices (39% importance) and Czech Republic prices (31%) are strongest predictors
* **Gradient Boosting Dominance**: XGBoost/LightGBM significantly outperform linear models (Ridge/Lasso) and single trees
* **Data Quality**: 41K+ hourly samples across 32 engineered features provided sufficient signal for ~57% R² performance
* **Next Gains**: Likely from demand data, cross-border flows, weather forecasts, and advanced time series architectures (LSTM, Transformer)

### Artifacts & Reproducibility

* Model results: `data/processed/model_results_v1_20251009.csv`
* Saved models: `best_model_xgboost_v1_20251009.joblib`, `best_model_xgboost_tuned_v1_20251009.joblib`
* Prediction visualizations: `images/model_predictions_*.png`, `images/model_timeseries_*.png`
* MLflow URI: `file:./mlruns` (local file-based tracking)
* View runs: `mlflow ui --backend-store-uri file:./mlruns --port 5000`

### Diagnostic Visualizations

* ✅ **Scatter Plots**: Actual vs Predicted prices with R², MAE, RMSE metrics
* ✅ **Time Series Plots**: Last 500 hours showing prediction accuracy over time
* ✅ **Automatic Generation**: Created during training and logged to MLflow artifacts
* ✅ **Error Analysis**: Visual identification of prediction patterns and outliers

### Experiment Tracking Preview

![MLflow-Experiment-Tracking](/images/MLflow-Experiment-Tracking.png)

## 📊 Milestone 3 Accomplishments

### FastAPI Inference Service

* ✅ **REST API Implementation**: Complete FastAPI application with `/health`, `/predict`, and `/docs` endpoints
* ✅ **Pydantic Validation**: Strict input validation for all 32 features with type checking and bounds enforcement
* ✅ **Interactive Documentation**: Auto-generated Swagger UI at `/docs` with example payloads
* ✅ **Model Loading**: XGBoost model (`best_model_xgboost_tuned_v1_20251009.joblib`) loaded at startup
* ✅ **Health Checks**: `/health` endpoint for Docker/Kubernetes liveness probes

### Docker Containerization

* ✅ **Training Image** (`Dockerfile.train`): Complete training pipeline in isolated environment
  * Includes all dependencies (pandas, scikit-learn, xgboost, lightgbm, mlflow)
  * Mounts volumes for data, mlruns, and source code
  * Automatic model training and artifact generation
  
* ✅ **Inference Image** (`Dockerfile.inference`): Lightweight FastAPI server
  * Minimal footprint with only inference dependencies (fastapi, uvicorn)
  * Copies pre-trained model and necessary artifacts
  * Exposes port 8000 for API requests
  * Optimized layer caching for fast builds

### Docker Compose Orchestration

* ✅ **Multi-Service Coordination**: Three services working together
  * **train**: Runs complete training pipeline on demand
  * **mlflow**: MLflow UI server (<http://localhost:5000>) for experiment tracking
  * **api**: FastAPI inference server (<http://localhost:8000/docs>)
  
* ✅ **Volume Mounts**: Persistent data sharing between containers
  * `./data`: Model-ready datasets and trained artifacts
  * `./mlruns`: MLflow experiment tracking data
  * `./src`: Source code for hot-reload capability
  
* ✅ **Network Communication**: Services discoverable by name (e.g., `mlflow` container for training service)
* ✅ **Environment Configuration**: Centralized MLflow tracking URI and Python settings

### Key Features

* **Production-Ready**: Proper error handling, logging, and HTTP status codes
* **Type Safety**: Full Pydantic validation prevents invalid requests
* **Reproducibility**: Exact feature names from training pipeline ensure consistency
* **Scalability**: Stateless design allows horizontal scaling via container orchestration
* **Observability**: Health checks and detailed API documentation

### 📷 Visualizations

**FastAPI Interactive Documentation**

![FastAPI-deployment](/images/FastAPI-deployment.png)

**Model Performance Diagnostics** - *Generated during training*

The training pipeline automatically creates diagnostic visualizations to assess model quality:
- **Scatter plots**: Actual vs Predicted prices for both validation and test sets
- **Time series plots**: Last 500 hours showing prediction tracking over time
- **Performance metrics**: R², MAE, RMSE displayed on plots for easy interpretation

These plots are saved to `images/` and logged to MLflow for experiment tracking.

---
