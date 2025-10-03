# German Electricity Price Forecasting: An End-to-End MLOps Project

![Project Status: In Progress](https://img.shields.io/badge/status-in%20progress-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.17-blue.svg)

This repository documents the development of a production-ready MLOps pipeline for forecasting day-ahead wholesale electricity prices in the German market. The goal of this project is to build a complete, automated system that demonstrates practical skills in MLOps and cloud engineering, moving beyond academic concepts to a deployable solution.

**Note:** This is an active project currently in its initial development phase. The features and architecture described below represent the final goal of the 8-week development sprint.

## 🎯 The Business Problem

The volatility of electricity prices is a major challenge for energy traders, grid operators, and industrial consumers. Accurate short-term price forecasting is critical for optimizing energy consumption, managing costs, and making strategic resource allocation decisions. This project will address this real-world need by building an automated system to predict future prices.

The data is sourced from **SMARD.de**, the official market data platform of the German Federal Network Agency (Bundesnetzagentur), ensuring relevance and credibility within the German market.

---

## 🚧 Project Roadmap

This project is being developed over an 8-week sprint. This roadmap outlines the key milestones and will be updated as the project progresses.

* [x] **Week 1:** Project Scaffolding, Data Ingestion & Complete EDA with Feature Engineering
* [ ] **Week 2:** Baseline Modeling & Experiment Tracking (MLflow)
* [ ] **Week 3:** Containerization for Reproducibility (Docker)
* [ ] **Week 4:** Automation with CI/CT Pipeline (GitHub Actions)
* [ ] **Week 5:** Advanced Modeling & Feature Enrichment (XGBoost)
* [ ] **Week 6:** Cloud Deployment on Microsoft Azure (FastAPI, Azure App Service)
* [ ] **Week 7:** Project Showcase & Narrative Crafting
* [ ] **Week 8:** Strategic Application & Interview Mastery

---

## 🔧 Planned Technical Architecture

The planned architecture is designed for automation, reproducibility, and scalability, incorporating modern MLOps best practices.

**Target Tech Stack:**

* **Language & Libraries**: Python, Pandas, Scikit-learn, XGBoost
* **Experiment Tracking**: MLflow
* **Containerization**: Docker
* **CI/CT Automation**: GitHub Actions
* **API Development**: FastAPI
* **Cloud Platform**: Microsoft Azure (App Service, Container Registry)

### Planned MLOps Features

* **📊 Experiment Tracking**: All model training runs **will be logged** with **MLflow**, capturing code versions, parameters, and metrics for full auditability.
* **📦 Reproducibility**: The entire application **will be containerized** with **Docker** to guarantee a consistent environment for training and deployment.
* **⚙️ Automation (CI/CT)**: A **GitHub Actions** workflow **will be implemented** to automate code quality checks (CI) and schedule weekly model retraining (CT).
* **☁️ Cloud Deployment**: The final model **will be served** via a **FastAPI** REST API, deployed as a Docker container on **Microsoft Azure App Service**.

---

## 📁 Project Structure

```text
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── interim/             # Clean intermediate datasets with metadata
│   │   ├── actual_generation_clean.parquet
│   │   ├── actual_generation_clean.metadata.json
│   │   ├── day_ahead_prices_clean.parquet
│   │   └── day_ahead_prices_clean.metadata.json
│   ├── processed/           # Model-ready feature datasets with train/test splits
│   │   ├── features_v1_20251003.parquet
│   │   ├── train_v1_20251003.parquet
│   │   ├── test_v1_20251003.parquet
│   │   └── corresponding .metadata.json files
│   └── raw/                 # Raw data files from SMARD.de
│       ├── Actual_generation_202101010000_202509180000_Hour.csv
│       └── Day-ahead_prices_202101010000_202509180000_Hour.csv
├── notebooks/               # Jupyter notebooks (contains 01-EDA.ipynb for EDA)
├── scripts/                 # Utility and pipeline scripts (currently empty)
├── src/                     # Source code for the project (currently empty)
└── tests/                   # Unit and integration tests (currently empty)
```

---

## 🛠️ Setup (Work in Progress)

As the project is in its initial phase, the primary setup involves cloning the repository and creating a local Python environment. Instructions will be updated as key milestones (like Dockerization) are completed.

**Week 1 Complete:** Comprehensive EDA and feature engineering has been implemented in `notebooks/01-EDA.ipynb`, with clean datasets and model-ready features saved to the `data/` directory. The rest of the pipeline (modeling, MLflow, Docker, CI/CD, API, and cloud deployment) is planned for upcoming weeks.

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

## 📊 Week 1 Accomplishments

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
