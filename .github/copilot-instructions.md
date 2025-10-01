# German Energy Price Forecasting MLOps Project

## Project Overview
This is an **8-week sprint project** developing a production-ready MLOps pipeline for forecasting day-ahead wholesale electricity prices in the German market using data from SMARD.de (German Federal Network Agency). Currently in **Week 1** with only EDA implemented.

## Data Architecture & Schema

### Core Datasets (CSV, semicolon-separated)
- **Day-ahead prices**: `data/raw/Day-ahead_prices_*.csv` - Hourly prices for Germany/Luxembourg and neighboring countries (€/MWh)
- **Actual generation**: `data/raw/Actual_generation_*.csv` - Hourly generation by source (Biomass, Hydropower, Wind offshore/onshore, Photovoltaics, Nuclear, Lignite, Hard coal, Fossil gas, etc.)

### Key Data Patterns
- **Time series structure**: Both datasets use `Start date;End date` format with hourly resolution
- **Missing values**: Represented as `-` (configured as `na_values=['-']` in pandas)
- **Nuclear power context**: Expect significant missing/zero nuclear generation data due to Germany's nuclear phase-out (completed April 2023)
- **Feature engineering focus**: Seasonal patterns (month, season, day_of_week, hour) are critical - December/autumn have highest prices, weekends show lower demand
- **Known challenge**: Weak correlation (≈0.02) between total generation and prices - other factors (demand, imports/exports, market mechanisms) are more influential

## Development Workflow

### Current Phase (Week 1): EDA Only
- Main analysis in `notebooks/01-EDA.ipynb`
- Uses relative paths: `../data/raw/` for data loading
- Key libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Critical insight**: Developer identified need to learn lagged time series features for forecasting

### Future MLOps Stack (Weeks 2-8)
- **Experiment tracking**: MLflow
- **Containerization**: Docker 
- **CI/CT**: GitHub Actions (weekly model retraining)
- **API**: FastAPI
- **Cloud**: Microsoft Azure (App Service, Container Registry)

## Code Conventions

### Data Loading Pattern
```python
# Standard pattern for loading SMARD.de datasets
df = pd.read_csv(path, sep=';', low_memory=False, na_values=['-'])
```

### Feature Engineering Approach
- Extract temporal features: `year`, `month`, `season`, `day_of_week`, `hour`
- Focus on seasonality and weekly patterns for price prediction
- Consider lagged features (developer's current learning objective)

### Project Structure Logic
- `src/`: Empty, will contain production ML code
- `scripts/`: Empty, will contain pipeline utilities  
- `notebooks/`: EDA and experimentation (numbered: `01-EDA.ipynb`)
- `data/processed/`: Empty, will contain engineered features
- `tests/`: Empty, will contain unit/integration tests

## AI Development Guidelines

1. **Sprint timeline flexibility** - While project follows 8-week phases, can implement broader scope features when beneficial for learning or architecture
2. **Use project-specific data loading** - Always use semicolon separator and handle `-` as NaN
3. **Focus on time series patterns** - This is energy market data with strong temporal dependencies
4. **Consider the learning curve** - Developer is actively learning time series forecasting techniques
5. **Prepare for MLOps** - Structure code with future containerization and automation in mind

## Key Technical Challenges
- Understanding and implementing lagged time series features
- Managing weak price-generation correlation in modeling
- Building reproducible MLOps pipeline from research notebook code
- Handling nuclear power phase-out data gaps in generation modeling