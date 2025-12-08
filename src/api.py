"""
ELECTRICITY PRICE FORECASTING - INFERENCE API
==============================================
This FastAPI application serves the trained XGBoost model for real-time predictions.

When deployed, it accepts HTTP POST requests with feature data and returns
predicted electricity prices for the German/Luxembourg market.

API Endpoints:
- GET /health - Check if the service is running
- POST /predict - Get price predictions from input features
- GET / - API documentation and info
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# ====================
# CONFIGURATION
# ====================
MODEL_PATH = Path("./data/processed/best_model_xgboost_v1_20251009.joblib")
APP_VERSION = "1.0.0"
MODEL_VERSION = "v1_20251009"

# All 32 features your model expects (from the metadata file)
EXPECTED_FEATURES = [
    "Germany/Luxembourg _EUR/MWh_ Previous Day",
    "Germany/Luxembourg _EUR/MWh_ Previous Day 24-Hour Std",
    "Germany/Luxembourg _EUR/MWh_ 3-Day MA",
    "Germany/Luxembourg _EUR/MWh_ Last Week",
    "Belgium _EUR/MWh_ Previous Day",
    "Denmark 1 _EUR/MWh_ Previous Day",
    "Denmark 2 _EUR/MWh_ Previous Day",
    "France _EUR/MWh_ Previous Day",
    "Netherlands _EUR/MWh_ Previous Day",
    "Austria _EUR/MWh_ Previous Day",
    "Poland _EUR/MWh_ Previous Day",
    "Czech Republic _EUR/MWh_ Previous Day",
    "year",
    "public_holiday",
    "Renewable Ratio Previous Day",
    "Renewable Ratio 3-Day MA",
    "Renewable Ratio Delta",
    "Hour-Renewable Interaction",
    "Neighboring Countries Previous Day Spread",
    "Neighboring Countries Previous Day Std",
    "∅ DE/LU neighbours _EUR/MWh_ Previous Day",
    "∅ DE/LU neighbours _EUR/MWh_ Previous Day 24-Hour Std",
    "∅ DE/LU neighbours _EUR/MWh_ 3-Day MA",
    "∅ DE/LU neighbours _EUR/MWh_ Last Week",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "season_sin",
    "season_cos",
    "month_sin",
    "month_cos"
]


# ====================
# INITIALIZE FASTAPI APP
# ====================
app = FastAPI(
    title="German Electricity Price Forecasting API",
    description="Predict day-ahead wholesale electricity prices for the German/Luxembourg market",
    version=APP_VERSION
)


# ====================
# LOAD MODEL ON STARTUP
# ====================
# Load the model once when the app starts (not on every request - that would be slow!)
try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    model = None


# ====================
# DEFINE REQUEST/RESPONSE SCHEMAS
# ====================
# Pydantic models define the structure of data we accept and return
# They automatically validate inputs and generate API documentation

class PredictionInput(BaseModel):
    """
    The data structure for a prediction request.
    
    Each field corresponds to one of the 32 features the model expects.
    Users send this data, and we validate it automatically.
    """
    # Price lag features (in EUR/MWh)
    germany_luxembourg_previous_day: float = Field(..., description="German price 24 hours ago (EUR/MWh)")
    germany_luxembourg_previous_day_24h_std: float = Field(..., description="24-hour rolling std of German prices")
    germany_luxembourg_3day_ma: float = Field(..., description="3-day moving average of German prices")
    germany_luxembourg_last_week: float = Field(..., description="German price 168 hours ago (EUR/MWh)")
    
    # Neighboring country prices (previous day, in EUR/MWh)
    belgium_previous_day: float
    denmark1_previous_day: float
    denmark2_previous_day: float
    france_previous_day: float
    netherlands_previous_day: float
    austria_previous_day: float
    poland_previous_day: float
    czech_republic_previous_day: float
    
    # Temporal features
    year: int = Field(..., ge=2021, le=2030, description="Year (e.g., 2025)")
    public_holiday: int = Field(..., ge=0, le=1, description="1 if public holiday, 0 otherwise")
    
    # Renewable energy features
    renewable_ratio_previous_day: float = Field(..., ge=0.0, le=1.0, description="Renewable generation ratio (0-1)")
    renewable_ratio_3day_ma: float = Field(..., ge=0.0, le=1.0)
    renewable_ratio_delta: float = Field(..., description="Change in renewable ratio")
    hour_renewable_interaction: float = Field(..., description="Interaction term: hour × renewable ratio")
    
    # Cross-border spreads and averages
    neighboring_countries_previous_day_spread: float = Field(..., description="Price spread across neighbors")
    neighboring_countries_previous_day_std: float = Field(..., description="Std dev of neighbor prices")
    avg_neighbours_previous_day: float = Field(..., description="Average neighbor price")
    avg_neighbours_previous_day_24h_std: float = Field(..., description="24h std of avg neighbor prices")
    avg_neighbours_3day_ma: float = Field(..., description="3-day MA of avg neighbor prices")
    avg_neighbours_last_week: float = Field(..., description="Avg neighbor price 168h ago")
    
    # Cyclic temporal encodings (sine/cosine transformations)
    hour_sin: float = Field(..., ge=-1.0, le=1.0, description="Hour encoded as sine")
    hour_cos: float = Field(..., ge=-1.0, le=1.0, description="Hour encoded as cosine")
    day_of_week_sin: float = Field(..., ge=-1.0, le=1.0)
    day_of_week_cos: float = Field(..., ge=-1.0, le=1.0)
    season_sin: float = Field(..., ge=-1.0, le=1.0)
    season_cos: float = Field(..., ge=-1.0, le=1.0)
    month_sin: float = Field(..., ge=-1.0, le=1.0)
    month_cos: float = Field(..., ge=-1.0, le=1.0)
    
    class Config:
        # Example data for API documentation
        json_schema_extra = {
            "example": {
                "germany_luxembourg_previous_day": 85.5,
                "germany_luxembourg_previous_day_24h_std": 12.3,
                "germany_luxembourg_3day_ma": 82.1,
                "germany_luxembourg_last_week": 90.2,
                "belgium_previous_day": 83.0,
                "denmark1_previous_day": 88.0,
                "denmark2_previous_day": 87.5,
                "france_previous_day": 80.0,
                "netherlands_previous_day": 84.0,
                "austria_previous_day": 82.0,
                "poland_previous_day": 78.0,
                "czech_republic_previous_day": 79.0,
                "year": 2025,
                "public_holiday": 0,
                "renewable_ratio_previous_day": 0.45,
                "renewable_ratio_3day_ma": 0.42,
                "renewable_ratio_delta": 0.03,
                "hour_renewable_interaction": 5.4,
                "neighboring_countries_previous_day_spread": 10.0,
                "neighboring_countries_previous_day_std": 3.5,
                "avg_neighbours_previous_day": 82.7,
                "avg_neighbours_previous_day_24h_std": 4.2,
                "avg_neighbours_3day_ma": 81.5,
                "avg_neighbours_last_week": 85.0,
                "hour_sin": 0.866,
                "hour_cos": 0.5,
                "day_of_week_sin": 0.0,
                "day_of_week_cos": 1.0,
                "season_sin": 0.0,
                "season_cos": -1.0,
                "month_sin": -0.5,
                "month_cos": -0.866
            }
        }


class PredictionOutput(BaseModel):
    """
    The structure of our prediction response.
    
    Contains the predicted price and metadata about the prediction.
    """
    predicted_price_eur_mwh: float = Field(..., description="Predicted electricity price in EUR/MWh")
    model_version: str = Field(..., description="Version of the model used")
    confidence_info: Dict[str, Any] = Field(..., description="Additional metadata about the prediction")


# ====================
# API ENDPOINTS
# ====================

@app.get("/")
async def root():
    """
    Root endpoint - provides basic API information.
    
    This is what users see when they visit the API URL in a browser.
    """
    return {
        "message": "German Electricity Price Forecasting API",
        "version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "endpoints": {
            "/health": "Check API health status",
            "/predict": "POST price prediction request",
            "/docs": "Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Used by Docker, Kubernetes, or cloud services to verify the app is running.
    Returns 200 if healthy, 503 if model failed to load.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded - service unavailable"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Main prediction endpoint.
    
    Accepts feature data via POST request, runs it through the model,
    and returns the predicted electricity price.
    
    Steps:
    1. Validate input (Pydantic does this automatically)
    2. Convert to DataFrame with correct feature names
    3. Make prediction with loaded model
    4. Return result with metadata
    """
    
    # Check if model loaded successfully
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded - cannot make predictions"
        )
    
    try:
        # ====================
        # PREPARE INPUT DATA
        # ====================
        # Convert Pydantic model to dictionary
        input_dict = {
            "Germany/Luxembourg _EUR/MWh_ Previous Day": input_data.germany_luxembourg_previous_day,
            "Germany/Luxembourg _EUR/MWh_ Previous Day 24-Hour Std": input_data.germany_luxembourg_previous_day_24h_std,
            "Germany/Luxembourg _EUR/MWh_ 3-Day MA": input_data.germany_luxembourg_3day_ma,
            "Germany/Luxembourg _EUR/MWh_ Last Week": input_data.germany_luxembourg_last_week,
            "Belgium _EUR/MWh_ Previous Day": input_data.belgium_previous_day,
            "Denmark 1 _EUR/MWh_ Previous Day": input_data.denmark1_previous_day,
            "Denmark 2 _EUR/MWh_ Previous Day": input_data.denmark2_previous_day,
            "France _EUR/MWh_ Previous Day": input_data.france_previous_day,
            "Netherlands _EUR/MWh_ Previous Day": input_data.netherlands_previous_day,
            "Austria _EUR/MWh_ Previous Day": input_data.austria_previous_day,
            "Poland _EUR/MWh_ Previous Day": input_data.poland_previous_day,
            "Czech Republic _EUR/MWh_ Previous Day": input_data.czech_republic_previous_day,
            "year": input_data.year,
            "public_holiday": input_data.public_holiday,
            "Renewable Ratio Previous Day": input_data.renewable_ratio_previous_day,
            "Renewable Ratio 3-Day MA": input_data.renewable_ratio_3day_ma,
            "Renewable Ratio Delta": input_data.renewable_ratio_delta,
            "Hour-Renewable Interaction": input_data.hour_renewable_interaction,
            "Neighboring Countries Previous Day Spread": input_data.neighboring_countries_previous_day_spread,
            "Neighboring Countries Previous Day Std": input_data.neighboring_countries_previous_day_std,
            "∅ DE/LU neighbours _EUR/MWh_ Previous Day": input_data.avg_neighbours_previous_day,
            "∅ DE/LU neighbours _EUR/MWh_ Previous Day 24-Hour Std": input_data.avg_neighbours_previous_day_24h_std,
            "∅ DE/LU neighbours _EUR/MWh_ 3-Day MA": input_data.avg_neighbours_3day_ma,
            "∅ DE/LU neighbours _EUR/MWh_ Last Week": input_data.avg_neighbours_last_week,
            "hour_sin": input_data.hour_sin,
            "hour_cos": input_data.hour_cos,
            "day_of_week_sin": input_data.day_of_week_sin,
            "day_of_week_cos": input_data.day_of_week_cos,
            "season_sin": input_data.season_sin,
            "season_cos": input_data.season_cos,
            "month_sin": input_data.month_sin,
            "month_cos": input_data.month_cos
        }
        
        # Convert to DataFrame (model expects this format)
        input_df = pd.DataFrame([input_dict])
        
        # ====================
        # MAKE PREDICTION
        # ====================
        prediction = model.predict(input_df)[0]  # [0] because predict returns array
        
        # ====================
        # RETURN RESULT
        # ====================
        return PredictionOutput(
            predicted_price_eur_mwh=float(prediction),
            model_version=MODEL_VERSION,
            confidence_info={
                "note": "This prediction is based on XGBoost model trained on 2021-2025 data",
                "features_used": len(input_dict),
                "model_type": "XGBoost Regressor"
            }
        )
        
    except Exception as e:
        # If anything goes wrong, return a proper error message
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ====================
# RUN THE APP (for local testing)
# ====================
if __name__ == "__main__":
    import uvicorn
    # Run the API locally on http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)