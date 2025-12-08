import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


DATA_DIR = Path("./data/processed")
VERSION = "v1"
STAMP = "20251009"
TARGET_COL = "Germany/Luxembourg [€/MWh]"

if __name__ == "__main__":
    # Load data
    train_path = DATA_DIR / f"train_{VERSION}_{STAMP}.parquet"
    test_path = DATA_DIR / f"test_{VERSION}_{STAMP}.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Separate features and target
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define baseline models
    models = {
        "ridge": Ridge(),
        "lasso": Lasso(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor()
    }
    results = {}
    for model_name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Fit model
        pipeline.fit(X_train, y_train)

        # Validate model
        y_val_pred = pipeline.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # Test model
        y_test_pred = pipeline.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Store results
        results[model_name] = {
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2
        }
