"""
ELECTRICITY PRICE FORECASTING - MODEL TRAINING PIPELINE
========================================================
This script trains multiple baseline ML models to predict German electricity prices
(day-ahead wholesale prices in €/MWh).

The pipeline:
1. Loads preprocessed features (temporal, renewable energy ratios, lagged prices, etc.)
2. Trains 4 different regression models on the training set
3. Evaluates each model on validation and test sets
4. Compares metrics (MSE, MAE, R²) to find the best performer
5. Extracts feature importance from tree-based models
6. Saves results and the best model for production use
"""

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
import joblib


# ====================
# CONFIGURATION
# ====================
DATA_DIR = Path("./data/processed")  # Where our processed features are stored
VERSION = "v1"                        # Feature engineering version
STAMP = "20251009"                    # Date stamp for reproducibility
TARGET_COL = "Germany/Luxembourg [€/MWh]"  # Column we're trying to predict

if __name__ == "__main__":
    # ====================
    # STEP 1: LOAD DATA
    # ====================
    # Load training and test data that was prepared by feature engineering
    train_path = DATA_DIR / f"train_{VERSION}_{STAMP}.parquet"
    test_path = DATA_DIR / f"test_{VERSION}_{STAMP}.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Training data shape: {train_df.shape}")  # (samples, features)
    print(f"Test data shape: {test_df.shape}\n")

    # ====================
    # STEP 2: PREPARE FEATURES AND TARGET
    # ====================
    # Separate features (X) from the target (y)
    # - X: All columns except our target (input variables)
    # - y: Our target column we want to predict (electricity price)
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # ====================
    # STEP 3: CREATE VALIDATION SET
    # ====================
    # Split training data into training (80%) and validation (20%) sets
    # This lets us test model performance on unseen data during training
    # random_state=42 ensures reproducibility (same split every time)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}\n")

    # ====================
    # STEP 4: DEFINE AND TRAIN MULTIPLE MODELS
    # ====================
    # We test 4 different model types to find which works best:
    # 1. Ridge: Linear model with L2 regularization (prevents overfitting)
    # 2. Lasso: Linear model with L1 regularization (also feature selection)
    # 3. Decision Tree: Single tree that splits data recursively
    # 4. Random Forest: Ensemble of many decision trees (usually more robust)
    models = {
        "ridge": Ridge(),
        "lasso": Lasso(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor()
    }
    results = {}
    
    # Train each model
    for model_name, model in models.items():
        print(f"Training {model_name}...", end=" ")
        
        # ====================
        # Create a Pipeline
        # ====================
        # A pipeline chains preprocessing and modeling steps together
        # This ensures the same transformations are applied to train/val/test
        # Steps:
        # 1. StandardScaler: Normalize features to have mean=0, std=1
        #    (Important for Ridge/Lasso which are sensitive to feature scaling)
        # 2. Model: The actual model we're training
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Train the model on training data
        pipeline.fit(X_train, y_train)
        print("Done")

        # ====================
        # Evaluate on Validation Set
        # ====================
        # Make predictions on the validation set (unseen during training)
        y_val_pred = pipeline.predict(X_val)
        
        # Calculate three metrics:
        val_mse = mean_squared_error(y_val, y_val_pred)  # Mean squared error
        val_mae = mean_absolute_error(y_val, y_val_pred)  # Mean absolute error
        val_r2 = r2_score(y_val, y_val_pred)  # R² score
        
        # ====================
        # Evaluate on Test Set
        # ====================
        # Test on completely unseen data (held out from start)
        # This is our final performance measure
        y_test_pred = pipeline.predict(X_test)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Store results for later comparison
        results[model_name] = {
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2
        }

    # ====================
    # EVALUATE RESULTS
    # ====================
    # Convert results dictionary to a pandas DataFrame for easy comparison
    # This makes it simple to see which model performs best
    results_df = pd.DataFrame(results).T  # Transpose so models are rows, metrics are columns
    
    # Sort by test R² score (higher is better for R²)
    # R² ranges from 0 to 1, where 1 means perfect predictions
    results_df_sorted = results_df.sort_values("test_r2", ascending=False)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print("\nMetrics explanation:")
    print("  - MSE (Mean Squared Error): Average of squared differences. Lower is better.")
    print("  - MAE (Mean Absolute Error): Average absolute difference. Lower is better.")
    print("  - R² Score: How well the model explains price variation (0-1, higher is better)")
    print("\n" + results_df_sorted.to_string())
    print("\n" + "="*80)
    
    # Identify best performing model
    best_model_name = results_df_sorted.index[0]
    best_model = models[best_model_name]
    best_results = results_df_sorted.iloc[0]
    
    print(f"\nBest Model: {best_model_name.upper()}")
    print(f"  Test R² Score: {best_results['test_r2']:.4f}")
    print(f"  Test MAE: {best_results['test_mae']:.4f} EUR/MWh")
    print(f"  Test MSE: {best_results['test_mse']:.4f}")
    
    # ====================
    # FEATURE IMPORTANCE (for tree-based models)
    # ====================
    # Tree-based models (Decision Tree, Random Forest) can tell us which features matter most
    # This is useful for understanding what drives electricity prices
    if best_model_name in ["decision_tree", "random_forest"]:
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE (Top 10)")
        print("="*80)
        print("Shows which features have the strongest influence on price predictions\n")
        
        # Get feature importances from the best model
        feature_importance = best_model.feature_importances_
        
        # Create a DataFrame pairing feature names with their importance scores
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        
        # Display top 10 features
        print(importance_df.head(10).to_string(index=False))
        print(f"\n(Total features: {len(importance_df)})")
    
    # ====================
    # SAVE RESULTS
    # ====================
    # Save the results for later reference and analysis
    results_path = DATA_DIR / f"model_results_{VERSION}_{STAMP}.csv"
    results_df_sorted.to_csv(results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Save the best model for later use in predictions
    import joblib
    model_path = DATA_DIR / f"best_model_{best_model_name}_{VERSION}_{STAMP}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Best model saved to: {model_path}")
