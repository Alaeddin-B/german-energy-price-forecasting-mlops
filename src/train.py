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
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from typing import cast
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("german-energy-price-forecasting")


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
    
    # Sanitize column names for XGBoost/LightGBM (they don't like special characters like [, ], <, >)
    # Replace problematic characters with underscores
    X.columns = X.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_').str.replace('>', '_').str.replace('€', 'EUR')
    X_test.columns = X_test.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_').str.replace('>', '_').str.replace('€', 'EUR')

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
                # Log to MLflow
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            mlflow.log_params({"model_type": model_name})
            mlflow.log_metric("val_mse", val_mse)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("val_r2", val_r2)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.sklearn.log_model(pipeline, name="model")

    # ====================
    # STEP 5: TRAIN SOPHISTICATED MODELS (Gradient Boosting)
    # ====================
    # Gradient boosting models are more powerful than simple ensembles
    # They build trees sequentially, where each new tree corrects errors from previous trees
    # XGBoost: eXtreme Gradient Boosting (industry standard, fast, handles regularization)
    # LightGBM: Light Gradient Boosting Machine (faster, uses less memory, good for large datasets)
    
    print("\nTraining XGBoost (this may take a moment)...", end=" ")
    
    # XGBoost model with sensible defaults
    # Parameters explained:
    # - n_estimators: How many trees to build (more = more complex, but takes longer)
    # - max_depth: How deep each tree can be (prevents overfitting)
    # - learning_rate: How much each tree's predictions are weighted (smaller = slower but more careful)
    # - subsample: Use 80% of data for each tree (reduces overfitting)
    # - colsample_bytree: Use 80% of features for each tree (adds randomness, reduces overfitting)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1  # Use all CPU cores for faster training
    )
    
    # Train directly (XGBoost doesn't need scaling like Ridge/Lasso do)
    xgb_model.fit(X_train, y_train)
    print("Done")
    
    # Evaluate XGBoost
    y_val_pred = xgb_model.predict(X_val)
    y_test_pred = xgb_model.predict(X_test)
    
    results["xgboost"] = {
        "val_mse": mean_squared_error(y_val, y_val_pred),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        "val_r2": r2_score(y_val, y_val_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    }
        # Log XGBoost to MLflow
    with mlflow.start_run(run_name="xgboost_baseline"):
        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        })
        mlflow.log_metrics(results["xgboost"])
        mlflow.xgboost.log_model(xgb_model, name="model")
    
    print("Training LightGBM (this may take a moment)...", end=" ")
    
    # LightGBM model - similar logic to XGBoost but faster
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    lgb_model.fit(X_train, y_train)
    print("Done")
    
    # Evaluate LightGBM
    y_val_pred = lgb_model.predict(X_val)
    y_test_pred = lgb_model.predict(X_test)

    # Convert predictions to numpy arrays to avoid type conflicts with scikit-learn metrics
    y_val_pred = np.asarray(y_val_pred)
    y_test_pred = np.asarray(y_test_pred)
    
    results["lightgbm"] = {
        "val_mse": mean_squared_error(y_val, y_val_pred),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        "val_r2": r2_score(y_val, y_val_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    }
        # Log LightGBM to MLflow
    with mlflow.start_run(run_name="lightgbm_baseline"):
        mlflow.log_params({
            "model_type": "lightgbm",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        })
        mlflow.log_metrics(results["lightgbm"])
        mlflow.lightgbm.log_model(lgb_model, name="model")
    # ====================
    # STEP 6: HYPERPARAMETER TUNING FOR BEST MODEL
    # ====================
    # After comparing all models, we take the best one and tune its hyperparameters
    # RandomizedSearchCV: Tests random combinations of parameters to find the best settings
    # This is more efficient than GridSearchCV (which tests ALL combinations)
    
    # First, find which model performed best on validation set
    val_r2_scores: dict[str, float] = {name: results[name]['val_r2'] for name in results}
    best_model_before_tuning = max(val_r2_scores, key=lambda name: val_r2_scores[name])
    
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING FOR BEST MODEL: {best_model_before_tuning.upper()}")
    print("="*80)
    print("Testing different parameter combinations to optimize performance...\n")
    
    # Define hyperparameter search space based on best model type
    if best_model_before_tuning == "xgboost":
        # XGBoost hyperparameters to tune
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
    elif best_model_before_tuning == "lightgbm":
        # LightGBM hyperparameters to tune
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        
    elif best_model_before_tuning == "random_forest":
        # Random Forest hyperparameters to tune
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [6, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
    else:
        # For Ridge/Lasso
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10]
        }
        def get_base_model(model_name: str) -> BaseEstimator:
            if model_name == "ridge":
                return Ridge()
            else:
                return Lasso()
        base_model = get_base_model(best_model_before_tuning)
    
    # RandomizedSearchCV: randomly samples 10 combinations and picks the best
    # cv=3: Use 3-fold cross-validation (split data 3 ways, test each way)
    # scoring='r2': Optimize for R² score (higher is better)
    # Cast to BaseEstimator for type checker (runtime: no-op)
    tuned_model = RandomizedSearchCV(
        estimator=cast(BaseEstimator, base_model),
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    # For tree-based models, no scaling needed
    # For Ridge/Lasso, we need to scale
    if best_model_before_tuning in ["ridge", "lasso"]:
        tuned_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', tuned_model)
        ])
        tuned_pipeline.fit(X_train, y_train)
        y_test_pred_tuned = tuned_pipeline.predict(X_test)
    else:
        tuned_model.fit(X_train, y_train)
        y_test_pred_tuned = tuned_model.predict(X_test)
    
    # Store tuned model results
    best_test_r2_tuned = r2_score(y_test, y_test_pred_tuned)
    best_test_mae_tuned = mean_absolute_error(y_test, y_test_pred_tuned)
    best_test_mse_tuned = mean_squared_error(y_test, y_test_pred_tuned)
    
    print("Best hyperparameters found:")
    if hasattr(tuned_model, 'best_params_'):
        for param, value in tuned_model.best_params_.items():
            print(f"  {param}: {value}")
    print("\nTuned Model Performance:")
    print(f"  Test R²: {best_test_r2_tuned:.4f} (was {results[best_model_before_tuning]['test_r2']:.4f})")
    print(f"  Test MAE: {best_test_mae_tuned:.4f} EUR/MWh (was {results[best_model_before_tuning]['test_mae']:.4f})")
    print(f"  Test MSE: {best_test_mse_tuned:.4f}")
    improvement = ((best_test_r2_tuned - results[best_model_before_tuning]['test_r2']) / 
                   results[best_model_before_tuning]['test_r2'] * 100)
    print(f"  Improvement: {improvement:+.2f}%")
        
    # Log tuned model to MLflow
    with mlflow.start_run(run_name=f"{best_model_before_tuning}_tuned"):
        # Log the best parameters found by RandomizedSearchCV
        if hasattr(tuned_model, 'best_params_'):
            mlflow.log_params(tuned_model.best_params_)
        
        # Log tuned performance metrics
        mlflow.log_metric("test_r2", best_test_r2_tuned)
        mlflow.log_metric("test_mae", best_test_mae_tuned)
        mlflow.log_metric("test_mse", best_test_mse_tuned)
        mlflow.log_metric("improvement_pct", improvement)
        
        # Log the tuned model
        if best_model_before_tuning in ["ridge", "lasso"]:
            mlflow.sklearn.log_model(tuned_pipeline, name="model")
        elif best_model_before_tuning == "xgboost":
            mlflow.xgboost.log_model(tuned_model.best_estimator_, name="model")
        elif best_model_before_tuning == "lightgbm":
            mlflow.lightgbm.log_model(tuned_model.best_estimator_, name="model")
        else:
            mlflow.sklearn.log_model(tuned_model, name="model")
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
    print("MODEL PERFORMANCE COMPARISON (BASELINE + SOPHISTICATED)")
    print("="*80)
    print("\nMetrics explanation:")
    print("  - MSE (Mean Squared Error): Average of squared differences. Lower is better.")
    print("  - MAE (Mean Absolute Error): Average absolute difference. Lower is better.")
    print("  - R² Score: How well the model explains price variation (0-1, higher is better)")
    print("\n" + results_df_sorted.to_string())
    print("\n" + "="*80)
    
    # Identify best performing model BEFORE tuning
    best_model_name = results_df_sorted.index[0]
    best_model = models.get(best_model_name)  # Try to get from original models dict
    if best_model is None:
        # For sophisticated models not in original dict
        if best_model_name == "xgboost":
            best_model = xgb_model
        elif best_model_name == "lightgbm":
            best_model = lgb_model
    
    best_results = results_df_sorted.iloc[0]
    
    print(f"\nBest Model (Before Tuning): {best_model_name.upper()}")
    print(f"  Test R² Score: {best_results['test_r2']:.4f}")
    print(f"  Test MAE: {best_results['test_mae']:.4f} EUR/MWh")
    print(f"  Test MSE: {best_results['test_mse']:.4f}")
    
    # ====================
    # FEATURE IMPORTANCE (for tree-based models)
    # ====================
    # Tree-based models (Decision Tree, Random Forest, XGBoost, LightGBM) can tell us which features matter most
    # This is useful for understanding what drives electricity prices
    if best_model_name in ["decision_tree", "random_forest", "xgboost", "lightgbm"]:
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE (Top 10)")
        print("="*80)
        print("Shows which features have the strongest influence on price predictions\n")
        
        # Get feature importances from the best model
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_ # type: ignore
        else:
            feature_importance = np.zeros(X.shape[1])
        
        # Create a DataFrame pairing feature names with their importance scores
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        
        # Display top 10 features
        print(importance_df.head(10).to_string(index=False))
        print(f"\n(Total features: {len(importance_df)})")
    
    # ====================
    # SAVE RESULTS AND MODELS
    # ====================
    # Save the results for later reference and analysis
    results_path = DATA_DIR / f"model_results_{VERSION}_{STAMP}.csv"
    results_df_sorted.to_csv(results_path)
    print("\n" + "="*80)
    print("SAVING MODELS AND RESULTS")
    print("="*80)
    print(f"\nResults saved to: {results_path}")
    
    # Save the best model for later use in predictions
    model_path = DATA_DIR / f"best_model_{best_model_name}_{VERSION}_{STAMP}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Best model (pre-tuning) saved to: {model_path}")
    
    # Save the tuned model
    tuned_model_path = DATA_DIR / f"best_model_{best_model_before_tuning}_tuned_{VERSION}_{STAMP}.joblib"
    joblib.dump(tuned_model, tuned_model_path)
    print(f"Tuned model saved to: {tuned_model_path}")
    
    print("\nTraining complete! Summary:")
    print("  - Baseline models tested: 4")
    print("  - Sophisticated models tested: 2 (XGBoost, LightGBM)")
    print(f"  - Best model: {best_model_name} (R² = {best_results['test_r2']:.4f})")
    print(f"  - Best tuned model: {best_model_before_tuning} (R² = {best_test_r2_tuned:.4f})")
