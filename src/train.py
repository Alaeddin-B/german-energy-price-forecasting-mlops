import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data/processed")
VERSION = "v1"
STAMP = "20251009"
TARGET_COL = "Germany/Luxembourg [€/MWh]"

def load_data():
    train_path = DATA_DIR / f"train_{VERSION}_{STAMP}.parquet"
    test_path = DATA_DIR / f"test_{VERSION}_{STAMP}.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df

def split_features(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    safe = np.where(denom == 0, 1, denom)
    return (200 * np.mean(np.abs(y_true - y_pred) / safe))

def evaluate(y_true, y_pred, label: str):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    return {
        "model": label,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "sMAPE": smape_val
    }

def print_metrics(rows):
    print("\n=== Evaluation Metrics ===")
    for row in rows:
        print(f"{row['model']:<18} | RMSE: {row['RMSE']:.2f} | MAE: {row['MAE']:.2f} | R2: {row['R2']:.3f} | sMAPE: {row['sMAPE']:.2f}%")

def naive_baseline(test_df: pd.DataFrame):
    # Use the engineered previous day lag feature
    lag_col = f"{TARGET_COL} Previous Day"
    y_true = test_df[TARGET_COL]
    y_pred = test_df[lag_col]
    return evaluate(y_true, y_pred, "Naive_24h_Lag")


def main():
    train_df, test_df = load_data()
    
    X_train_full, y_train_full = split_features(train_df)
    X_test, y_test = split_features(test_df)

    # Baseline (naive)
    metrics = []
    metrics.append(naive_baseline(test_df))

    # Models to try
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=2025))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, random_state=2025, max_iter=10000))
        ]),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=50,
            random_state=2025
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=2025
        )
    }

    # Final train on full train set -> evaluate on test
    for name, model in models.items():
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)
        metrics.append(evaluate(y_test, y_pred, name))

    print_metrics(metrics)

    # Feature importance for tree models
    if hasattr(models["RandomForest"], "feature_importances_"):
        fi = models["RandomForest"].feature_importances_
        imp_df = pd.DataFrame({"feature": X_train_full.columns, "importance": fi}) \
                    .sort_values("importance", ascending=False).head(15)
        print("\nTop 15 RandomForest features:")
        for _, row in imp_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    main()