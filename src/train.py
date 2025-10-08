import pandas as pd
from sklearn.model_selection import train_test_split



train = pd.read_parquet("../data/processed/train_v1_20251003.parquet")
validation = pd.read_parquet("../data/processed/test_v1_20251003.parquet")
target_col = 'Germany/Luxembourg [€/MWh]'

X = train.drop(columns=[target_col])
y = train[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

# baseline model