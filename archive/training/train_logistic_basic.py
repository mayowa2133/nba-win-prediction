import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

INPUT_CSV = "games_all_2015_2025_features_basic.csv"

# seasons for splits
TRAIN_START, TRAIN_END = 2015, 2020
VAL_START, VAL_END = 2021, 2022
TEST_START, TEST_END = 2023, 2025

FEATURE_COLS = [
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "is_postseason",
]

def load_data():
    df = pd.read_csv(INPUT_CSV)

    # make sure numeric
    df["home_rest_days"] = df["home_rest_days"].astype(float)
    df["away_rest_days"] = df["away_rest_days"].astype(float)

    # target
    y = df["home_win"].values

    X = df[FEATURE_COLS].values
    seasons = df["season"].values

    def season_mask(start, end):
        return (seasons >= start) & (seasons <= end)

    mask_train = season_mask(TRAIN_START, TRAIN_END)
    mask_val = season_mask(VAL_START, VAL_END)
    mask_test = season_mask(TEST_START, TEST_END)

    data = {
        "X_train": X[mask_train],
        "y_train": y[mask_train],
        "X_val": X[mask_val],
        "y_val": y[mask_val],
        "X_test": X[mask_test],
        "y_test": y[mask_test],
    }

    return data

if __name__ == "__main__":
    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")

    # basic logistic regression
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # evaluate on val
    p_val = model.predict_proba(X_val)[:, 1]
    yhat_val = (p_val >= 0.5).astype(int)
    acc_val = accuracy_score(y_val, yhat_val)
    ll_val = log_loss(y_val, p_val)

    print("\n=== Validation (2021–2022) ===")
    print(f"Accuracy: {acc_val:.3f}")
    print(f"LogLoss:  {ll_val:.3f}")

    # evaluate on test (this is what we really care about)
    p_test = model.predict_proba(X_test)[:, 1]
    yhat_test = (p_test >= 0.5).astype(int)
    acc_test = accuracy_score(y_test, yhat_test)
    ll_test = log_loss(y_test, p_test)

    print("\n=== Test (2023–2025) ===")
    print(f"Accuracy: {acc_test:.3f}")
    print(f"LogLoss:  {ll_test:.3f}")
