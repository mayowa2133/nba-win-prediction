# train_xgboost_prevstats.py

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import xgboost as xgb

MODEL_DIR = "models"
LOGREG_CONFIG_PATH = os.path.join(
    MODEL_DIR, "logreg_elo_rolling_last10_prevstats_scaled_config.json"
)

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_prevstats.pkl")
XGB_CONFIG_PATH = os.path.join(MODEL_DIR, "xgb_prevstats_config.json")


def main():
    if not os.path.exists(LOGREG_CONFIG_PATH):
        raise FileNotFoundError(
            f"{LOGREG_CONFIG_PATH} not found. "
            "Make sure you ran train_logistic_prevstats_scaled.py first."
        )

    with open(LOGREG_CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    data_csv = cfg["data_csv"]
    feature_cols = cfg["feature_cols"]
    train_seasons = cfg["train_seasons"]  # [2016, 2020]
    val_seasons = cfg["val_seasons"]      # [2021, 2022]
    test_seasons = cfg["test_seasons"]    # [2023, 2025]

    print(f"Loading data from {data_csv}...")
    df = pd.read_csv(data_csv)

    print("Using feature columns:")
    for c in feature_cols:
        print(" -", c)

    # Ensure numeric
    df["home_rest_days"] = df["home_rest_days"].astype(float)
    df["away_rest_days"] = df["away_rest_days"].astype(float)

    y = df["home_win"].astype(int).values
    X = df[feature_cols].values

    # Time-based split
    train_mask = (df["season"] >= train_seasons[0]) & (df["season"] <= train_seasons[1])
    val_mask = (df["season"] >= val_seasons[0]) & (df["season"] <= val_seasons[1])
    test_mask = (df["season"] >= test_seasons[0]) & (df["season"] <= test_seasons[1])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(
        f"Train size: {len(y_train)}, "
        f"Val size: {len(y_val)}, "
        f"Test size: {len(y_test)}"
    )

    # XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=400,          # reduced since we have no early stopping
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",        # if your version doesn’t support this, change to "auto"
        random_state=42,
        n_jobs=-1,
    )

    # We can still pass eval_set just to see logloss during training
    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True,
    )

    def eval_split(name, Xs, ys):
        probs = model.predict_proba(Xs)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        ll = log_loss(ys, probs)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"LogLoss:  {ll:.3f}")

    eval_split("Validation (2021–2022)", X_val, y_val)
    eval_split("Test (2023–2025)", X_test, y_test)

    # Save model + config
    import joblib

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, XGB_MODEL_PATH)

    xgb_cfg = {
        "feature_cols": feature_cols,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "test_seasons": test_seasons,
        "data_csv": data_csv,
        "model_type": "xgboost",
        "params": {
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "reg_lambda": model.reg_lambda,
            "reg_alpha": model.reg_alpha,
        },
    }

    with open(XGB_CONFIG_PATH, "w") as f:
        json.dump(xgb_cfg, f, indent=2)

    print(f"\nSaved XGBoost model to {XGB_MODEL_PATH}")
    print(f"Saved config to {XGB_CONFIG_PATH}")


if __name__ == "__main__":
    main()
