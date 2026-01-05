# train_logistic_prevstats_scaled.py

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_CSV = "games_all_2015_2025_features_rolling_last10_prevstats.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_prevstats_scaled.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_prevstats_scaled_config.json")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)

    # Base features from your current final model
    base_feature_cols = [
        "elo_diff",
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
        "is_postseason",
        "simple_off_diff",
        "simple_def_diff",
        "simple_net_diff",
        "simple_win_pct_diff",
        "games_played_diff",
        "lastN_off_diff",
        "lastN_def_diff",
        "lastN_net_diff",
        "lastN_win_pct_diff",
        "lastN_games_diff",
    ]

    # New prev-season features
    prev_feature_cols = [
        "prev_off_rating_diff",
        "prev_def_rating_diff",
        "prev_net_rating_diff",
        "prev_efg_pct_diff",
        "prev_ts_pct_diff",
        "prev_pace_diff",
        "prev_oreb_pct_diff",
        "prev_dreb_pct_diff",
        "prev_reb_pct_diff",
        "prev_tm_tov_pct_diff",
        "prev_w_pct_diff",
        "prev_pie_diff",
    ]

    feature_cols = base_feature_cols + prev_feature_cols
    print("Using feature columns:")
    for c in feature_cols:
        print(" -", c)

    # Make sure rest days are numeric
    df["home_rest_days"] = df["home_rest_days"].astype(float)
    df["away_rest_days"] = df["away_rest_days"].astype(float)

    # Target
    y = df["home_win"].astype(int).values
    X = df[feature_cols].values

    # Time-based split:
    # train: 2016-2020
    # val:   2021-2022
    # test:  2023-2025
    train_seasons = (2016, 2020)
    val_seasons = (2021, 2022)
    test_seasons = (2023, 2025)

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

    # Pipeline: Standardize -> Logistic Regression
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    C=0.5,          # stronger regularization than default 1.0
                    penalty="l2",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    def eval_split(name, Xs, ys):
        probs = clf.predict_proba(Xs)[:, 1]
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

    joblib.dump(clf, MODEL_PATH)
    config = {
        "feature_cols": feature_cols,
        "train_seasons": list(train_seasons),
        "val_seasons": list(val_seasons),
        "test_seasons": list(test_seasons),
        "data_csv": DATA_CSV,
        "scaled": True,
        "C": 0.5,
        "model_type": "logistic_scaled",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved scaled model to {MODEL_PATH}")
    print(f"Saved config to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
