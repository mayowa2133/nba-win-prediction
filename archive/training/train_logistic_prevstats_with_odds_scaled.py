# train_logistic_prevstats_with_odds_scaled.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Use only games where we have odds and seasons where Kaggle has data
    df = df[(df["season"] <= 2022) & (~df["market_home_prob"].isna())].copy()

    print("Games with odds by season:")
    print(df["season"].value_counts().sort_index())

    feature_cols = [
        # existing features
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
        # NEW: market features
        "market_home_prob",
        "market_spread",
        "market_total",
    ]

    # Time-based split: train <=2019, val 2020-2021, test 2022
    train = df[df["season"] <= 2019].copy()
    val = df[(df["season"] >= 2020) & (df["season"] <= 2021)].copy()
    test = df[df["season"] == 2022].copy()

    print(
        f"\nTrain size: {len(train)}, "
        f"Val size: {len(val)}, "
        f"Test size: {len(test)}"
    )

    X_train = train[feature_cols].values
    y_train = train["home_win"].values

    X_val = val[feature_cols].values
    y_val = val["home_win"].values

    X_test = test[feature_cols].values
    y_test = test["home_win"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Try a few C values (inverse regularization strength)
    C_values = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_C = None
    best_val_ll = np.inf
    best_model = None

    print("\nTuning C on validation set:")
    for C in C_values:
        model = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            C=C,
        )
        model.fit(X_train_scaled, y_train)
        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        val_ll = log_loss(y_val, val_probs)
        print(f"  C={C:.2f} -> Val Accuracy={val_acc:.3f}, LogLoss={val_ll:.3f}")

        if val_ll < best_val_ll:
            best_val_ll = val_ll
            best_C = C
            best_model = model

    print(f"\nBest C based on val logloss: {best_C} (logloss={best_val_ll:.3f})")

    # Optionally, retrain best model on train+val for final test metrics
    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        C=best_C,
    )
    final_model.fit(X_trainval, y_trainval)

    # Evaluate on test (2022)
    test_probs = final_model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_preds)
    test_ll = log_loss(y_test, test_probs)

    print("\n=== Test (2022 with odds, scaled + tuned) ===")
    print(f"Accuracy: {test_acc:.3f}")
    print(f"LogLoss:  {test_ll:.3f}")


if __name__ == "__main__":
    main()
