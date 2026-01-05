# train_logistic_prevstats_with_odds.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Define label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Only use games where we actually have odds + only seasons where Kaggle has data
    df = df[(df["season"] <= 2022) & (~df["market_home_prob"].isna())].copy()

    print("Games with odds by season:")
    print(df["season"].value_counts().sort_index())

    # Feature set: previous best features + market-based features
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

    # Split by season: train <=2019, val 2020-2021, test 2022
    train = df[df["season"] <= 2019].copy()
    val = df[(df["season"] >= 2020) & (df["season"] <= 2021)].copy()
    test = df[df["season"] == 2022].copy()

    print(
        f"\nTrain size: {len(train)}, "
        f"Val size: {len(val)}, "
        f"Test size: {len(test)}"
    )

    X_train = train[feature_cols]
    y_train = train["home_win"]

    X_val = val[feature_cols]
    y_val = val["home_win"]

    X_test = test[feature_cols]
    y_test = test["home_win"]

    # Basic logistic regression; scaling often helps, but let's start raw
    model = LogisticRegression(max_iter=2000, solver="lbfgs")

    print("\nFitting logistic regression with odds + team features...")
    model.fit(X_train, y_train)

    def eval_split(name, X, y):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        ll = log_loss(y, probs)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"LogLoss:  {ll:.3f}")

    eval_split("Validation (2020–2021)", X_val, y_val)
    eval_split("Test (2022 w/ odds)", X_test, y_test)


if __name__ == "__main__":
    main()
