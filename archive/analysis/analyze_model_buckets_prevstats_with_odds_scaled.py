# analyze_model_buckets_prevstats_with_odds_scaled.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")


FEATURE_COLS = [
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
    # odds features
    "market_home_prob",
    "market_spread",
    "market_total",
]


def bucket(p: float) -> str:
    """Bucket a probability into human-readable ranges."""
    if p < 0.50:
        return "[0.00,0.50)"
    elif p < 0.55:
        return "[0.50,0.55)"
    elif p < 0.60:
        return "[0.55,0.60)"
    elif p < 0.65:
        return "[0.60,0.65)"
    elif p < 0.70:
        return "[0.65,0.70)"
    elif p < 0.75:
        return "[0.70,0.75)"
    elif p < 0.80:
        return "[0.75,0.80)"
    else:
        return "[0.80,1.00]"


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Keep seasons where we have odds, up to 2022 (same as before)
    df = df[(df["season"] <= 2022) & (~df["market_home_prob"].isna())].copy()

    print("Games with odds by season:")
    print(df["season"].value_counts().sort_index())

    # Split: train <= 2019, val 2020–2021, test 2022
    train = df[df["season"] <= 2019].copy()
    val = df[(df["season"] >= 2020) & (df["season"] <= 2021)].copy()
    test = df[df["season"] == 2022].copy()

    print(f"\nTrain size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

    X_train = train[FEATURE_COLS].values
    y_train = train["home_win"].values

    X_val = val[FEATURE_COLS].values
    y_val = val["home_win"].values

    X_test = test[FEATURE_COLS].values
    y_test = test["home_win"].values

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning on C using val logloss (same as before)
    Cs = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_C = None
    best_ll = None

    print("\nTuning C on validation set:")
    for C in Cs:
        model = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=C
        )
        model.fit(X_train_scaled, y_train)
        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        ll = log_loss(y_val, val_probs)
        acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
        print(f"  C={C:.2f} -> Val Accuracy={acc:.3f}, LogLoss={ll:.3f}")
        if best_ll is None or ll < best_ll:
            best_ll = ll
            best_C = C

    print(f"\nBest C based on val logloss: {best_C} (logloss={best_ll:.3f})")

    # Train final model on train+val with best C
    train_val = pd.concat([train, val], axis=0)
    X_train_val = train_val[FEATURE_COLS].values
    y_train_val = train_val["home_win"].values
    X_train_val_scaled = scaler.transform(X_train_val)  # still using scaler fit on train

    final_model = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=best_C
    )
    final_model.fit(X_train_val_scaled, y_train_val)

    # Evaluate on test (2022)
    test_probs = final_model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, test_preds)
    test_ll = log_loss(y_test, test_probs)

    print("\n=== Test (2022, model with odds) ===")
    print(f"Accuracy: {test_acc:.3f}")
    print(f"LogLoss:  {test_ll:.3f}")

    # Attach probs & buckets to test DF
    test = test.copy()
    test["model_prob"] = test_probs
    test["model_bucket"] = test["model_prob"].apply(bucket)

    print("\nBucket-wise performance for MODEL (2022):")
    print("Bucket         Games   AvgProb   ActualWin%")
    print("-------------------------------------------")
    for b, group in test.groupby("model_bucket"):
        g = len(group)
        avg_p = group["model_prob"].mean()
        win_rate = group["home_win"].mean()
        print(f"{b:12} {g:5d}   {avg_p:7.3f}     {win_rate:7.3f}")

    # Optional: compare with market in same buckets
    print("\nBucket-wise comparison: MODEL vs MARKET (2022):")
    print("Bucket         Games   AvgModelP  AvgMarketP  ActualWin%")
    print("--------------------------------------------------------")
    for b, group in test.groupby("model_bucket"):
        g = len(group)
        avg_model = group["model_prob"].mean()
        avg_market = group["market_home_prob"].mean()
        win_rate = group["home_win"].mean()
        print(f"{b:12} {g:5d}   {avg_model:9.3f}  {avg_market:10.3f}  {win_rate:9.3f}")


if __name__ == "__main__":
    main()
