# simulate_roi_high_confidence.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# Path to the dataset that includes prev-season stats + rolling + odds
DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")

# Feature columns (same idea as your prevstats_with_odds_scaled setup)
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
    # odds-derived features
    "market_home_prob",
    "market_spread",
    "market_total",
]

# These are the actual moneyline columns in your CSV
HOME_ML_COL = "home_ml"
AWAY_ML_COL = "away_ml"


def american_to_return(odds_american: float) -> float:
    """
    Convert American odds to profit for 1 unit stake.
    Returns the net profit (excluding stake) if the bet wins.

    Examples:
        +150 -> 1.5 units profit
        -150 -> 0.666... units profit
    """
    if odds_american > 0:
        return odds_american / 100.0
    else:
        return 100.0 / abs(odds_american)


def main():
    # ---------------------------------------------------------------------
    # 1) Load data and basic filtering
    # ---------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Home win label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Restrict to seasons where we actually have odds (and non-null moneylines)
    mask = (
        (df["season"] <= 2022)
        & (~df["market_home_prob"].isna())
        & (~df[HOME_ML_COL].isna())
        & (~df[AWAY_ML_COL].isna())
    )
    df = df[mask].copy()

    print("Games with odds by season:")
    print(df["season"].value_counts().sort_index())
    print()

    # ---------------------------------------------------------------------
    # 2) Train/val/test split by season
    #     - Train: <= 2019
    #     - Val:   2020–2021
    #     - Test:  2022
    # ---------------------------------------------------------------------
    train = df[df["season"] <= 2019].copy()
    val = df[(df["season"] >= 2020) & (df["season"] <= 2021)].copy()
    test = df[df["season"] == 2022].copy()

    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

    X_train = train[FEATURE_COLS].values
    y_train = train["home_win"].values

    X_val = val[FEATURE_COLS].values
    y_val = val["home_win"].values

    X_test = test[FEATURE_COLS].values
    y_test = test["home_win"].values

    # ---------------------------------------------------------------------
    # 3) Scale features and tune C on validation set
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    Cs = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_C = None
    best_ll = None

    print("\nTuning C on validation set:")
    for C in Cs:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=C,
        )
        model.fit(X_train_scaled, y_train)
        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        ll = log_loss(y_val, val_probs)
        acc = (val_probs >= 0.5).mean() if len(val_probs) > 0 else 0.0
        print(f"  C={C:5.2f} -> Val Accuracy={acc:.3f}, LogLoss={ll:.3f}")
        if best_ll is None or ll < best_ll:
            best_ll = ll
            best_C = C

    print(f"\nBest C based on val logloss: {best_C} (logloss={best_ll:.3f})")

    # ---------------------------------------------------------------------
    # 4) Train final model on train+val, re-fit scaler, and evaluate on test
    # ---------------------------------------------------------------------
    train_val = pd.concat([train, val], axis=0)
    X_train_val = train_val[FEATURE_COLS].values
    y_train_val = train_val["home_win"].values

    scaler_final = StandardScaler()
    X_train_val_scaled = scaler_final.fit_transform(X_train_val)
    X_test_scaled = scaler_final.transform(X_test)

    final_model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=best_C,
    )
    final_model.fit(X_train_val_scaled, y_train_val)

    test_probs = final_model.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_probs >= 0.5).astype(int)

    base_acc = (test_pred == y_test).mean()
    base_ll = log_loss(y_test, test_probs)

    print("\n=== Test (2022, model with odds) ===")
    print(f"Accuracy: {base_acc:.3f}")
    print(f"LogLoss:  {base_ll:.3f}")

    # Attach probabilities for simulation
    test = test.copy()
    test["model_prob"] = test_probs

    # ---------------------------------------------------------------------
    # 5) Simulate flat-stake betting with confidence thresholds
    # ---------------------------------------------------------------------
    def simulate(threshold: float):
        """
        Simulate 1u bets on 2022 games, only when model is at least `threshold`
        confident on a side (home or away).

        Returns:
            (threshold, bets, wins, profit, roi)
        """
        stakes = 0.0
        profit = 0.0
        bets = 0
        wins = 0

        for _, row in test.iterrows():
            p_home = row["model_prob"]
            home_win = row["home_win"]
            home_ml = row[HOME_ML_COL]
            away_ml = row[AWAY_ML_COL]

            # Decide side based on which team model thinks is more likely
            if p_home >= 0.5:
                # Consider betting home
                if p_home < threshold:
                    continue
                odds = home_ml
                win = (home_win == 1)
            else:
                # Consider betting away
                p_away = 1.0 - p_home
                if p_away < threshold:
                    continue
                odds = away_ml
                win = (home_win == 0)

            bets += 1
            stakes += 1.0  # flat 1 unit stake

            if win:
                wins += 1
                profit += american_to_return(odds)
            else:
                profit -= 1.0

        if bets == 0:
            return threshold, 0, 0, 0.0, 0.0

        roi = profit / stakes if stakes > 0 else 0.0
        return threshold, bets, wins, profit, roi

    print("\nSimulated ROI on 2022 (model-based, 1u flat stakes):")
    print("Threshold   Bets   HitRate   Profit   ROI")
    print("-----------------------------------------")
    for th in [0.60, 0.65, 0.70, 0.75]:
        th_val, bets, wins, profit, roi = simulate(th)
        hit_rate = wins / bets if bets > 0 else 0.0
        print(f"{th_val:8.2f} {bets:6d} {hit_rate:8.3f} {profit:8.2f} {roi:7.3f}")


if __name__ == "__main__":
    main()
