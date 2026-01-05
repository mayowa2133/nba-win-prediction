#!/usr/bin/env python3
"""
simulate_value_bets_pure_model_bbref.py

Train a *pure* model (no odds features) using game features + BBRef team talent,
then simulate value betting vs the market on the 2022 season (if odds available).

- Data: games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv
- Train: seasons 2016–2019
- Val:   seasons 2020–2021
- Test:  season  2022
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CSV_PATH = "games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv"

HOME_ML_COL = "market_home_ml"
AWAY_ML_COL = "market_away_ml"
TARGET_COL = "home_win"

TRAIN_SEASONS = [2016, 2017, 2018, 2019]
VAL_SEASONS   = [2020, 2021]
TEST_SEASON   = 2022

EDGE_THRESHOLDS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def american_odds_to_prob(ml: np.ndarray) -> np.ndarray:
    """
    Convert American moneyline odds to implied win probability.
    ml: array-like of American odds (e.g. -150, +200)

    returns: numpy array of probabilities in [0,1]
    """
    ml = np.asarray(ml, dtype=float)
    out = np.full_like(ml, np.nan, dtype=float)

    pos = ml > 0
    neg = ml < 0

    # For positive odds: prob = 100 / (ml + 100)
    out[pos] = 100.0 / (ml[pos] + 100.0)

    # For negative odds: prob = (-ml) / ((-ml) + 100)
    out[neg] = (-ml[neg]) / ((-ml[neg]) + 100.0)

    return out


def one_unit_profit(ml: float, win: bool) -> float:
    """
    Profit for a 1-unit stake on a given American moneyline.
    - If win: profit = payout - stake
    - If lose: -1.0
    """
    if np.isnan(ml):
        return 0.0

    if win:
        if ml > 0:
            # +200 => win 2 units on 1 staked
            return ml / 100.0
        else:
            # -150 => win 100/150 units on 1 staked
            return 100.0 / (-ml)
    else:
        return -1.0


def build_feature_matrix(df: pd.DataFrame) -> list:
    """
    Choose pure model feature columns:
    - Use all numeric columns
    - Exclude:
        * identifiers / meta
        * target
        * all market/odds columns (start with 'market_')
    """
    exclude_cols = {
        "game_id",
        "date",
        "status",
        "postseason",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "season",      # avoid season leakage
    }

    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col == TARGET_COL:
            continue
        if col.startswith("market_"):
            continue
        # Use numeric columns only
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def simulate_value_bets_for_thresholds(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    p_home: np.ndarray,
    thresholds=EDGE_THRESHOLDS
):
    """
    For each edge threshold:
      - For each game, compute model edges for home and away vs market
      - If max(edge_home, edge_away) >= threshold, bet 1u on that side
      - Compute hit rate, avg edge, profit, and ROI.
    """
    # Ensure we only use rows with valid moneylines
    mask = (~df_test[HOME_ML_COL].isna()) & (~df_test[AWAY_ML_COL].isna())
    df = df_test.loc[mask].copy()
    y = y_test[mask]
    p_home = p_home[mask]

    home_ml = df[HOME_ML_COL].to_numpy(dtype=float)
    away_ml = df[AWAY_ML_COL].to_numpy(dtype=float)

    # Market implied probabilities
    p_mkt_home = american_odds_to_prob(home_ml)
    p_mkt_away = american_odds_to_prob(away_ml)

    # Model probabilities for both sides
    p_model_home = p_home
    p_model_away = 1.0 - p_model_home

    edges_home = p_model_home - p_mkt_home
    edges_away = p_model_away - p_mkt_away

    is_home_win = (y == 1)

    print("\nSimulated VALUE betting on test season (pure model + BBRef, 1u flat stakes):")
    print("EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
    print("--------------------------------------------------")

    for thr in thresholds:
        total_bets = 0
        total_profit = 0.0
        edges_taken = []
        wins = 0

        for i in range(len(df)):
            # Decide which side to bet (if any)
            best_side = None
            best_edge = -1e9

            if edges_home[i] >= thr and edges_home[i] > best_edge:
                best_side = "home"
                best_edge = edges_home[i]

            if edges_away[i] >= thr and edges_away[i] > best_edge:
                best_side = "away"
                best_edge = edges_away[i]

            if best_side is None:
                continue  # no bet

            total_bets += 1
            edges_taken.append(best_edge)

            if best_side == "home":
                win = bool(is_home_win[i])
                ml = home_ml[i]
            else:
                win = not bool(is_home_win[i])
                ml = away_ml[i]

            profit = one_unit_profit(ml, win)
            total_profit += profit
            if win:
                wins += 1

        if total_bets == 0:
            print(f"{thr:7.2f}      0      N/A      N/A      0.00   0.000")
        else:
            hit_rate = wins / total_bets
            avg_edge = float(np.mean(edges_taken)) if edges_taken else 0.0
            roi = total_profit / total_bets
            print(
                f"{thr:7.2f}  {total_bets:5d}   {hit_rate:7.3f}   "
                f"{avg_edge:7.3f}  {total_profit:7.2f}  {roi:7.3f}"
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading games from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # -----------------------------------------------------------------
    # Make sure odds columns exist (handle different naming schemes)
    # -----------------------------------------------------------------
    cols = set(df.columns)

    # If market_* missing but raw home_ml/away_ml exist, create them
    if HOME_ML_COL not in cols and "home_ml" in cols:
        print("INFO: 'market_home_ml' not found, using 'home_ml' as market_home_ml.")
        df[HOME_ML_COL] = df["home_ml"]

    if AWAY_ML_COL not in cols and "away_ml" in cols:
        print("INFO: 'market_away_ml' not found, using 'away_ml' as market_away_ml.")
        df[AWAY_ML_COL] = df["away_ml"]

    cols = set(df.columns)
    have_market = (HOME_ML_COL in cols) and (AWAY_ML_COL in cols)

    if have_market:
        # Keep only rows with odds (like earlier sims)
        df = df[~df[HOME_ML_COL].isna() & ~df[AWAY_ML_COL].isna()].copy()
        print("Games with odds by season (after dropping NaNs):")
        print(df.groupby("season")[TARGET_COL].count())
    else:
        print("WARNING: No market odds columns found (neither 'market_home_ml'/'market_away_ml' nor 'home_ml'/'away_ml').")
        print("         Will train model + report accuracy/logloss, but skip value betting simulation.")

    print()

    # Build feature set
    feature_cols = build_feature_matrix(df)
    print(f"Total feature columns (pure model + BBRef): {len(feature_cols)}")
    print("Example features:", feature_cols[:10], "...")
    print()

    # Train/val/test split by season
    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask   = df["season"].isin(VAL_SEASONS)
    test_mask  = df["season"] == TEST_SEASON

    X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train = df.loc[train_mask, TARGET_COL].to_numpy(dtype=int)

    X_val   = df.loc[val_mask, feature_cols].to_numpy(dtype=float)
    y_val   = df.loc[val_mask, TARGET_COL].to_numpy(dtype=int)

    X_test  = df.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test  = df.loc[test_mask, TARGET_COL].to_numpy(dtype=int)

    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    print()

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Tune C on validation set
    Cs = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_C = None
    best_val_logloss = np.inf

    print("Tuning C on validation set (pure model + BBRef, no odds features):")
    for C in Cs:
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )
        clf.fit(X_train_scaled, y_train)

        val_proba = clf.predict_proba(X_val_scaled)[:, 1]  # P(home win)
        val_loss = log_loss(y_val, val_proba)
        val_acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))

        print(f"  C={C:4.2f} -> Val Accuracy={val_acc:.3f}, LogLoss={val_loss:.3f}")

        if val_loss < best_val_logloss:
            best_val_logloss = val_loss
            best_C = C

    print(f"\nBest C based on val logloss: {best_C} (logloss={best_val_logloss:.3f})\n")

    # Retrain on train+val with best C
    trainval_mask = df["season"].isin(TRAIN_SEASONS + VAL_SEASONS)
    X_trainval = df.loc[trainval_mask, feature_cols].to_numpy(dtype=float)
    y_trainval = df.loc[trainval_mask, TARGET_COL].to_numpy(dtype=int)
    X_trainval_scaled = scaler.fit_transform(X_trainval)  # refit scaler on train+val
    X_test_scaled = scaler.transform(X_test)              # re-transform test with new scaler

    final_clf = LogisticRegression(
        C=best_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    final_clf.fit(X_trainval_scaled, y_trainval)

    # Evaluate on test season (2022)
    test_proba = final_clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_proba >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, test_preds)
    test_loss = log_loss(y_test, test_proba)

    print(f"=== Test season {TEST_SEASON} (pure model + BBRef) ===")
    print(f"Accuracy: {test_acc:.3f}")
    print(f"LogLoss:  {test_loss:.3f}")

    # Run value-bet simulation on the 2022 test season, if we have market odds
    if have_market:
        simulate_value_bets_for_thresholds(
            df_test=df.loc[test_mask].copy(),
            y_test=y_test,
            p_home=test_proba,
            thresholds=EDGE_THRESHOLDS,
        )
    else:
        print("\nSkipping value-bet simulation because no market odds columns were found.")


if __name__ == "__main__":
    main()
