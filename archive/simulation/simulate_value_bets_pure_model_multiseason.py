# simulate_value_bets_pure_model_multiseason.py

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# Path to the dataset that includes rolling+last10+prev-season stats AND odds
DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")

# PURE basketball feature columns (NO odds in features)
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
]

# Moneyline columns (from oddsData)
HOME_ML_COL = "home_ml"
AWAY_ML_COL = "away_ml"


# ---------------------------------------------------------------------
# Helpers for odds ↔ probabilities
# ---------------------------------------------------------------------
def american_to_return(odds_american: float) -> float:
    """
    Convert American odds to profit for 1 unit stake.
    Returns net profit (excluding stake) if the bet wins.

    Examples:
        +150 -> 1.5 units profit
        -150 -> 0.666... units profit
    """
    if odds_american > 0:
        return odds_american / 100.0
    else:
        return 100.0 / abs(odds_american)


def american_implied_prob(odds_american: float) -> float:
    """
    Convert American odds to implied probability (including vig).
    This is the break-even probability for that single side.
    """
    if odds_american > 0:
        return 100.0 / (odds_american + 100.0)
    else:
        return abs(odds_american) / (abs(odds_american) + 100.0)


# ---------------------------------------------------------------------
# Value simulation for a single season
# ---------------------------------------------------------------------
def simulate_value_bets_for_season(test_df: pd.DataFrame, edge_thresholds):
    """
    Given a test_df for ONE season containing:
        - 'model_prob' (P(home win) from model)
        - 'home_win'   (0/1)
        - home/away moneylines

    Simulate flat 1u value bets for multiple edge thresholds.

    Returns:
        stats_per_thr: dict[thr] -> dict with
            bets, wins, edge_sum, profit
    """
    # Ensure thresholds are sorted ascending
    thresholds = sorted(edge_thresholds)

    stats_per_thr = {
        thr: {"bets": 0, "wins": 0, "edge_sum": 0.0, "profit": 0.0}
        for thr in thresholds
    }

    for _, row in test_df.iterrows():
        p_home = row["model_prob"]
        home_win = int(row["home_win"])
        home_ml = row[HOME_ML_COL]
        away_ml = row[AWAY_ML_COL]

        # Skip if odds are missing
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue

        # Break-even probs from market odds
        p_be_home = american_implied_prob(home_ml)
        p_be_away = american_implied_prob(away_ml)

        p_away = 1.0 - p_home

        edge_home = p_home - p_be_home
        edge_away = p_away - p_be_away

        # No positive edge on either side → skip game
        if edge_home <= 0 and edge_away <= 0:
            continue

        # Choose side with larger edge
        if edge_home >= edge_away:
            chosen_edge = edge_home
            odds = home_ml
            win = (home_win == 1)
        else:
            chosen_edge = edge_away
            odds = away_ml
            win = (home_win == 0)

        # Update all thresholds this bet qualifies for
        for thr in thresholds:
            if chosen_edge < thr:
                # thresholds are sorted ascending, so we can break here
                break
            stats = stats_per_thr[thr]
            stats["bets"] += 1
            stats["edge_sum"] += chosen_edge
            if win:
                stats["wins"] += 1
                stats["profit"] += american_to_return(odds)
            else:
                stats["profit"] -= 1.0

    return stats_per_thr


def main():
    # -----------------------------------------------------------------
    # 1) Load and clean data
    # -----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Create label: did home team win?
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Keep games where moneylines exist and season <= 2022
    mask = (
        (df["season"] <= 2022)
        & (~df[HOME_ML_COL].isna())
        & (~df[AWAY_ML_COL].isna())
    )
    df = df[mask].copy()

    # Drop any rows missing features or scores
    df = df.dropna(
        subset=FEATURE_COLS + [HOME_ML_COL, AWAY_ML_COL, "home_score", "away_score"]
    )

    print("Games with odds by season (after dropping NaNs):")
    print(df["season"].value_counts().sort_index())
    print()

    all_seasons = sorted(df["season"].unique())
    print(f"Seasons available with odds: {all_seasons}")

    # We'll do walk-forward for later seasons (e.g., 2018–2022)
    # so that we always have train+val behind each test season.
    test_seasons = [s for s in all_seasons if s >= 2018]
    print(f"Will run walk-forward value sim on seasons: {test_seasons}\n")

    # Edge thresholds for value betting
    edge_thresholds = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

    # To accumulate stats across seasons
    global_value_stats = {
        thr: {"bets": 0, "wins": 0, "edge_sum": 0.0, "profit": 0.0}
        for thr in edge_thresholds
    }

    # Also track classification metrics per season
    season_class_results = []

    # -----------------------------------------------------------------
    # 2) Walk-forward: train/val/test per season
    # -----------------------------------------------------------------
    for test_season in test_seasons:
        # Define train and val seasons for this test season
        train_seasons = [s for s in all_seasons if s <= test_season - 2]
        val_seasons = [test_season - 1] if (test_season - 1) in all_seasons else []

        if not train_seasons or not val_seasons:
            print(
                f"Skipping season {test_season}: insufficient train ({train_seasons}) or val ({val_seasons}) seasons."
            )
            continue

        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"].isin(val_seasons)].copy()
        test_df = df[df["season"] == test_season].copy()

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print(
                f"Skipping season {test_season}: empty split sizes "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})."
            )
            continue

        print(f"\n=== Season {test_season} ===")
        print(f"Train seasons: {train_seasons}")
        print(f"Val season:   {val_seasons[0]}")
        print(f"Split sizes:  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["home_win"].values

        X_val = val_df[FEATURE_COLS].values
        y_val = val_df["home_win"].values

        X_test = test_df[FEATURE_COLS].values
        y_test = test_df["home_win"].values

        # -------------------------------------------------------------
        # Scale + tune C on validation
        # -------------------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        Cs = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
        best_C = None
        best_ll = None

        print("  Tuning C on validation set:")
        for C in Cs:
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=C,
            )
            model.fit(X_train_scaled, y_train)
            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            ll = log_loss(y_val, val_probs)
            acc = (val_probs >= 0.5).mean()
            print(f"    C={C:5.2f} -> Val Accuracy={acc:.3f}, LogLoss={ll:.3f}")
            if best_ll is None or ll < best_ll:
                best_ll = ll
                best_C = C

        print(f"  Best C for season {test_season}: {best_C} (val logloss={best_ll:.3f})")

        # -------------------------------------------------------------
        # Train final model on train+val, evaluate on that season
        # -------------------------------------------------------------
        train_val_df = pd.concat([train_df, val_df], axis=0)
        X_train_val = train_val_df[FEATURE_COLS].values
        y_train_val = train_val_df["home_win"].values

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
        acc = (test_pred == y_test).mean()
        ll = log_loss(y_test, test_probs)

        season_class_results.append(
            {"season": test_season, "accuracy": acc, "logloss": ll, "num_games": len(test_df)}
        )

        print(f"  Classification on season {test_season}:")
        print(f"    Accuracy: {acc:.3f}")
        print(f"    LogLoss:  {ll:.3f}")

        # Attach_probs to test_df for value simulation
        test_df = test_df.copy()
        test_df["model_prob"] = test_probs

        # -------------------------------------------------------------
        # Value betting simulation for this season
        # -------------------------------------------------------------
        season_value_stats = simulate_value_bets_for_season(test_df, edge_thresholds)

        print("  Value betting (flat 1u) for this season:")
        print("    EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
        print("    --------------------------------------------------")
        for thr in edge_thresholds:
            stats = season_value_stats[thr]
            bets = stats["bets"]
            wins = stats["wins"]
            profit = stats["profit"]
            edge_sum = stats["edge_sum"]
            hit_rate = wins / bets if bets > 0 else 0.0
            avg_edge = edge_sum / bets if bets > 0 else 0.0
            roi = profit / bets if bets > 0 else 0.0
            print(
                f"    {thr:7.2f} {bets:6d} {hit_rate:9.3f} {avg_edge:9.3f} "
                f"{profit:8.2f} {roi:7.3f}"
            )

            # Accumulate into global stats
            g = global_value_stats[thr]
            g["bets"] += bets
            g["wins"] += wins
            g["profit"] += profit
            g["edge_sum"] += edge_sum

    # -----------------------------------------------------------------
    # 3) Summary across all seasons
    # -----------------------------------------------------------------
    print("\n=== Classification summary across test seasons ===")
    print("Season   Games   Accuracy   LogLoss")
    print("------------------------------------")
    total_games = 0
    weighted_acc_sum = 0.0
    weighted_ll_sum = 0.0
    for res in season_class_results:
        s = res["season"]
        n = res["num_games"]
        a = res["accuracy"]
        ll = res["logloss"]
        total_games += n
        weighted_acc_sum += a * n
        weighted_ll_sum += ll * n
        print(f"{s}      {n:5d}     {a:.3f}     {ll:.3f}")

    if total_games > 0:
        overall_acc = weighted_acc_sum / total_games
        overall_ll = weighted_ll_sum / total_games
        print("------------------------------------")
        print(f"Overall {total_games:5d}     {overall_acc:.3f}     {overall_ll:.3f}")

    print("\n=== Aggregated VALUE betting across all test seasons ===")
    print("EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
    print("--------------------------------------------------")
    for thr in edge_thresholds:
        g = global_value_stats[thr]
        bets = g["bets"]
        wins = g["wins"]
        profit = g["profit"]
        edge_sum = g["edge_sum"]
        hit_rate = wins / bets if bets > 0 else 0.0
        avg_edge = edge_sum / bets if bets > 0 else 0.0
        roi = profit / bets if bets > 0 else 0.0
        print(
            f"{thr:7.2f} {bets:6d} {hit_rate:9.3f} {avg_edge:9.3f} "
            f"{profit:8.2f} {roi:7.3f}"
        )


if __name__ == "__main__":
    main()
