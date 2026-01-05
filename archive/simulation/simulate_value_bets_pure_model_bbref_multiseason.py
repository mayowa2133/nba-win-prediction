import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


CSV_PATH = "games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv"
TARGET_COL = "home_win"


def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability (no vig removal).
    """
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        # +150 -> 100 / (150 + 100)
        return 100.0 / (odds + 100.0)
    else:
        # -150 -> 150 / (150 + 100)
        return -odds / (-odds + 100.0)


def american_to_profit_per_unit(odds: float) -> float:
    """
    Profit (excluding stake) for 1 unit staked at given American odds.
    Example:
      +150 -> profit 1.5 units on a win
      -150 -> profit 100/150 ≈ 0.666 units on a win
    """
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return odds / 100.0
    else:
        return 100.0 / (-odds)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data and basic odds housekeeping
    # ------------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Ensure we have explicit market ML columns
    if "market_home_ml" not in df.columns:
        print("INFO: 'market_home_ml' not found, using 'home_ml' as market_home_ml.")
        df["market_home_ml"] = df["home_ml"]
    if "market_away_ml" not in df.columns:
        print("INFO: 'market_away_ml' not found, using 'away_ml' as market_away_ml.")
        df["market_away_ml"] = df["away_ml"]

    # Keep only rows where we actually have odds
    df = df[~df["market_home_ml"].isna() & ~df["market_away_ml"].isna()].copy()

    print("Games with odds by season (after dropping NaNs):")
    print(df.groupby("season")[TARGET_COL].count())
    print()

    seasons_with_odds = sorted(df["season"].unique().tolist())
    print("Seasons available with odds:", seasons_with_odds)

    # Walk-forward: start testing once we have at least 2 past seasons
    test_seasons = seasons_with_odds[2:]
    print("Will run walk-forward value sim on seasons:", test_seasons)
    print()

    # ------------------------------------------------------------------
    # 2. Build feature set (drop leakage columns)
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Columns that are hard leaks (post-game or market info)
    leakage_candidates = [
        "home_score",
        "away_score",
        "total",
        "spread",
        "home_ml",
        "away_ml",
        "home_prob_raw",
        "away_prob_raw",
        "market_home_prob",
        "market_away_prob",
        "market_spread",
        "market_total",
        TARGET_COL,  # target must never be in features
    ]

    leak_cols_present = [c for c in leakage_candidates if c in numeric_cols]
    feature_cols = [c for c in numeric_cols if c not in leak_cols_present]

    print(
        f"Total feature columns (pure model + BBRef, "
        f"no odds / no leaks): {len(feature_cols)}"
    )
    print("Example features:", feature_cols[:10], "...")
    print()

    # To collect classification summary
    summary_rows = []
    # To collect all bets from all seasons
    all_bets = []

    # ------------------------------------------------------------------
    # 3. Walk-forward over seasons
    # ------------------------------------------------------------------
    for test_season in test_seasons:
        idx = seasons_with_odds.index(test_season)
        train_seasons = seasons_with_odds[: max(0, idx - 1)]
        val_season = seasons_with_odds[idx - 1]

        print(f"=== Season {test_season} ===")
        print(f"Train seasons: {train_seasons}")
        print(f"Val season:   {val_season}")

        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"] == val_season].copy()
        test_df = df[df["season"] == test_season].copy()

        print(
            f"Split sizes:  train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )

        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_COL].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_COL].values
        X_test = test_df[feature_cols].values
        y_test = test_df[TARGET_COL].values

        # --------------------------------------------------------------
        # 3a. Hyperparameter tuning for C on validation season
        # --------------------------------------------------------------
        Cs = [0.01, 0.10, 0.30, 1.00, 3.00, 10.00]
        best_C = None
        best_val_loss = np.inf

        print("  Tuning C on validation set:")
        for C in Cs:
            clf = LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=2000,
            )
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            val_loss = log_loss(y_val, val_proba)
            val_acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))
            print(
                f"    C={C:4.2f} -> "
                f"Val Accuracy={val_acc:.3f}, LogLoss={val_loss:.3f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_C = C

        print(
            f"  Best C for season {test_season}: "
            f"{best_C} (val logloss={best_val_loss:.3f})"
        )

        # --------------------------------------------------------------
        # 3b. Retrain on train + val, test on test_season
        # --------------------------------------------------------------
        full_train_df = pd.concat([train_df, val_df], axis=0)
        X_full = full_train_df[feature_cols].values
        y_full = full_train_df[TARGET_COL].values

        clf = LogisticRegression(
            C=best_C,
            solver="lbfgs",
            max_iter=2000,
        )
        clf.fit(X_full, y_full)

        test_proba = clf.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)
        test_loss = log_loss(y_test, test_proba)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"  Classification on season {test_season}:")
        print(f"    Accuracy: {test_acc:.3f}")
        print(f"    LogLoss:  {test_loss:.3f}")

        summary_rows.append(
            {
                "season": int(test_season),
                "games": len(test_df),
                "accuracy": float(test_acc),
                "logloss": float(test_loss),
            }
        )

        # --------------------------------------------------------------
        # 3c. Build bet-level table for this season
        # --------------------------------------------------------------
        test_df = test_df.copy()
        test_df["model_p_home"] = test_proba
        test_df["model_p_away"] = 1.0 - test_df["model_p_home"]

        test_df["market_p_home"] = test_df["market_home_ml"].apply(
            american_to_implied_prob
        )
        test_df["market_p_away"] = test_df["market_away_ml"].apply(
            american_to_implied_prob
        )

        test_df["profit_home"] = test_df["market_home_ml"].apply(
            american_to_profit_per_unit
        )
        test_df["profit_away"] = test_df["market_away_ml"].apply(
            american_to_profit_per_unit
        )

        bets_rows = []
        for _, row in test_df.iterrows():
            # home side
            bets_rows.append(
                {
                    "season": int(row["season"]),
                    "side": "home",
                    "model_prob": float(row["model_p_home"]),
                    "market_prob": float(row["market_p_home"]),
                    "edge": float(row["model_p_home"] - row["market_p_home"]),
                    "is_win": int(row[TARGET_COL] == 1),
                    "profit_if_win": float(row["profit_home"]),
                }
            )
            # away side
            bets_rows.append(
                {
                    "season": int(row["season"]),
                    "side": "away",
                    "model_prob": float(row["model_p_away"]),
                    "market_prob": float(row["market_p_away"]),
                    "edge": float(row["model_p_away"] - row["market_p_away"]),
                    "is_win": int(row[TARGET_COL] == 0),
                    "profit_if_win": float(row["profit_away"]),
                }
            )

        bets_df = pd.DataFrame(bets_rows)
        all_bets.append(bets_df)

        # Per-season value betting report
        print("  Value betting (flat 1u) for this season:")
        print("    EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
        print("    " + "-" * 50)
        for thr in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
            subset = bets_df[bets_df["edge"] >= thr]
            n_bets = len(subset)
            if n_bets == 0:
                hit_rate = np.nan
                avg_edge = np.nan
                profit = 0.0
                roi = np.nan
            else:
                wins = subset["is_win"].sum()
                hit_rate = wins / n_bets
                profit = (
                    subset["is_win"] * subset["profit_if_win"]
                    - (1 - subset["is_win"])
                ).sum()
                avg_edge = subset["edge"].mean()
                roi = profit / n_bets
            print(
                f"      {thr:4.2f}  {n_bets:6d}   "
                f"{hit_rate:7.3f}   {avg_edge:7.3f}  "
                f"{profit:8.2f}  {roi:7.3f}"
            )
        print()

    # ------------------------------------------------------------------
    # 4. Classification summary across seasons
    # ------------------------------------------------------------------
    print("=== Classification summary across test seasons ===")
    print("Season   Games   Accuracy   LogLoss")
    print("------------------------------------")
    total_games = 0
    weighted_acc = 0.0
    weighted_loss = 0.0
    for row in summary_rows:
        print(
            f" {row['season']:4d}  {row['games']:7d}    "
            f"{row['accuracy']:.3f}     {row['logloss']:.3f}"
        )
        total_games += row["games"]
        weighted_acc += row["accuracy"] * row["games"]
        weighted_loss += row["logloss"] * row["games"]

    overall_acc = weighted_acc / total_games
    overall_loss = weighted_loss / total_games
    print("------------------------------------")
    print(f"Overall  {total_games:7d}    {overall_acc:.3f}     {overall_loss:.3f}")
    print()

    # ------------------------------------------------------------------
    # 5. Aggregated value betting across all seasons
    # ------------------------------------------------------------------
    all_bets_df = pd.concat(all_bets, ignore_index=True)

    print("=== Aggregated VALUE betting across all test seasons ===")
    print("EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
    print("--------------------------------------------------")
    for thr in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
        subset = all_bets_df[all_bets_df["edge"] >= thr]
        n_bets = len(subset)
        if n_bets == 0:
            hit_rate = np.nan
            avg_edge = np.nan
            profit = 0.0
            roi = np.nan
        else:
            wins = subset["is_win"].sum()
            hit_rate = wins / n_bets
            profit = (
                subset["is_win"] * subset["profit_if_win"]
                - (1 - subset["is_win"])
            ).sum()
            avg_edge = subset["edge"].mean()
            roi = profit / n_bets

        print(
            f"  {thr:4.2f}  {n_bets:6d}    "
            f"{hit_rate:7.3f}   {avg_edge:7.3f}  "
            f"{profit:8.2f}  {roi:7.3f}"
        )


if __name__ == "__main__":
    main()
