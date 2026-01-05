import pandas as pd
import numpy as np
import sys
from typing import List, Dict
from sklearn.metrics import accuracy_score, log_loss

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    print("ERROR: xgboost is not installed. Inside your venv run:")
    print("  pip install xgboost")
    sys.exit(1)


GAMES_CSV = "games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv"
LABEL_COL = "home_win"
HOME_ML_COL = "market_home_ml"
AWAY_ML_COL = "market_away_ml"
EDGE_THRESHOLDS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]


def american_to_prob(ml: float) -> float:
    """Convert American moneyline to implied probability."""
    if pd.isna(ml):
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return -ml / (-ml + 100.0)


def profit_from_american(ml: float, won: bool, stake: float = 1.0) -> float:
    """Profit (positive or negative) from a 1-unit bet at American odds."""
    if pd.isna(ml):
        return 0.0
    if not won:
        return -stake
    if ml > 0:
        return stake * (ml / 100.0)
    else:
        return stake * (100.0 / (-ml))


def make_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Build leak-free feature column list:
    - numeric / bool features only
    - exclude label, final scores, odds columns
    """
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    # Columns we must never use as features
    hard_exclude = {
        LABEL_COL,
        "home_score",
        "away_score",
        "status",
        "postseason",
        "is_postseason",  # we'll allow this back explicitly just below if present
        HOME_ML_COL,
        AWAY_ML_COL,
        "home_ml",
        "away_ml",
        "home_spread",
        "home_spread_odds",
        "away_spread",
        "away_spread_odds",
        "total_points",
        "over_odds",
        "under_odds",
    }

    # Some columns we dropped above but actually want as features
    allowed_back = {"is_postseason"}

    feat_cols: List[str] = []
    for c in numeric_cols:
        if c in hard_exclude and c not in allowed_back:
            continue
        feat_cols.append(c)

    return sorted(feat_cols)


def fit_xgb_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """Tune a small set of XGBoost params on the validation set, return best model."""
    param_grid: List[Dict] = [
        {"max_depth": 3, "learning_rate": 0.05},
        {"max_depth": 4, "learning_rate": 0.05},
        {"max_depth": 5, "learning_rate": 0.05},
        {"max_depth": 4, "learning_rate": 0.10},
    ]

    best_model = None
    best_logloss = float("inf")
    best_params = None

    for params in param_grid:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        # NOTE: we DON'T use early_stopping_rounds here to avoid the error
        model.fit(
            X_train,
            y_train,
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, val_proba)
        print(
            f"    max_depth={params['max_depth']}, "
            f"eta={params['learning_rate']:.3f} -> Val LogLoss={ll:.3f}"
        )
        if ll < best_logloss:
            best_logloss = ll
            best_model = model
            best_params = params

    print(
        f"  Best XGB params: max_depth={best_params['max_depth']}, "
        f"eta={best_params['learning_rate']:.3f} (val logloss={best_logloss:.3f})"
    )
    return best_model


def summarize_value_bets(bets_df: pd.DataFrame, label: str) -> None:
    """Print value-betting metrics for multiple edge thresholds."""
    print(f"  Value betting (flat 1u) for {label}:")
    print("    EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
    print("    --------------------------------------------------")
    for thr in EDGE_THRESHOLDS:
        subset = bets_df[bets_df["edge"] >= thr]
        n_bets = len(subset)
        if n_bets == 0:
            print(f"     {thr:0.2f}      0       -         -      0.00    0.000")
            continue
        wins = (subset["profit"] > 0).sum()
        hit_rate = wins / n_bets
        avg_edge = subset["edge"].mean()
        profit = subset["profit"].sum()
        roi = profit / n_bets
        print(
            f"     {thr:0.2f}  {n_bets:5d}    {hit_rate:0.3f}    {avg_edge:0.3f}   "
            f"{profit:7.2f}   {roi:0.3f}"
        )
    print()


def main() -> None:
    print(f"Loaded games from {GAMES_CSV}")
    df = pd.read_csv(GAMES_CSV)

    # Ensure we have market_* columns pointing at the current-book odds
    if HOME_ML_COL not in df.columns:
        print(f"INFO: '{HOME_ML_COL}' not in CSV, using 'home_ml' instead.")
        df[HOME_ML_COL] = df.get("home_ml")
    if AWAY_ML_COL not in df.columns:
        print(f"INFO: '{AWAY_ML_COL}' not in CSV, using 'away_ml' instead.")
        df[AWAY_ML_COL] = df.get("away_ml")

    # Keep only rows where we actually have odds
    df = df[~df[HOME_ML_COL].isna() & ~df[AWAY_ML_COL].isna()].copy()

    seasons_with_odds_raw = sorted(df["season"].unique())
    # Convert np.int64 to plain int for nicer printing / indexing
    seasons_with_odds = [int(s) for s in seasons_with_odds_raw]

    print("Games with odds by season:")
    print(df.groupby("season")[LABEL_COL].size())
    print()

    if len(seasons_with_odds) < 3:
        print("Not enough seasons with odds to run walk-forward.")
        sys.exit(0)

    # Walk-forward: train on seasons < val_season,
    # validate on the season immediately before test_season, test on test_season
    test_seasons = seasons_with_odds[2:]
    print(f"Seasons available with odds: {seasons_with_odds}")
    print(f"Will run walk-forward value sim on test seasons: {test_seasons}")
    print()

    feature_cols = make_feature_columns(df)
    print(
        f"Total feature columns (pure model + BBRef, no odds / no leaks): "
        f"{len(feature_cols)}"
    )
    print("Example features:", feature_cols[:10], "...\n")

    # Prepare numeric feature matrix
    # Cast bools to ints and fill NAs
    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    all_bets_rows = []
    cls_summary = []

    for test_season in test_seasons:
        idx = seasons_with_odds.index(test_season)
        val_season = seasons_with_odds[idx - 1]
        train_seasons = seasons_with_odds[: idx - 1]

        print(f"=== Season {test_season} ===")
        print(f"Train seasons: {train_seasons}")
        print(f"Val season:   {val_season}")

        df_train = df[df["season"].isin(train_seasons)]
        df_val = df[df["season"] == val_season]
        df_test = df[df["season"] == test_season]

        print(
            f"Split sizes:  train={len(df_train)}, val={len(df_val)}, "
            f"test={len(df_test)}"
        )

        X_train = df_train[feature_cols].values
        y_train = df_train[LABEL_COL].values
        X_val = df_val[feature_cols].values
        y_val = df_val[LABEL_COL].values
        X_test = df_test[feature_cols].values
        y_test = df_test[LABEL_COL].values

        print("  Tuning XGBoost on validation set:")
        model = fit_xgb_with_val(X_train, y_train, X_val, y_val)

        # Evaluate classification on test
        p_home_test = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, (p_home_test >= 0.5).astype(int))
        ll = log_loss(y_test, p_home_test)
        print(f"  Classification on season {test_season}:")
        print(f"    Accuracy: {acc:0.3f}")
        print(f"    LogLoss:  {ll:0.3f}")

        cls_summary.append(
            {
                "season": test_season,
                "games": len(df_test),
                "accuracy": acc,
                "logloss": ll,
            }
        )

        # Build per-bet rows for this season
        season_bets_rows = []
        ml_home = df_test[HOME_ML_COL].values
        ml_away = df_test[AWAY_ML_COL].values
        outcomes = y_test

        for i in range(len(df_test)):
            p_h = float(p_home_test[i])
            p_a = 1.0 - p_h

            m_h = float(ml_home[i])
            m_a = float(ml_away[i])

            market_p_h = american_to_prob(m_h)
            market_p_a = american_to_prob(m_a)

            # Home side
            edge_h = p_h - market_p_h
            profit_h = profit_from_american(m_h, bool(outcomes[i] == 1))
            season_bets_rows.append(
                {
                    "season": test_season,
                    "side": "home",
                    "edge": edge_h,
                    "profit": profit_h,
                }
            )

            # Away side (win if home loses)
            edge_a = p_a - market_p_a
            profit_a = profit_from_american(m_a, bool(outcomes[i] == 0))
            season_bets_rows.append(
                {
                    "season": test_season,
                    "side": "away",
                    "edge": edge_a,
                    "profit": profit_a,
                }
            )

        season_bets_df = pd.DataFrame(season_bets_rows)
        summarize_value_bets(season_bets_df, label=f"season {test_season}")
        all_bets_rows.extend(season_bets_rows)

    # Classification summary across seasons
    print("=== Classification summary across test seasons ===")
    print("Season   Games   Accuracy   LogLoss")
    print("------------------------------------")
    total_games = 0
    weighted_ll = 0.0
    for row in cls_summary:
        print(
            f" {row['season']:4d}  {row['games']:6d}   "
            f"{row['accuracy']:0.3f}     {row['logloss']:0.3f}"
        )
        total_games += row["games"]
        weighted_ll += row["logloss"] * row["games"]
    if total_games > 0:
        overall_acc = sum(r["accuracy"] * r["games"] for r in cls_summary) / total_games
        overall_ll = weighted_ll / total_games
    else:
        overall_acc = 0.0
        overall_ll = float("nan")
    print("------------------------------------")
    print(f"Overall  {total_games:6d}   {overall_acc:0.3f}     {overall_ll:0.3f}")
    print()

    # Aggregated value betting across all seasons
    if all_bets_rows:
        all_bets_df = pd.DataFrame(all_bets_rows)
        print("=== Aggregated VALUE betting across all test seasons ===")
        summarize_value_bets(all_bets_df, label="ALL seasons combined")
    else:
        print("No bets generated at all; something is off.")
        print()


if __name__ == "__main__":
    main()
