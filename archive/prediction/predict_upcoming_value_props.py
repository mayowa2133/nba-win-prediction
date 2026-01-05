#!/usr/bin/env python
"""
predict_upcoming_value_props.py

Scan NBA player POINTS props for a given date and identify value bets
using an XGBRegressor trained on player rolling features.

Usage:
  python predict_upcoming_value_props.py 2025-11-24
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import norm  # for normal-approx to get P_over

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

FEATURES_CSV = DATA_DIR / "player_points_features.csv"
MODEL_PATH = MODELS_DIR / "player_points_xgb.json"

# Placeholder path – this would come from a separate fetch_player_props_for_date.py
PLAYER_PROPS_CSV_TEMPLATE = DATA_DIR / "player_props_raw" / "player_props_{date}.csv"


def parse_date(argv) -> str:
    if len(argv) > 1:
        return argv[1]
    # default: today in UTC (you can change)
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def load_model() -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def load_historical_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_market_props_for_date(date_str: str) -> pd.DataFrame:
    """
    TODO: wire this to a real API or CSV.

    For now we assume you have a CSV like:

      date, player_id, player_name, team_abbrev, opp_team_abbrev, market_points,
      over_odds, under_odds

    stored at data/player_props_raw/player_props_YYYY-MM-DD.csv
    """
    path = PLAYER_PROPS_CSV_TEMPLATE.with_name(
        PLAYER_PROPS_CSV_TEMPLATE.name.format(date=date_str)
    )
    df = pd.read_csv(path)

    required = [
        "date",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_team_abbrev",
        "market_points",
        "over_odds",
        "under_odds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in market props CSV: {missing}")

    return df


def american_odds_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability (ignoring vig).
    """
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def get_latest_player_feature_row(
    hist_df: pd.DataFrame, player_id: int
) -> Tuple[pd.Series | None, list]:
    """
    For a given player_id, grab their most recent feature row.
    We assume FEATURES_CSV was built up to yesterday's games.
    """
    df_player = hist_df[hist_df["player_id"] == player_id]
    if df_player.empty:
        return None, []
    df_player = df_player.sort_values("game_date")
    row = df_player.iloc[-1]
    id_cols = [
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_team_abbrev",
        "is_home",
        "pts",  # target
    ]
    feature_cols = [c for c in hist_df.columns if c not in id_cols]
    return row[feature_cols], feature_cols


def prob_over_normal(mean: float, std: float, line: float) -> float:
    """
    Approximate P(points > line) assuming normal distribution with (mean, std).
    We’ll do strict > line; for half-point lines this is fine.
    """
    if std <= 0:
        # Avoid degenerate cases; if std ~ 0, treat it as coinflip around mean
        return 0.5 if mean <= line else 1.0
    z = (line - mean) / std
    # P(X > line) = 1 - Phi(z)
    return 1.0 - norm.cdf(z)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    date_str = parse_date(argv)
    print(f"\n=== Player points prop value scan for {date_str} ===\n")

    hist_df = load_historical_features()
    model = load_model()

    # Rough global sigma: std of residuals on historical data.
    # You can replace this with player-specific rolling std if you want.
    target = "pts"
    id_cols = [
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_team_abbrev",
        "is_home",
    ]
    feature_cols_all = [c for c in hist_df.columns if c not in id_cols + [target]]
    X_all = hist_df[feature_cols_all].values
    y_all = hist_df[target].values
    preds_all = model.predict(X_all)
    global_resid_std = np.std(y_all - preds_all)
    print(f"Global residual std for points: {global_resid_std:.3f}\n")

    market_df = load_market_props_for_date(date_str)

    rows = []

    for _, mrow in market_df.iterrows():
        player_id = int(mrow["player_id"])
        feature_row, feature_cols = get_latest_player_feature_row(hist_df, player_id)
        if feature_row is None:
            # Not enough history; skip
            continue

        X = feature_row.values.reshape(1, -1)
        pred_mean = float(model.predict(X)[0])

        # Here we could try a player-specific std (e.g. pts_rolling_std_10)
        # For now, use global residual std as a fallback.
        line = float(mrow["market_points"])
        std = float(global_resid_std)

        p_over = prob_over_normal(pred_mean, std, line)
        p_under = 1.0 - p_over

        implied_over = american_odds_to_implied_prob(float(mrow["over_odds"]))
        implied_under = american_odds_to_implied_prob(float(mrow["under_odds"]))

        edge_over = p_over - implied_over
        edge_under = p_under - implied_under

        rows.append(
            {
                "date": date_str,
                "player_id": player_id,
                "player_name": mrow["player_name"],
                "team_abbrev": mrow["team_abbrev"],
                "opp_team_abbrev": mrow["opp_team_abbrev"],
                "market_points": line,
                "pred_mean_points": pred_mean,
                "std_points": std,
                "over_odds": mrow["over_odds"],
                "under_odds": mrow["under_odds"],
                "p_over_model": p_over,
                "p_under_model": p_under,
                "p_over_implied": implied_over,
                "p_under_implied": implied_under,
                "edge_over": edge_over,
                "edge_under": edge_under,
            }
        )

    if not rows:
        print("No props could be evaluated (maybe no CSV or no matching features).")
        return

    out_df = pd.DataFrame(rows)

    # Sort by best edge on either side
    out_df["best_side"] = np.where(out_df["edge_over"] >= out_df["edge_under"], "over", "under")
    out_df["best_edge"] = out_df[["edge_over", "edge_under"]].max(axis=1)

    out_df = out_df.sort_values("best_edge", ascending=False)

    print("Top 20 props by model edge:")
    print(
        out_df[
            [
                "player_name",
                "team_abbrev",
                "opp_team_abbrev",
                "market_points",
                "pred_mean_points",
                "best_side",
                "best_edge",
                "over_odds",
                "under_odds",
                "p_over_model",
                "p_over_implied",
            ]
        ].head(20).to_string(index=False)
    )

    # Save results
    OUT_DIR = Path("predictions")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"player_points_value_props_{date_str}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved full prop table to {out_path}")


if __name__ == "__main__":
    main()
