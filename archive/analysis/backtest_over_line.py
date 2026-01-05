#!/usr/bin/env python
"""
backtest_over_line.py

Backtest the regression-based P(OVER) engine on historical data.

Idea:
- Use data/player_points_features.csv (same file as for training).
- Use models/points_regression.pkl (RandomForestRegressor + sigma).
- Restrict to holdout seasons (>= season_min, default 2023).
- For each player-game, define a synthetic "line" as pts_roll5 rounded
  to the nearest 0.5 (a naive book-style line based on recent scoring).
- Compute P(OVER line) using the regression model (mu) + sigma
  with a normal approximation.
- Compare predicted P(OVER) vs actual outcome (pts > line).

Outputs:
- Overall Brier score for P(OVER).
- Calibration table: for bins of predicted P(OVER), show:
    - count
    - average predicted P
    - actual over rate
- Simple "edge" stats: what happens if we only take bets where
  model P(OVER) >= 0.60, 0.65, 0.70, etc.
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

# Paths
FEATURES_CSV = Path("data/player_points_features.csv")
MODEL_PATH = Path("models/points_regression.pkl")

# Default feature columns (used if bundle doesn't specify its own)
FEATURE_COLS_DEFAULT = [
    "minutes_roll5",
    "minutes_roll15",
    "pts_roll5",
    "pts_roll15",
    "reb_roll5",
    "reb_roll15",
    "ast_roll5",
    "ast_roll15",
    "fg3m_roll5",
    "fg3m_roll15",
    "fg3a_roll5",
    "fg3a_roll15",
    "fga_roll5",
    "fga_roll15",
    "fta_roll5",
    "fta_roll15",
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
    "days_since_last_game",
    "is_home",
]


def normal_over_probs(mu: np.ndarray, sigma: float, line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized normal approximation:
      pts ~ N(mu, sigma^2)
      return (p_over, p_under) for given line.

    mu and line should be 1D numpy arrays of equal length.
    """
    if sigma <= 0:
        # Degenerate, but handle gracefully
        p_over = (mu > line).astype(float)
        p_under = 1.0 - p_over
        return p_over, p_under

    z = (line - mu) / sigma
    # use math.erf via numpy.vectorize since np.erf isn't available
    z_scaled = z / math.sqrt(2.0)
    erf_vec = np.vectorize(math.erf)(z_scaled)

    p_under = 0.5 * (1.0 + erf_vec)
    p_over = 1.0 - p_under
    return p_over, p_under



def load_regression_model(path: Path = MODEL_PATH) -> Tuple[Any, float, List[str]]:
    """
    Load the regression bundle {model, sigma, feature_cols} from pickle.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle["model"]
        sigma = float(bundle.get("sigma", 7.0))
        feature_cols = bundle.get("feature_cols", FEATURE_COLS_DEFAULT)
    else:
        # Fallback if only model was pickled
        model = bundle
        sigma = 7.0
        feature_cols = FEATURE_COLS_DEFAULT

    return model, sigma, list(feature_cols)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest regression-based P(OVER line) on historical seasons."
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season (start year) to include in backtest (default 2023).",
    )
    parser.add_argument(
        "--season-max",
        type=int,
        default=9999,
        help="Maximum season (start year) to include in backtest (default: no upper bound).",
    )
    parser.add_argument(
        "--min-line",
        type=float,
        default=8.0,
        help="Minimum synthetic line to include (default 8.0). "
             "This filters out bench guys who would never have player props.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of probability bins for calibration (default 10).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_CSV}")

    print(f"Loading features from {FEATURES_CSV} ...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns.")

    # Basic sanity check
    required_cols = {"season", "pts", "pts_roll5"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in features CSV: {missing}")

    # Restrict to holdout seasons
    mask_season = (df["season"] >= args.season_min) & (df["season"] <= args.season_max)
    df_val = df.loc[mask_season].copy()
    print(f"Using seasons in [{args.season_min}, {args.season_max}] "
          f"-> {len(df_val):,} rows for backtest.")

    if df_val.empty:
        print("No rows available in the chosen season range. Exiting.")
        return

    # ------------------------------------------------------------------
    # Build synthetic lines from pts_roll5
    # ------------------------------------------------------------------
    # Naive "book line": last-5-games average points, rounded to nearest 0.5
    df_val["line_naive"] = (df_val["pts_roll5"] * 2.0).round() / 2.0

    # Filter to lines above a threshold (we don't care about 2.5-point props for random bench guys)
    mask_line = df_val["line_naive"] >= args.min_line
    df_bt = df_val.loc[mask_line].copy()
    print(f"After filtering to line >= {args.min_line}, we have {len(df_bt):,} rows.")

    if df_bt.empty:
        print("No rows left after min-line filtering. Try lowering --min-line.")
        return

    # Drop rows with NaNs in features or target
    df_bt = df_bt.dropna(subset=["pts", "line_naive"])
    print(f"After dropping rows with NaN pts/line: {len(df_bt):,} rows.")

    # ------------------------------------------------------------------
    # Load regression model
    # ------------------------------------------------------------------
    print(f"\nLoading regression model from {MODEL_PATH} ...")
    model, sigma, feature_cols = load_regression_model(MODEL_PATH)
    print(f"Model uses {len(feature_cols)} feature columns:")
    for col in feature_cols:
        print(f"  - {col}")
    print(f"Sigma used for normal approximation: {sigma:.3f}")

    # Ensure all required feature columns exist
    missing_feats = set(feature_cols) - set(df_bt.columns)
    if missing_feats:
        raise RuntimeError(f"Features CSV is missing required model feature columns: {missing_feats}")

    # ------------------------------------------------------------------
    # Build design matrix & predict mu
    # ------------------------------------------------------------------
    X = df_bt[feature_cols].to_numpy()
    y_actual_pts = df_bt["pts"].to_numpy()
    lines = df_bt["line_naive"].to_numpy()

    print("\nPredicting expected points (mu) for each game...")
    mu = model.predict(X).astype(float)

    # ------------------------------------------------------------------
    # Compute P(OVER line) using normal approximation
    # ------------------------------------------------------------------
    print("Computing P(OVER line) using normal approximation...")
    p_over, p_under = normal_over_probs(mu, sigma, lines)

    # Ground truth: did the player actually go over this synthetic line?
    y_over = (y_actual_pts > lines).astype(int)

    # ------------------------------------------------------------------
    # Overall Brier score
    # ------------------------------------------------------------------
    brier = float(np.mean((p_over - y_over) ** 2))
    print("\n=== Overall backtest metrics (synthetic line = pts_roll5 rounded) ===")
    print(f"Total samples:      {len(df_bt):,}")
    print(f"Brier score (lower better): {brier:.4f}")

    # ------------------------------------------------------------------
    # "Edge" stats: if we only bet when model P(OVER) is high
    # ------------------------------------------------------------------
    print("\n=== Edge-style stats: P(OVER) >= threshold ===")
    for thr in [0.55, 0.60, 0.65, 0.70]:
        mask = p_over >= thr
        n = int(mask.sum())
        if n == 0:
            print(f"P(OVER) >= {thr:.2f}: no samples")
            continue
        hit_rate = float(y_over[mask].mean())
        avg_p = float(p_over[mask].mean())
        print(
            f"P(OVER) >= {thr:.2f}: n={n:6d}, "
            f"avg_pred={avg_p:.3f}, actual_over_rate={hit_rate:.3f}"
        )

    # ------------------------------------------------------------------
    # Calibration table by probability bins
    # ------------------------------------------------------------------
    print("\n=== Calibration by predicted P(OVER) bins ===")

    num_bins = max(2, args.bins)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    df_bt["p_over"] = p_over
    df_bt["y_over"] = y_over

    df_bt["p_bin"] = pd.cut(
        df_bt["p_over"],
        bins=bin_edges,
        include_lowest=True,
        right=True,
    )

    grouped = df_bt.groupby("p_bin", observed=True)

    print(f"{'Bin range':>18} | {'Count':>7} | {'Avg p_over':>10} | {'Actual rate':>12}")
    print("-" * 58)
    for bin_label, group in grouped:
        n = len(group)
        if n == 0:
            continue
        avg_p = float(group["p_over"].mean())
        actual_rate = float(group["y_over"].mean())
        print(
            f"{str(bin_label):>18} | {n:7d} | {avg_p:10.3f} | {actual_rate:12.3f}"
        )


if __name__ == "__main__":
    main()
