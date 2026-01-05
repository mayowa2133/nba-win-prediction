#!/usr/bin/env python
"""
analyze_points_residuals.py

Small analysis script to slice residuals from data/points_regression_val_preds.csv.

Expected columns (from build_points_regression.py export):
  - y_true
  - y_pred
  - residual
  - season, game_date, game_id, player_id, player_name, team_abbrev, opp_abbrev (optional)
  - star_tier_pts (for star slices)
  - minutes_roll5 (for minutes buckets)
  - player_position / position / pos (for position slices, if present)
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze residuals from points regression validation predictions."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/points_regression_val_preds.csv",
        help="Path to CSV with validation predictions (default: data/points_regression_val_preds.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading validation preds from {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows.\n")

    # Basic sanity checks
    required = ["y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Ensure residual column exists
    if "residual" not in df.columns:
        df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_residual"] = df["residual"].abs()

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("=== Overall performance on this validation set ===")
    print(f"MAE:   {mae:7.3f}")
    print(f"RMSE:  {rmse:7.3f}")
    print(f"R^2:   {r2:7.3f}\n")

    # ------------------------------------------------------------------
    # NEW: Star-only global slice (star_tier_pts >= 2)
    # ------------------------------------------------------------------
    if "star_tier_pts" in df.columns:
        star_mask = df["star_tier_pts"] >= 2
        df_star = df[star_mask].copy()

        if not df_star.empty:
            y_true_star = df_star["y_true"].to_numpy()
            y_pred_star = df_star["y_pred"].to_numpy()

            mae_star = mean_absolute_error(y_true_star, y_pred_star)
            rmse_star = math.sqrt(mean_squared_error(y_true_star, y_pred_star))
            r2_star = r2_score(y_true_star, y_pred_star)

            print("=== Star-only performance (star_tier_pts >= 2) ===")
            print(f"Count: {len(df_star):,}")
            print(f"MAE:   {mae_star:7.3f}")
            print(f"RMSE:  {rmse_star:7.3f}")
            print(f"R^2:   {r2_star:7.3f}\n")
        else:
            print("=== Star-only performance ===")
            print("No rows with star_tier_pts >= 2 found.\n")
    else:
        print("[WARN] No 'star_tier_pts' column; skipping star-only global metrics.\n")

    # ------------------------------------------------------------------
    # Residuals by star_tier_pts
    # ------------------------------------------------------------------
    if "star_tier_pts" in df.columns:
        rows = []
        for tier, g in df.groupby("star_tier_pts"):
            if g.empty:
                continue
            y_t = g["y_true"].to_numpy()
            y_p = g["y_pred"].to_numpy()
            tier_mae = mean_absolute_error(y_t, y_p)
            tier_rmse = math.sqrt(mean_squared_error(y_t, y_p))
            rows.append(
                {
                    "star_tier_pts": int(tier),
                    "count": len(g),
                    "mae": tier_mae,
                    "rmse": tier_rmse,
                }
            )
        if rows:
            table = pd.DataFrame(rows).sort_values("star_tier_pts")
            print("=== Residuals by star_tier_pts ===")
            print(table.to_string(index=False))
            print()
    else:
        print("[WARN] No 'star_tier_pts' column; skipping star_tier slice.\n")

    # ------------------------------------------------------------------
    # Residuals by minutes_roll5 buckets
    # ------------------------------------------------------------------
    if "minutes_roll5" in df.columns:
        # Define buckets: 0–10, 10–20, 20–30, 30–40, 40–60
        bins = [0, 10, 20, 30, 40, 60]
        labels = ["0–10", "10–20", "20–30", "30–40", "40–60"]

        df["minutes_bucket"] = pd.cut(
            df["minutes_roll5"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        rows = []
        for bucket, g in df.groupby("minutes_bucket"):
            if g.empty or pd.isna(bucket):
                continue
            y_t = g["y_true"].to_numpy()
            y_p = g["y_pred"].to_numpy()
            b_mae = mean_absolute_error(y_t, y_p)
            b_rmse = math.sqrt(mean_squared_error(y_t, y_p))
            rows.append(
                {
                    "minutes_bucket": str(bucket),
                    "count": len(g),
                    "mae": b_mae,
                    "rmse": b_rmse,
                }
            )

        if rows:
            table = pd.DataFrame(rows)
            # Keep the bucket order as defined in labels
            table["minutes_bucket"] = pd.Categorical(table["minutes_bucket"], categories=labels, ordered=True)
            table = table.sort_values("minutes_bucket")
            print("=== Residuals by minutes_roll5 buckets (recent role) ===")
            print(table.to_string(index=False))
            print()
    else:
        print("[WARN] No 'minutes_roll5' column; skipping minutes bucket slice.\n")

    # ------------------------------------------------------------------
    # Residuals by position (if we have any position column)
    # ------------------------------------------------------------------
    pos_col = None
    for cand in ["player_position", "position", "pos"]:
        if cand in df.columns:
            pos_col = cand
            break

    if pos_col is not None:
        rows = []
        for pos, g in df.groupby(pos_col):
            if g.empty or pd.isna(pos):
                continue
            y_t = g["y_true"].to_numpy()
            y_p = g["y_pred"].to_numpy()
            p_mae = mean_absolute_error(y_t, y_p)
            p_rmse = math.sqrt(mean_squared_error(y_t, y_p))
            rows.append(
                {
                    pos_col: pos,
                    "count": len(g),
                    "mae": p_mae,
                    "rmse": p_rmse,
                }
            )

        if rows:
            table = pd.DataFrame(rows).sort_values("count", ascending=False)
            print(f"=== Residuals by {pos_col} ===")
            print(table.to_string(index=False))
            print()
    else:
        print("[WARN] No position column found; skipping position slice.\n")

    print("Done.")


if __name__ == "__main__":
    main()