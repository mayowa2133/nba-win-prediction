#!/usr/bin/env python
"""
compare_star_mae.py

Compare prediction accuracy for different star tiers between two models:
- A "baseline" model (no star weights, or old config)
- A "star-weighted" model (trained with --use-star-weights)

Both inputs should be CSVs exported by build_points_regression.py:
    data/points_regression_val_preds.csv

Expected columns:
    - y_true
    - y_pred
    - star_tier_pts

Usage example:

    python compare_star_mae.py \
        --baseline-csv data/points_regression_val_preds_baseline.csv \
        --weighted-csv data/points_regression_val_preds_starweights.csv
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MAE by star tier between baseline and star-weighted models."
    )
    parser.add_argument(
        "--baseline-csv",
        type=str,
        required=True,
        help="CSV of validation preds for the baseline model (no star weights).",
    )
    parser.add_argument(
        "--weighted-csv",
        type=str,
        required=True,
        help="CSV of validation preds for the star-weighted model.",
    )
    return parser.parse_args()


def load_preds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    required_cols = {"y_true", "y_pred", "star_tier_pts"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def mae_and_count(df: pd.DataFrame) -> Tuple[float, int]:
    """Return (MAE, count) for a DataFrame with y_true, y_pred."""
    if df.empty:
        return float("nan"), 0
    errors = (df["y_true"] - df["y_pred"]).abs().to_numpy()
    mae = float(errors.mean())
    return mae, len(df)


def print_section_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    args = parse_args()

    baseline_path = Path(args.baseline_csv)
    weighted_path = Path(args.weighted_csv)

    print(f"Loading baseline preds from: {baseline_path}")
    df_base = load_preds(baseline_path)

    print(f"Loading weighted preds from: {weighted_path}")
    df_weight = load_preds(weighted_path)

    # Sanity: make sure they cover the same games/rows
    # (Not strictly required, but nice to see.)
    print_section_header("Basic info")
    print(f"Baseline rows: {len(df_base):,}")
    print(f"Weighted rows: {len(df_weight):,}")

    # Overall MAE
    print_section_header("Overall MAE")
    mae_base, n_base = mae_and_count(df_base)
    mae_weight, n_weight = mae_and_count(df_weight)
    print(f"Baseline: MAE={mae_base:6.3f} over {n_base:,} rows")
    print(f"Weighted: MAE={mae_weight:6.3f} over {n_weight:,} rows")

    # MAE by star_tier_pts
    print_section_header("MAE by star_tier_pts")
    tiers = sorted(int(t) for t in np.unique(df_base["star_tier_pts"]))
    print(f"{'tier':>4} | {'baseline_MAE':>12} | {'weighted_MAE':>12} | {'n_rows':>8}")
    print("-" * 52)
    for tier in tiers:
        mask_base = df_base["star_tier_pts"] == tier
        mask_weight = df_weight["star_tier_pts"] == tier

        mae_b, n_b = mae_and_count(df_base[mask_base])
        mae_w, n_w = mae_and_count(df_weight[mask_weight])

        # we assume they have the same n_rows per tier, but report baseline count
        print(
            f"{tier:>4} | {mae_b:12.3f} | {mae_w:12.3f} | {n_b:8d}"
        )

    # MAE for "bettable" pool: tiers >= 2 (adjust if you want)
    print_section_header("MAE for 'bettable' stars (tiers >= 2)")
    mask_base_stars = df_base["star_tier_pts"] >= 2
    mask_weight_stars = df_weight["star_tier_pts"] >= 2

    mae_bet_base, n_bet_base = mae_and_count(df_base[mask_base_stars])
    mae_bet_weight, n_bet_weight = mae_and_count(df_weight[mask_weight_stars])

    print(f"Baseline (tiers >=2): MAE={mae_bet_base:6.3f} over {n_bet_base:,} rows")
    print(f"Weighted (tiers >=2): MAE={mae_bet_weight:6.3f} over {n_bet_weight:,} rows")

    print_section_header("Done")
    print("If weighted MAE is lower for tiers >=2, star weights helped where it matters.")


if __name__ == "__main__":
    main()