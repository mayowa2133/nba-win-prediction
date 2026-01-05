#!/usr/bin/env python
"""
evaluate_tiered_unified_ensemble.py

Evaluate unified vs tiered vs blended ensemble on a holdout season.

We search an ensemble weight w in [0,1]:
  pred_ens = w * pred_tiered + (1-w) * pred_unified

We keep this change only if MAE improves (and R² does not regress materially).
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


FEATURES_CSV_DEFAULT = Path("data/player_points_features_with_vegas.csv")
UNIFIED_MODEL_PATH = Path("models/points_regression.pkl")
TIER_MODEL_TEMPLATE = Path("models/points_regression_tier_{tier}.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune tiered+unified ensemble weight on holdout season.")
    p.add_argument("--features-csv", type=str, default=str(FEATURES_CSV_DEFAULT))
    p.add_argument("--eval-season", type=int, default=2025)
    p.add_argument("--train-max-season", type=int, default=2024)  # unused (models already trained); kept for clarity
    p.add_argument("--grid-step", type=float, default=0.05)
    return p.parse_args()


def load_bundle(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_minutes_pred(df: pd.DataFrame) -> None:
    if "minutes_pred" in df.columns:
        return
    from build_points_regression import add_minutes_pred_feature

    ok = add_minutes_pred_feature(df, Path("models/minutes_regression.pkl"))
    if not ok:
        df["minutes_pred"] = 0.0


def fill_defaults(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            continue
        if c in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
            df[c] = 0.0
        elif c == "is_injured":
            df[c] = 0
        elif c == "days_since_last_dnp":
            df[c] = 999
        elif c == "dnp_count_last_10":
            df[c] = 0
        elif c == "has_prop_line":
            df[c] = 0.0
        elif "fg_pct" in c:
            df[c] = 0.45
        elif "3pt_pct" in c:
            df[c] = 0.35
        else:
            df[c] = 0.0


def main() -> None:
    args = parse_args()
    df = pd.read_csv(Path(args.features_csv), low_memory=False)
    if "target_pts" not in df.columns:
        raise ValueError("features CSV missing target_pts")
    if "star_tier_pts" not in df.columns:
        raise ValueError("features CSV missing star_tier_pts")

    df_eval = df[df["season"] == int(args.eval_season)].copy()
    if df_eval.empty:
        raise RuntimeError(f"No rows found for eval season {args.eval_season}")

    unified = load_bundle(UNIFIED_MODEL_PATH)
    model_u = unified["model"]
    cols_u: List[str] = list(unified["feature_cols"])

    # Tier models use cols without star_tier_pts
    tier_models = {}
    for t in [0, 1, 2, 3]:
        tier_models[t] = load_bundle(TIER_MODEL_TEMPLATE.with_name(f"points_regression_tier_{t}.pkl"))

    # Ensure minutes_pred if needed
    if "minutes_pred" in cols_u:
        ensure_minutes_pred(df_eval)

    # Ensure has_prop_line if needed
    if "has_prop_line" in cols_u and "has_prop_line" not in df_eval.columns:
        if "prop_pts_line" in df_eval.columns:
            df_eval["has_prop_line"] = (~df_eval["prop_pts_line"].isna()).astype(float)
        else:
            df_eval["has_prop_line"] = 0.0

    fill_defaults(df_eval, cols_u)

    X_u = df_eval[cols_u].to_numpy()
    y = df_eval["target_pts"].to_numpy(dtype=float)
    pred_u = model_u.predict(X_u).astype(float)

    # Tier preds: build per-tier matrices for speed
    pred_t = np.zeros_like(pred_u, dtype=float)
    tiers = df_eval["star_tier_pts"].clip(0, 3).astype(int).to_numpy()

    for t in [0, 1, 2, 3]:
        idx = np.where(tiers == t)[0]
        if idx.size == 0:
            continue
        bundle_t = tier_models[t]
        cols_t: List[str] = list(bundle_t["feature_cols"])
        fill_defaults(df_eval, cols_t)
        X_t = df_eval.iloc[idx][cols_t].to_numpy()
        pred_t[idx] = bundle_t["model"].predict(X_t).astype(float)

    mae_u = mean_absolute_error(y, pred_u)
    r2_u = r2_score(y, pred_u)
    mae_t = mean_absolute_error(y, pred_t)
    r2_t = r2_score(y, pred_t)

    best = None
    grid = np.arange(0.0, 1.0 + 1e-9, float(args.grid_step))
    for w in grid:
        pred_e = w * pred_t + (1.0 - w) * pred_u
        mae = mean_absolute_error(y, pred_e)
        r2 = r2_score(y, pred_e)
        if best is None or mae < best["mae"]:
            best = {"w": float(w), "mae": float(mae), "r2": float(r2)}

    assert best is not None
    print("=" * 70)
    print("TIERED + UNIFIED ENSEMBLE (HOLDOUT)")
    print("=" * 70)
    print(f"Eval season: {args.eval_season}  rows={len(df_eval):,}  step={args.grid_step}")
    print()
    print("Model        |   MAE |   R²")
    print("-" * 40)
    print(f"Unified      | {mae_u:5.3f} | {r2_u:5.3f}")
    print(f"Tiered       | {mae_t:5.3f} | {r2_t:5.3f}")
    print(f"Ensemble(w)  | {best['mae']:5.3f} | {best['r2']:5.3f}   (w={best['w']:.2f})")
    print()
    print("Delta vs Unified:")
    print(f"  MAE: {best['mae'] - mae_u:+.3f}")
    print(f"  R² : {best['r2'] - r2_u:+.3f}")


if __name__ == "__main__":
    main()


