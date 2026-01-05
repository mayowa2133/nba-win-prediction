#!/usr/bin/env python
"""
analyze_points_model_errors.py

Analyze where the points regression model is doing well / poorly.

- Loads:
    - data/player_points_features.csv
    - models/points_regression.pkl ({"model", "sigma", "feature_cols"})
- Filters to a season range (default: latest season only).
- Computes residuals and aggregates error metrics by:
    - Overall
    - Predicted points band (as a proxy for line bands)
    - Player
    - Team
    - Rest flags (is_b2b, is_long_rest)

Optional:
    --output-errors-csv to save per-row errors to a CSV for deeper analysis.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

FEATURES_CSV_DEFAULT = Path("data/player_points_features.csv")
MODEL_PATH_DEFAULT = Path("models/points_regression.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze errors of the player points regression model."
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV_DEFAULT),
        help="Path to player_points_features.csv "
             "(default: data/player_points_features.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH_DEFAULT),
        help="Path to saved regression model bundle "
             "(default: models/points_regression.pkl)",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=None,
        help="Minimum season (start year) to include in analysis. "
             "Default: latest season in the data.",
    )
    parser.add_argument(
        "--season-max",
        type=int,
        default=None,
        help="Maximum season (start year) to include in analysis. "
             "Default: same as season-min.",
    )
    parser.add_argument(
        "--output-errors-csv",
        type=str,
        default=None,
        help="Optional path to write per-row errors CSV "
             "(e.g. data/points_model_errors_2025.csv).",
    )
    return parser.parse_args()


def load_model_bundle(model_path: Path) -> Dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    if "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError(
            f"Model bundle at {model_path} does not contain 'model' and 'feature_cols'."
        )
    return bundle


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    bias = float(np.mean(y_true - y_pred))  # positive => model underpredicts on avg
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "bias": bias,
    }


def print_overall_metrics(df_err: pd.DataFrame) -> None:
    y_true = df_err["pts"].to_numpy()
    y_pred = df_err["y_pred"].to_numpy()
    metrics = compute_basic_metrics(y_true, y_pred)

    print("\n=== Overall metrics for selected seasons ===")
    print(f"Rows: {len(df_err):,}")
    print(f"MAE:   {metrics['mae']:6.3f}")
    print(f"RMSE:  {metrics['rmse']:6.3f}")
    print(f"R^2:   {metrics['r2']:6.3f}")
    print(f"Bias (y_true - y_pred): {metrics['bias']:6.3f}")


def add_predicted_band(df_err: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'pred_band' column bucketed by predicted points.
    This acts like a line band proxy (since lines ~ expected points).
    """
    bins = [0, 15, 20, 25, 30, 35, 40, 100]
    labels = [
        "<15",
        "15-19.5",
        "20-24.5",
        "25-29.5",
        "30-34.5",
        "35-39.5",
        "40+",
    ]
    df_err = df_err.copy()
    df_err["pred_band"] = pd.cut(
        df_err["y_pred"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return df_err


def print_group_metrics(
    df_err: pd.DataFrame,
    group_col: str,
    min_count: int = 100,
    top_n: int = 20,
    sort_by: str = "mae",
    desc: bool = True,
    title: Optional[str] = None,
) -> None:
    if group_col not in df_err.columns:
        print(f"\n[WARN] Column {group_col!r} not found; skipping group metrics.")
        return

    g = df_err.groupby(group_col)
    stats = g.agg(
        n=("pts", "size"),
        mae=("abs_resid", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        bias=("resid", "mean"),
    ).reset_index()

    stats = stats[stats["n"] >= min_count]
    if stats.empty:
        print(f"\n[INFO] No groups with at least {min_count} rows for {group_col}.")
        return

    stats = stats.sort_values(by=sort_by, ascending=not desc)

    if title is None:
        title = f"Metrics by {group_col}"
    print(f"\n=== {title} (min_count={min_count}, top_n={top_n}) ===")
    print(f"{group_col:20} | {'n':>6} | {'MAE':>6} | {'RMSE':>6} | {'Bias':>7}")
    print("-" * 60)
    for _, row in stats.head(top_n).iterrows():
        grp = str(row[group_col])
        print(
            f"{grp:20} | "
            f"{int(row['n']):6d} | "
            f"{row['mae']:6.3f} | "
            f"{row['rmse']:6.3f} | "
            f"{row['bias']:7.3f}"
        )


def main():
    args = parse_args()

    features_csv = Path(args.features_csv)
    model_path = Path(args.model_path)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")

    print(f"Loading features from {features_csv} ...")
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in features CSV.")
    if "pts" not in df.columns:
        raise ValueError("Expected a 'pts' (target) column in features CSV.")

    seasons = sorted(df["season"].unique())
    print("Seasons in dataset:", seasons)

    min_season_data = int(min(seasons))
    max_season_data = int(max(seasons))

    # Resolve season range for analysis
    if args.season_min is not None:
        season_min = args.season_min
    else:
        # Default: latest season only
        season_min = max_season_data

    if args.season_max is not None:
        season_max = args.season_max
    else:
        season_max = season_min

    # Clamp to data range
    season_min = max(season_min, min_season_data)
    season_max = min(season_max, max_season_data)

    if season_min > season_max:
        raise ValueError(
            f"Invalid season range [{season_min}, {season_max}] "
            f"given data seasons [{min_season_data}, {max_season_data}]"
        )

    print("\n=== Error analysis season configuration ===")
    print(f"Seasons: [{season_min}, {season_max}]")

    df = df[(df["season"] >= season_min) & (df["season"] <= season_max)].copy()
    print(f"Rows after season filter: {len(df):,}")
    if df.empty:
        raise RuntimeError("No rows left after season filtering; nothing to analyze.")

    # Load model bundle
    print(f"\nLoading regression model bundle from {model_path} ...")
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    print("Model expects feature columns:")
    for c in feature_cols:
        print(f"  - {c}")

    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Features CSV is missing expected feature columns: {missing_feats}"
        )

    # Build design matrix
    X = df[feature_cols].to_numpy()
    y_true = df["pts"].to_numpy()

    print("\nPredicting points for selected rows...")
    y_pred = model.predict(X)

    # Build error dataframe
    df_err = df.copy()
    df_err["y_pred"] = y_pred
    df_err["resid"] = df_err["pts"] - df_err["y_pred"]
    df_err["abs_resid"] = df_err["resid"].abs()
    df_err["squared_error"] = df_err["resid"] ** 2

    # Add predicted band for "line-like" buckets
    df_err = add_predicted_band(df_err)

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    print_overall_metrics(df_err)

    # ------------------------------------------------------------------
    # Metrics by predicted band (proxy for line bands)
    # ------------------------------------------------------------------
    print_group_metrics(
        df_err,
        group_col="pred_band",
        min_count=50,
        top_n=20,
        sort_by="mae",
        desc=True,
        title="Metrics by predicted points band (proxy for line band)",
    )

    # ------------------------------------------------------------------
    # Metrics by player (top offenders)
    # ------------------------------------------------------------------
    if "player_name" in df_err.columns:
        print_group_metrics(
            df_err,
            group_col="player_name",
            min_count=50,   # only look at players with decent sample size
            top_n=25,
            sort_by="mae",
            desc=True,
            title="Metrics by player (players with highest MAE)",
        )

    # ------------------------------------------------------------------
    # Metrics by team
    # ------------------------------------------------------------------
    if "team_abbrev" in df_err.columns:
        print_group_metrics(
            df_err,
            group_col="team_abbrev",
            min_count=100,
            top_n=30,
            sort_by="mae",
            desc=True,
            title="Metrics by team",
        )

    # ------------------------------------------------------------------
    # Metrics by rest flags
    # ------------------------------------------------------------------
    for flag_col in ["is_b2b", "is_long_rest"]:
        if flag_col in df_err.columns:
            print_group_metrics(
                df_err,
                group_col=flag_col,
                min_count=50,
                top_n=10,
                sort_by="mae",
                desc=True,
                title=f"Metrics by {flag_col}",
            )

    # ------------------------------------------------------------------
    # Optional: save per-row errors
    # ------------------------------------------------------------------
    if args.output_errors_csv:
        out_path = Path(args.output_errors_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_err.to_csv(out_path, index=False)
        print(f"\nSaved per-row error details to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()