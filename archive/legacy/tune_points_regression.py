#!/usr/bin/env python
"""
tune_points_regression.py

Hyperparameter tuning script for the player points regression model.

- Uses the same FEATURE_COLS / TARGET_COL as build_points_regression.py
- Train: seasons <= train_max_season (default: 2023)
- Dev:   season == dev_season       (default: 2024)

Usage:
    python tune_points_regression.py
    python tune_points_regression.py --max-evals 30
"""

import argparse
import itertools
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import feature definitions from your existing script so we stay in sync
from build_points_regression import FEATURE_COLS  # type: ignore

# Must match the target column in player_points_features.csv
TARGET_COL = "pts"


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FEATURES_PATH = DATA_DIR / "player_points_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for HistGradientBoostingRegressor on player points."
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=str(FEATURES_PATH),
        help="Path to player_points_features.csv (default: data/player_points_features.csv)",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=2023,
        help="Max season to include in the training set (default: 2023).",
    )
    parser.add_argument(
        "--dev-season",
        type=int,
        default=2024,
        help="Season to use as a dev/validation set (default: 2024).",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=20,
        help="Maximum number of hyperparameter combinations to evaluate (default: 20).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffling hyperparameter combinations (default: 42).",
    )
    return parser.parse_args()


def load_data(features_file: str) -> pd.DataFrame:
    print(f"Loading features from {features_file} ...")
    df = pd.read_csv(features_file)

    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in player_points_features.csv")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected TARGET_COL='{TARGET_COL}' to be present in the features file")

    # Drop rows where target is NaN (keep NaNs in features, HistGB can handle them)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with NaN targets ({TARGET_COL}).")

    seasons = sorted(df["season"].unique())
    print(f"Loaded {len(df):,} rows with seasons: {seasons}")
    print(f"Number of feature columns used: {len(FEATURE_COLS)}")
    return df


def split_train_dev(
    df: pd.DataFrame, train_max_season: int, dev_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["season"] <= train_max_season
    dev_mask = df["season"] == dev_season

    df_train = df[train_mask].copy()
    df_dev = df[dev_mask].copy()

    print(f"Train seasons: <= {train_max_season}")
    print(f"Dev   seasons: == {dev_season}")
    print(f"Train rows: {len(df_train):,}")
    print(f"Dev   rows: {len(df_dev):,}")

    if len(df_dev) == 0:
        raise ValueError("Dev set has 0 rows. Check that dev_season exists in the data.")

    return df_train, df_dev


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values.astype(float)
    return X, y


def compute_baselines(y_train: np.ndarray, y_dev: np.ndarray, pts_roll5_dev: np.ndarray) -> None:
    """
    Compute same baselines as build_points_regression.py for reference:
        - Global mean of train target
        - pts_roll5 (last-5-games average)
    """
    print("\n=== Baseline metrics on dev set ===")

    # Global mean (use train-set average)
    global_mean = float(np.mean(y_train))
    mae_mean = mean_absolute_error(y_dev, np.full_like(y_dev, global_mean))
    rmse_mean = math.sqrt(mean_squared_error(y_dev, np.full_like(y_dev, global_mean)))
    r2_mean = r2_score(y_dev, np.full_like(y_dev, global_mean))
    print(f"MEAN   - MAE: {mae_mean:7.3f}  RMSE: {rmse_mean:7.3f}  R^2: {r2_mean:7.3f}")

    # pts_roll5 baseline (if available)
    mask_valid_roll5 = ~np.isnan(pts_roll5_dev)
    if mask_valid_roll5.any():
        mae_roll5 = mean_absolute_error(y_dev[mask_valid_roll5], pts_roll5_dev[mask_valid_roll5])
        rmse_roll5 = math.sqrt(
            mean_squared_error(y_dev[mask_valid_roll5], pts_roll5_dev[mask_valid_roll5])
        )
        r2_roll5 = r2_score(y_dev[mask_valid_roll5], pts_roll5_dev[mask_valid_roll5])
        print(f"ROLL5  - MAE: {mae_roll5:7.3f}  RMSE: {rmse_roll5:7.3f}  R^2: {r2_roll5:7.3f}")
    else:
        print("ROLL5  - skipped (no non-NaN pts_roll5 on dev set).")


def generate_param_combos(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    df = load_data(args.features_file)
    df_train, df_dev = split_train_dev(df, args.train_max_season, args.dev_season)

    X_train, y_train = get_X_y(df_train)
    X_dev, y_dev = get_X_y(df_dev)

    # Baseline comparison (MEAN and ROLL5)
    if "pts_roll5" in df_dev.columns:
        compute_baselines(y_train, y_dev, df_dev["pts_roll5"].to_numpy(dtype=float))
    else:
        print("\npts_roll5 column not found; skipping ROLL5 baseline metrics.")

    # ---------------------------------------------------------------------
    # Hyperparameter grid
    # ---------------------------------------------------------------------
    param_grid = {
        "max_iter": [200, 400, 600],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_leaf_nodes": [31, 63, 127],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0],
    }

    all_combos = generate_param_combos(param_grid)
    random.shuffle(all_combos)

    n_evals = min(args.max_evals, len(all_combos))
    combos_to_try = all_combos[:n_evals]

    print(f"\n=== Hyperparameter tuning ===")
    print(f"Total grid size: {len(all_combos)} combinations")
    print(f"Evaluating:      {n_evals} combinations (max_evals={args.max_evals})")

    results = []

    for i, params in enumerate(combos_to_try, start=1):
        print(f"\n[{i}/{n_evals}] Training HistGradientBoostingRegressor with params:")
        for k, v in params.items():
            print(f"  - {k} = {v}")

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=args.random_seed,
        )

        model.fit(X_train, y_train)
        y_pred_dev = model.predict(X_dev)

        mae = mean_absolute_error(y_dev, y_pred_dev)
        rmse = math.sqrt(mean_squared_error(y_dev, y_pred_dev))
        r2 = r2_score(y_dev, y_pred_dev)

        print(f"DEV METRICS -> MAE: {mae:7.3f}  RMSE: {rmse:7.3f}  R^2: {r2:7.3f}")

        results.append(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                **params,
            }
        )

    if not results:
        print("No hyperparameter combinations were evaluated. Check your grid or max_evals.")
        return

    # Sort by MAE ascending (primary), then RMSE ascending
    results_sorted = sorted(results, key=lambda r: (r["mae"], r["rmse"]))

    print("\n=== Top hyperparameter configs (sorted by MAE) ===")
    max_to_show = min(10, len(results_sorted))
    for i, r in enumerate(results_sorted[:max_to_show], start=1):
        print(
            f"[{i}] MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  R^2={r['r2']:.3f} | "
            f"max_iter={r['max_iter']}, lr={r['learning_rate']}, "
            f"max_leaf_nodes={r['max_leaf_nodes']}, "
            f"min_samples_leaf={r['min_samples_leaf']}, "
            f"l2={r['l2_regularization']}"
        )

    best = results_sorted[0]
    print("\n=== Best config (to paste into build_points_regression.py) ===")
    print("HISTGB_PARAMS = {")
    print(f"    'max_iter': {best['max_iter']},")
    print(f"    'learning_rate': {best['learning_rate']},")
    print(f"    'max_leaf_nodes': {best['max_leaf_nodes']},")
    print(f"    'min_samples_leaf': {best['min_samples_leaf']},")
    print(f"    'l2_regularization': {best['l2_regularization']},")
    print("}")
    print(
        f"# Dev (season={args.dev_season}) metrics with this config: "
        f"MAE={best['mae']:.3f}, RMSE={best['rmse']:.3f}, R^2={best['r2']:.3f}"
    )


if __name__ == "__main__":
    main()