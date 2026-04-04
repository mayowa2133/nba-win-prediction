#!/usr/bin/env python
"""
build_points_sigma_model.py

Train a heteroscedastic sigma model for player points:

- Loads:
    - data/player_points_features.csv
    - models/points_regression.pkl
- Uses the main regression model to get mu_hat (expected points).
- Computes |residual| as the target for a second regressor (i.e., sigma, not variance).
- Optionally log-transforms the target for stability.
- Trains a HistGradientBoostingRegressor to predict per-row sigma.
- Saves to models/points_sigma_model.pkl:
    {
        "model": sigma_regressor,
        "feature_cols": sigma_feature_cols,
        "config": {
            "use_log_target": bool,
            "eps": float,          # small constant for numerical stability
            "sigma_scale": float,  # multiplicative factor applied to sigma at inference
        },
    }
"""

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.utils.artifact_metadata import (
    build_model_bundle_metadata,
    build_training_window,
)

FEATURES_CSV = Path("data/player_points_features.csv")
MAIN_MODEL_PATH = Path("models/points_regression.pkl")
SIGMA_MODEL_PATH = Path("models/points_sigma_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a heteroscedastic sigma model (per-row std dev) for player points."
        )
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV),
        help="Path to player_points_features.csv (default: data/player_points_features.csv)",
    )
    parser.add_argument(
        "--main-model-path",
        type=str,
        default=str(MAIN_MODEL_PATH),
        help="Path to main points regression model bundle "
             "(default: models/points_regression.pkl)",
    )
    parser.add_argument(
        "--sigma-model-path",
        type=str,
        default=str(SIGMA_MODEL_PATH),
        help="Where to save the sigma model bundle "
             "(default: models/points_sigma_model.pkl)",
    )
    parser.add_argument(
        "--train-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include for sigma training. "
             "Default: min season in the data.",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include for sigma training. "
             "Default: second-most-recent season in the data (exclude latest).",
    )
    parser.add_argument(
        "--use-log-target",
        action="store_true",
        help="If set, train on log(|residual| + eps) instead of |residual| directly.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Small constant for numerical stability in sigma / log transforms.",
    )
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="Global multiplicative factor applied to predicted sigma at inference time.",
    )
    parser.add_argument(
        "--add-derived-sigma-features",
        action="store_true",
        help="If set, add derived sigma features (mu_hat transforms/interactions).",
    )
    return parser.parse_args()


def load_main_model(main_model_path: Path) -> Dict:
    if not main_model_path.exists():
        raise FileNotFoundError(f"Main model bundle not found: {main_model_path}")
    with open(main_model_path, "rb") as f:
        bundle = pickle.load(f)
    if "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError(
            f"Main model bundle at {main_model_path} must contain 'model' and 'feature_cols'."
        )
    return bundle


def make_sigma_model() -> HistGradientBoostingRegressor:
    # We don't need super crazy tuning here; this is a first-pass.
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=0.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


def main():
    args = parse_args()

    features_csv = Path(args.features_csv)
    main_model_path = Path(args.main_model_path)
    sigma_model_path = Path(args.sigma_model_path)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")

    print(f"Loading features from {features_csv} ...")
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in features CSV.")

    # --- NEW: gracefully handle target_pts vs pts -----------------------
    if "pts" not in df.columns:
        if "target_pts" in df.columns:
            print("[INFO] 'pts' column not found; using 'target_pts' as alias for sigma training.")
            df["pts"] = df["target_pts"]
        else:
            raise ValueError("Expected a 'pts' (target) column or 'target_pts' in features CSV.")
    # --------------------------------------------------------------------

    seasons = sorted(df["season"].unique())
    print("Seasons in dataset:", seasons)
    if not seasons:
        raise ValueError("No seasons found in data.")

    min_season_data = int(min(seasons))
    max_season_data = int(max(seasons))

    # Default: train on all but the latest season (so sigma model doesn't peek at holdout by default)
    if args.train_min_season is not None:
        train_min = args.train_min_season
    else:
        train_min = min_season_data

    if args.train_max_season is not None:
        train_max = args.train_max_season
    else:
        if min_season_data == max_season_data:
            train_max = max_season_data
        else:
            # second-most-recent season
            sorted_seasons = sorted(seasons)
            train_max = int(sorted_seasons[-2])

    # Clamp to data range
    train_min = max(train_min, min_season_data)
    train_max = min(train_max, max_season_data)

    if train_min > train_max:
        raise ValueError(
            f"Invalid train season range: [{train_min}, {train_max}] "
            f"given data seasons [{min_season_data}, {max_season_data}]"
        )

    print("\n=== Sigma model training season configuration ===")
    print(f"Train seasons: [{train_min}, {train_max}]")

    df_train = df[(df["season"] >= train_min) & (df["season"] <= train_max)].copy()
    print(f"Rows for sigma training: {len(df_train):,}")
    if df_train.empty:
        raise RuntimeError("No rows for sigma model training after season filtering.")

    # Load main model + features
    print(f"\nLoading main regression model from {main_model_path} ...")
    bundle = load_main_model(main_model_path)
    main_model = bundle["model"]
    feature_cols: List[str] = bundle["feature_cols"]

    print("Main model feature columns:")
    for c in feature_cols:
        print(f"  - {c}")

    # Handle missing Vegas features gracefully (for backward compatibility)
    vegas_cols = ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]
    for col in vegas_cols:
        if col in feature_cols and col not in df_train.columns:
            print(f"[INFO] Vegas column '{col}' not in features; filling with 0.0")
            df_train[col] = 0.0

    # Handle missing prop features gracefully (for historical data without props)
    prop_cols = [
        "prop_pts_line",
        "prop_over_odds_best",
        "prop_under_odds_best",
        "has_prop_line",
        "prop_minus_pts_roll5",
        "prop_minus_pts_roll15",
        "prop_minus_season_mean",
        "prop_minus_career_mean",
        "prop_minus_model_baseline",
    ]
    for col in prop_cols:
        if col in feature_cols and col not in df_train.columns:
            if col == "has_prop_line":
                print(f"[INFO] Prop column '{col}' not in features; filling with 0.0 (no prop)")
                df_train[col] = 0.0
            else:
                print(f"[INFO] Prop column '{col}' not in features; filling with 0.0")
                df_train[col] = 0.0

    # Handle missing injury features gracefully (for backward compatibility)
    injury_cols = ["is_injured", "days_since_last_dnp", "dnp_count_last_10"]
    for col in injury_cols:
        if col in feature_cols and col not in df_train.columns:
            if col == "is_injured":
                print(f"[INFO] Injury column '{col}' not in features; filling with 0 (healthy)")
                df_train[col] = 0
            elif col == "days_since_last_dnp":
                print(f"[INFO] Injury column '{col}' not in features; filling with 999 (never injured)")
                df_train[col] = 999
            else:
                print(f"[INFO] Injury column '{col}' not in features; filling with 0")
                df_train[col] = 0

    # Handle minutes_pred feature (computed dynamically from minutes model)
    if "minutes_pred" in feature_cols and "minutes_pred" not in df_train.columns:
        print("[INFO] Computing 'minutes_pred' feature using minutes model...")
        minutes_model_path = Path("models/minutes_regression.pkl")
        if minutes_model_path.exists():
            try:
                with open(minutes_model_path, "rb") as f:
                    minutes_bundle = pickle.load(f)
                minutes_model = minutes_bundle.get("model")
                minutes_feature_cols = minutes_bundle.get("feature_cols")
                
                if minutes_model is not None and minutes_feature_cols is not None:
                    missing_min = [c for c in minutes_feature_cols if c not in df_train.columns]
                    if not missing_min:
                        X_min = df_train[minutes_feature_cols].to_numpy()
                        df_train["minutes_pred"] = minutes_model.predict(X_min)
                        print("[INFO] Successfully computed 'minutes_pred' feature.")
                    else:
                        print(f"[WARN] Missing minutes feature columns: {missing_min}; filling minutes_pred with 0.0")
                        df_train["minutes_pred"] = 0.0
                else:
                    print("[WARN] Minutes model bundle invalid; filling minutes_pred with 0.0")
                    df_train["minutes_pred"] = 0.0
            except Exception as e:
                print(f"[WARN] Failed to load minutes model: {e}; filling minutes_pred with 0.0")
                df_train["minutes_pred"] = 0.0
        else:
            print(f"[WARN] Minutes model not found at {minutes_model_path}; filling minutes_pred with 0.0")
            df_train["minutes_pred"] = 0.0

    missing_feats = [c for c in feature_cols if c not in df_train.columns]
    if missing_feats:
        raise ValueError(
            f"Training data is missing expected feature columns: {missing_feats}"
        )

    X_train_main = df_train[feature_cols].to_numpy()
    y_true = df_train["pts"].to_numpy()

    print("\nComputing main model predictions (mu_hat) for sigma training rows...")
    mu_hat = main_model.predict(X_train_main)

    residuals = y_true - mu_hat
    abs_resid = np.abs(residuals)

    print(f"Residual stats on train seasons [{train_min}, {train_max}]:")
    print(f"  mean |resid|: {np.mean(abs_resid):.3f}")
    print(f"  RMSE (sqrt(mean sq_error)): {math.sqrt(mean_squared_error(y_true, mu_hat)):.3f}")

    # Build sigma training design matrix: reuse main features + mu_hat as an extra feature
    df_sigma = df_train[feature_cols].copy()
    df_sigma["mu_hat"] = mu_hat

    sigma_feature_cols = feature_cols + ["mu_hat"]
    if args.add_derived_sigma_features:
        from src.utils.sigma_features import SIGMA_DERIVED_COLS, add_sigma_derived_features_df

        add_sigma_derived_features_df(df_sigma)
        sigma_feature_cols = sigma_feature_cols + [c for c in SIGMA_DERIVED_COLS if c in df_sigma.columns]
    X_sigma = df_sigma[sigma_feature_cols].to_numpy()

    eps = float(args.eps)
    if args.use_log_target:
        y_sigma = np.log(abs_resid + eps)
        target_desc = "log(|residual| + eps)"
    else:
        y_sigma = abs_resid
        target_desc = "|residual| (sigma)"

    print(f"\nTraining sigma model to predict {target_desc} ...")
    sigma_model = make_sigma_model()
    sigma_model.fit(X_sigma, y_sigma)

    # ------------------------------------------------------------------
    # Quick sanity check: how well does sigma model match actual error?
    # ------------------------------------------------------------------
    print("\nEvaluating sigma model fit on training data (sanity check)...")
    y_sigma_pred = sigma_model.predict(X_sigma)

    sigma_scale = float(args.sigma_scale)
    if args.use_log_target:
        sigma_pred = np.exp(y_sigma_pred)
    else:
        sigma_pred = y_sigma_pred

    sigma_hat = np.maximum(sigma_pred * sigma_scale, eps)

    # Compare by band of mu_hat (like we did for error analysis)
    df_eval = df_train.copy()
    df_eval["mu_hat"] = mu_hat
    df_eval["sigma_hat"] = sigma_hat
    df_eval["resid"] = residuals

    bins = [0, 15, 20, 25, 30, 35, 40, 100]
    labels = ["<15", "15-19.5", "20-24.5", "25-29.5", "30-34.5", "35-39.5", "40+"]

    df_eval["mu_band"] = pd.cut(
        df_eval["mu_hat"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    g = df_eval.groupby("mu_band")
    print("\n=== Sigma model sanity check by mu_hat band ===")
    print("mu_band             |    n |  RMSE_emp | sigma_hat_mean")
    print("--------------------------------------------------------")
    for band, grp in g:
        if grp.empty:
            continue
        n = len(grp)
        rmse_emp = math.sqrt(np.mean(grp["resid"] ** 2))
        sigma_mean = float(np.mean(grp["sigma_hat"]))
        print(f"{str(band):20} | {n:5d} |  {rmse_emp:8.3f} |       {sigma_mean:8.3f}")

    # ------------------------------------------------------------------
    # Save sigma model bundle
    # ------------------------------------------------------------------
    sigma_bundle = {
        "model": sigma_model,
        "feature_cols": sigma_feature_cols,
        "config": {
            "use_log_target": bool(args.use_log_target),
            "eps": eps,
            "sigma_scale": sigma_scale,
        },
        **build_model_bundle_metadata(
            target="target_pts_abs_residual",
            training_window=build_training_window(
                train_min=train_min,
                train_max=train_max,
                val_min=None,
                val_max=None,
            ),
            readiness_status="experimental",
            model_type="histgb",
            extra={"uses_derived_sigma_features": bool(args.add_derived_sigma_features)},
        ),
    }

    sigma_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sigma_model_path, "wb") as f:
        pickle.dump(sigma_bundle, f)

    print(f"\nSaved sigma model bundle to {sigma_model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
