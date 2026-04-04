#!/usr/bin/env python
"""
build_minutes_regression.py

Train a regression model to predict player minutes for a game, using the same
engineered features used for points modeling.

Inputs:
  - data/player_points_features.csv

Outputs:
  - models/minutes_regression.pkl: {
        "model": regressor (HistGradientBoostingRegressor or XGBRegressor),
        "feature_cols": [list of feature names used for X]
    }

This model is intended to be used as an upstream "minutes predictor", whose
output (min_pred) is then fed into the points regression model via
minutes_utils.add_minutes_predictions(df).
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.artifact_metadata import (
    build_model_bundle_metadata,
    build_training_window,
)

# Optional: XGBoost support
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

FEATURES_CSV = Path("data") / "player_points_features.csv"
MODEL_PATH = Path("models") / "minutes_regression.pkl"

# NEW: target column name in features CSV
TARGET_MIN_COL = "target_min"

# Focused feature set for minutes prediction
MINUTES_FEATURE_COLS: List[str] = [
    # recent minutes / role
    "minutes_roll5",
    "minutes_roll15",
    "player_minutes_career_mean",
    "player_minutes_season_mean",
    "rel_minutes_vs_career",
    "star_tier_minutes",

    # scoring/usage proxies (players who score/shoot more often get more run)
    "pts_roll5",
    "pts_roll15",
    "fga_roll5",
    "fga_roll15",

    # direct usage volume
    "usg_events_roll5",
    "usg_events_roll15",

    # environment / context
    "team_pace_roll5",
    "team_pace_roll15",
    "team_margin_roll5",
    "team_margin_roll15",
    "days_since_last_game",
    "is_b2b",
    "is_long_rest",
    "is_home",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regression model to predict player minutes per game."
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV),
        help="Path to player_points_features.csv (default: data/player_points_features.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Where to save the minutes model bundle (default: models/minutes_regression.pkl)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="histgb",
        choices=["histgb", "xgboost"],
        help="Regressor type: 'histgb' (default) or 'xgboost'.",
    )
    parser.add_argument(
        "--train-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include in training. Default: min season in data.",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include in training. "
             "Default: second-most-recent season in data.",
    )
    parser.add_argument(
        "--val-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include in validation. "
             "Default: train_max_season + 1, if available.",
    )
    parser.add_argument(
        "--val-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include in validation. "
             "Default: max season in data.",
    )
    return parser.parse_args()


def make_histgb_model(params: Dict) -> HistGradientBoostingRegressor:
    """
    Basic HistGradientBoostingRegressor config for minutes.
    Defaults roughly aligned with what worked for points.
    """
    return HistGradientBoostingRegressor(
        max_iter=params.get("max_iter", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("max_leaf_nodes", 63),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        l2_regularization=params.get("l2_regularization", 0.1),
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


def make_xgb_model(params: Dict):
    """Create an XGBRegressor with given params for minutes."""
    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install it with 'pip install xgboost' "
            "or use --model-type histgb."
        )

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        random_state=42,
        n_jobs=-1,
    )


def build_model(model_type: str) -> object:
    """Factory for minutes model; uses fixed reasonable defaults."""
    if model_type == "histgb":
        params = {
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
        }
        return make_histgb_model(params)
    elif model_type == "xgboost":
        params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        }
        return make_xgb_model(params)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model_type: {model_type!r}")


def main() -> None:
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
    if TARGET_MIN_COL not in df.columns:
        raise ValueError(f"Expected a '{TARGET_MIN_COL}' target column in features CSV.")

    seasons = sorted(df["season"].unique())
    if not seasons:
        raise ValueError("No seasons found in data.")

    min_season = int(min(seasons))
    max_season = int(max(seasons))
    print("Seasons in dataset:", seasons)

    # ------------------------------------------------------------------
    # Resolve train/val season ranges
    # ------------------------------------------------------------------
    train_min = args.train_min_season if args.train_min_season is not None else min_season

    if args.train_max_season is not None:
        train_max = args.train_max_season
    else:
        if min_season == max_season:
            train_max = max_season
        else:
            sorted_seasons = sorted(seasons)
            train_max = int(sorted_seasons[-2])

    if args.val_min_season is not None:
        val_min = args.val_min_season
    else:
        val_min = train_max + 1 if train_max < max_season else train_max

    val_max = args.val_max_season if args.val_max_season is not None else max_season

    # Clamp
    train_min = max(train_min, min_season)
    train_max = min(train_max, max_season)
    val_min = max(val_min, min_season)
    val_max = min(val_max, max_season)

    if train_min > train_max:
        raise ValueError(
            f"Invalid train season range: [{train_min}, {train_max}] "
            f"given data seasons [{min_season}, {max_season}]"
        )
    if val_min > val_max:
        raise ValueError(
            f"Invalid val season range: [{val_min}, {val_max}] "
            f"given data seasons [{min_season}, {max_season}]"
        )

    print("\n=== Season split configuration (minutes model) ===")
    print(f"Train seasons: [{train_min}, {train_max}]")
    print(f"Val   seasons: [{val_min}, {val_max}]")

    train_mask = (df["season"] >= train_min) & (df["season"] <= train_max)
    val_mask = (df["season"] >= val_min) & (df["season"] <= val_max)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    print(f"\nTrain rows: {len(df_train):,}")
    print(f"Val   rows: {len(df_val):,}")

    if df_train.empty:
        raise RuntimeError("Training set is empty with the chosen season range.")

    # Make sure all minutes features exist
    missing_feats = [c for c in MINUTES_FEATURE_COLS if c not in df_train.columns]
    if missing_feats:
        raise ValueError(
            f"Training data is missing expected minutes feature columns: {missing_feats}"
        )

    # Use target_min as the label
    X_train = df_train[MINUTES_FEATURE_COLS].to_numpy()
    y_train = df_train[TARGET_MIN_COL].to_numpy()

    if df_val.empty:
        X_val = None
        y_val = None
        print(
            "WARNING: Validation set is empty with the chosen season range. "
            "Model will still train, but evaluation metrics will be skipped."
        )
    else:
        X_val = df_val[MINUTES_FEATURE_COLS].to_numpy()
        y_val = df_val[TARGET_MIN_COL].to_numpy()

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    model = build_model(args.model_type)
    print(f"\nTraining minutes model (type={args.model_type}) ...")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate on holdout seasons
    # ------------------------------------------------------------------
    if X_val is not None and y_val is not None and len(df_val) > 0:
        print(
            f"\nEvaluating minutes model on holdout seasons "
            f"[{val_min}, {val_max}] ..."
        )
        y_pred_val = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred_val)
        rmse = mean_squared_error(y_val, y_pred_val) ** 0.5
        r2 = r2_score(y_val, y_pred_val)

        print(f"MINUTES MODEL - MAE:  {mae:6.3f} (minutes)")
        print(f"MINUTES MODEL - RMSE: {rmse:6.3f}")
        print(f"MINUTES MODEL - R^2:  {r2:6.3f}")

        # Optional: dump val preds for debugging
        df_val_export = df_val.copy()
        df_val_export["minutes_true"] = y_val
        df_val_export["minutes_pred"] = y_pred_val
        df_val_export["minutes_resid"] = (
            df_val_export["minutes_true"] - df_val_export["minutes_pred"]
        )

        cols_front = [
            c
            for c in [
                "season",
                "game_date",
                "game_id",
                "player_id",
                "player_name",
                "team_abbrev",
                "opp_abbrev",
            ]
            if c in df_val_export.columns
        ]
        other_cols = [c for c in df_val_export.columns if c not in cols_front]
        df_val_export = df_val_export[cols_front + other_cols]

        out_path = Path("data/minutes_regression_val_preds.csv")
        df_val_export.to_csv(out_path, index=False)
        print(f"[DEBUG] Saved minutes validation preds to {out_path}")
    else:
        print("\nNo validation set; skipping eval for minutes model.")

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    bundle = {
        "model": model,
        "feature_cols": MINUTES_FEATURE_COLS,
        **build_model_bundle_metadata(
            target=TARGET_MIN_COL,
            training_window=build_training_window(
                train_min=train_min,
                train_max=train_max,
                val_min=val_min,
                val_max=val_max,
            ),
            readiness_status="experimental",
            model_type=args.model_type,
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        import pickle

        pickle.dump(bundle, f)

    print(f"\nSaved minutes regression model bundle to {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
