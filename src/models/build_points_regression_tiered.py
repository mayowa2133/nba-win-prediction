#!/usr/bin/env python
"""
build_points_regression_tiered.py

Train separate regression models for each player tier (0, 1, 2, 3) based on star_tier_pts.
This allows each tier to have its own specialized model, improving accuracy.

Tier definitions (from build_player_points_features.py):
  - Tier 0: < 8 ppg (low-usage/bench)
  - Tier 1: 8-15 ppg (rotation scorer)
  - Tier 2: 15-22 ppg (primary/secondary option)
  - Tier 3: 22+ ppg (star/elite scorer)

Outputs:
  - models/points_regression_tier_0.pkl
  - models/points_regression_tier_1.pkl
  - models/points_regression_tier_2.pkl
  - models/points_regression_tier_3.pkl

Each bundle contains:
  {
    "model": regressor,
    "sigma": float,
    "feature_cols": [list],
    "tier": int,
  }
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.artifact_metadata import (
    build_model_bundle_metadata,
    build_training_window,
)

# Try to import xgboost lazily
try:
    import xgboost as xgb
except ImportError:
    xgb = None

FEATURES_CSV = Path("data/player_points_features_with_vegas.csv")
MODELS_DIR = Path("models")
TARGET_COL = "target_pts"
UNIFIED_MODEL_PATH = Path("models/points_regression.pkl")

# Legacy base features (kept as fallback).
FALLBACK_FEATURE_COLS = [
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
    "usg_events_roll5",
    "usg_events_roll15",
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
    "team_margin_roll5",
    "team_margin_roll15",
    "days_since_last_game",
    "is_home",
    "opp_dvp_pos_pts_roll5",
    "opp_dvp_pos_pts_roll15",
    "team_pace_roll5",
    "team_pace_roll15",
    "player_pts_career_mean",
    "player_pts_season_mean",
    "player_minutes_career_mean",
    "player_minutes_season_mean",
    "rel_minutes_vs_career",
    "rel_pts_vs_career",
    "star_tier_minutes",  # Keep this as it's different from star_tier_pts
    "pts_trend_5_15",
    "minutes_trend_5_15",
    "fga_trend_5_15",
    "pts_std5",
    "minutes_std5",
    "fga_std5",
    "pts_per_min_roll5",
    "fga_per_min_roll5",
    "fta_per_min_roll5",
    "is_b2b",
    "is_long_rest",
    "vegas_game_total",
    "vegas_spread",
    "vegas_abs_spread",
    "is_injured",
    "days_since_last_dnp",
    "dnp_count_last_10",
    "minutes_pred",
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


def load_unified_feature_cols() -> Optional[List[str]]:
    """
    Prefer using the unified model's feature list so tiered models stay in sync
    with Phase 4A/4B additions (and any future additions).
    """
    if not UNIFIED_MODEL_PATH.exists():
        return None
    try:
        with open(UNIFIED_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        cols = bundle.get("feature_cols")
        if not isinstance(cols, list) or not cols:
            return None
        return cols
    except Exception:
        return None


def fill_missing_feature_defaults(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Fill missing feature columns with safe defaults (backwards compatible).
    Mirrors the behavior in build_points_regression.py.
    """
    for col in feature_cols:
        if col in df.columns:
            continue
        if col in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
            df[col] = 0.0
        elif col == "is_injured":
            df[col] = 0
        elif col == "days_since_last_dnp":
            df[col] = 999
        elif col == "dnp_count_last_10":
            df[col] = 0
        elif col == "has_prop_line":
            df[col] = 0.0
        elif "fg_pct" in col:
            df[col] = 0.45
        elif "3pt_pct" in col:
            df[col] = 0.35
        else:
            df[col] = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate regression models for each player tier (0, 1, 2, 3)."
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV),
        help="Path to features CSV (default: data/player_points_features_with_vegas.csv)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["histgb", "xgboost"],
        help="Model type (default: xgboost)",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=2024,
        help="Max season for training (default: 2024)",
    )
    parser.add_argument(
        "--val-min-season",
        type=int,
        default=2025,
        help="Min season for validation (default: 2025)",
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Enable hyperparameter tuning",
    )
    parser.add_argument(
        "--n-tune-iter",
        type=int,
        default=30,
        help="Number of tuning iterations (default: 30)",
    )
    return parser.parse_args()


def add_minutes_pred_feature(df: pd.DataFrame, minutes_model_path: Path) -> bool:
    """Add minutes_pred feature using minutes model."""
    if not minutes_model_path.exists():
        return False
    try:
        with open(minutes_model_path, "rb") as f:
            minutes_bundle = pickle.load(f)
        minutes_model = minutes_bundle.get("model")
        minutes_feature_cols = minutes_bundle.get("feature_cols")
        if minutes_model is None or minutes_feature_cols is None:
            return False
        missing = [c for c in minutes_feature_cols if c not in df.columns]
        if missing:
            return False
        X_min = df[minutes_feature_cols].to_numpy()
        df["minutes_pred"] = minutes_model.predict(X_min)
        return True
    except Exception:
        return False


def make_model(model_type: str, params: Optional[Dict] = None) -> object:
    """Create a model instance."""
    if params is None:
        params = {}
    
    if model_type == "xgboost":
        if xgb is None:
            raise ImportError("xgboost not installed")
        return xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
    else:
        return HistGradientBoostingRegressor(
            random_state=42,
            **params
        )


def train_tier_model(
    df_tier: pd.DataFrame,
    tier: int,
    model_type: str,
    feature_cols: List[str],
    train_max_season: int,
    val_min_season: int,
    tune: bool,
    n_tune_iter: int,
) -> tuple[object, float, Dict]:
    """Train a model for a specific tier."""
    print(f"\n{'='*60}")
    print(f"Training Tier {tier} Model")
    print(f"{'='*60}")
    print(f"Tier {tier} games: {len(df_tier):,}")
    
    # Split by season
    df_train = df_tier[df_tier["season"] <= train_max_season].copy()
    df_val = df_tier[df_tier["season"] >= val_min_season].copy()
    
    print(f"Train: {len(df_train):,} games")
    print(f"Val:   {len(df_val):,} games")
    
    if len(df_train) < 100:
        print(f"[WARN] Tier {tier} has < 100 training samples. Skipping.")
        return None, 0.0, {}
    
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[TARGET_COL].to_numpy()
    X_val = df_val[feature_cols].to_numpy() if len(df_val) > 0 else None
    y_val = df_val[TARGET_COL].to_numpy() if len(df_val) > 0 else None
    
    # Hyperparameter tuning if requested
    if tune and X_val is not None and len(df_val) > 50:
        print(f"\n[Tier {tier}] Tuning hyperparameters ({n_tune_iter} iterations)...")
        best_params = None
        best_mae = float('inf')
        
        # Simplified param grid for tiered models
        param_grids = []
        for _ in range(n_tune_iter):
            params = {
                "learning_rate": np.random.choice([0.03, 0.05, 0.08]),
                "max_depth": np.random.choice([3, 4, 6]),
                "n_estimators": np.random.choice([300, 500]) if model_type == "xgboost" else None,
                "subsample": np.random.choice([0.8, 1.0]) if model_type == "xgboost" else None,
                "colsample_bytree": np.random.choice([0.6, 0.8, 1.0]) if model_type == "xgboost" else None,
                "min_child_weight": np.random.choice([1.0, 5.0]) if model_type == "xgboost" else None,
            }
            if model_type != "xgboost":
                params = {k: v for k, v in params.items() if v is not None}
            param_grids.append(params)
        
        for i, params in enumerate(param_grids[:n_tune_iter]):
            model = make_model(model_type, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            if mae < best_mae:
                best_mae = mae
                best_params = params
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_tune_iter}] Best MAE so far: {best_mae:.3f}")
        
        print(f"[Tier {tier}] Best params: {best_params}")
        print(f"[Tier {tier}] Best MAE: {best_mae:.3f}")
        final_params = best_params
    else:
        # Default params
        if model_type == "xgboost":
            final_params = {
                "learning_rate": 0.05,
                "max_depth": 4,
                "n_estimators": 300,
                "subsample": 1.0,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
            }
        else:
            final_params = {
                "max_iter": 400,
                "learning_rate": 0.05,
                "max_leaf_nodes": 63,
            }
    
    # Train final model
    print(f"\n[Tier {tier}] Training final model...")
    model = make_model(model_type, final_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    if X_val is not None and len(df_val) > 0:
        y_pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred_val)
        rmse = math.sqrt(mean_squared_error(y_val, y_pred_val))
        r2 = r2_score(y_val, y_pred_val)
        
        print(f"[Tier {tier}] Validation Metrics:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.3f}")
        
        residuals = y_val - y_pred_val
        sigma = float(np.std(residuals, ddof=1))
        print(f"  Sigma: {sigma:.3f}")
    else:
        # Use training residuals if no validation
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        sigma = float(np.std(residuals, ddof=1))
        print(f"[Tier {tier}] Training sigma: {sigma:.3f}")
    
    return model, sigma, final_params


def main():
    args = parse_args()
    
    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")
    
    print(f"Loading features from {features_csv}...")
    df = pd.read_csv(features_csv, low_memory=False)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")
    
    if "star_tier_pts" not in df.columns:
        raise ValueError("Features CSV must contain 'star_tier_pts' column")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Features CSV must contain '{TARGET_COL}' column")
    
    # Prefer unified model's feature cols (keeps tiered models in sync with latest features)
    unified_cols = load_unified_feature_cols()
    if unified_cols:
        # Tier selection is based on star_tier_pts, so don't include it as an input feature.
        feature_cols = [c for c in unified_cols if c != "star_tier_pts"]
        print(f"[INFO] Using unified model feature set (minus star_tier_pts): {len(feature_cols)} features")
    else:
        feature_cols = FALLBACK_FEATURE_COLS.copy()
        print(f"[WARN] Could not load unified feature cols; using fallback set: {len(feature_cols)} features")

    # Ensure minutes_pred if needed by feature set
    if "minutes_pred" in feature_cols and "minutes_pred" not in df.columns:
        minutes_model_path = Path("models/minutes_regression.pkl")
        if add_minutes_pred_feature(df, minutes_model_path):
            print("[INFO] Added 'minutes_pred' feature")
        else:
            print("[WARN] Could not add 'minutes_pred', filling with 0.0")
            df["minutes_pred"] = 0.0

    # Ensure has_prop_line if the feature set includes it
    if "has_prop_line" in feature_cols and "has_prop_line" not in df.columns:
        if "prop_pts_line" in df.columns:
            df["has_prop_line"] = (~df["prop_pts_line"].isna()).astype(float)
        else:
            df["has_prop_line"] = 0.0

    # Fill missing columns with safe defaults
    fill_missing_feature_defaults(df, feature_cols)

    # Ensure all feature cols are present
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Tiered training data is missing expected feature columns: {missing_feats}")
    
    # Train models for each tier
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tier_models = {}
    
    for tier in [0, 1, 2, 3]:
        df_tier = df[df["star_tier_pts"] == tier].copy()
        
        if len(df_tier) == 0:
            print(f"\n[Tier {tier}] No data. Skipping.")
            continue
        
        model, sigma, params = train_tier_model(
            df_tier=df_tier,
            tier=tier,
            model_type=args.model_type,
            feature_cols=feature_cols,
            train_max_season=args.train_max_season,
            val_min_season=args.val_min_season,
            tune=args.tune_hyperparams,
            n_tune_iter=args.n_tune_iter,
        )
        
        if model is None:
            continue
        
        # Save model
        model_path = MODELS_DIR / f"points_regression_tier_{tier}.pkl"
        bundle = {
            "model": model,
            "sigma": sigma,
            "feature_cols": feature_cols,
            "tier": tier,
            "params": params,
            **build_model_bundle_metadata(
                target=TARGET_COL,
                training_window=build_training_window(
                    train_min=None,
                    train_max=args.train_max_season,
                    val_min=args.val_min_season,
                    val_max=None,
                ),
                readiness_status="experimental",
                model_type=args.model_type,
                extra={"tier": tier},
            ),
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        
        print(f"[Tier {tier}] Saved model to {model_path}")
        tier_models[tier] = bundle
    
    print(f"\n{'='*60}")
    print("TIERED MODEL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Trained models for tiers: {list(tier_models.keys())}")
    print("\nModel files:")
    for tier in sorted(tier_models.keys()):
        model_path = MODELS_DIR / f"points_regression_tier_{tier}.pkl"
        print(f"  Tier {tier}: {model_path}")


if __name__ == "__main__":
    main()
