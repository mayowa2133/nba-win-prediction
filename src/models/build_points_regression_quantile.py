#!/usr/bin/env python
"""
build_points_regression_quantile.py

Train quantile regression models to predict different percentiles of the score distribution.
This addresses systematic bias: model under-predicts high scores and over-predicts low scores.

Approach:
1. Train models for 10th, 50th (median), and 90th percentiles
2. Use base model prediction to determine which quantile to use
3. For high expected scores -> use 90th percentile
4. For low expected scores -> use 10th percentile
5. For medium scores -> use 50th percentile (median)

Outputs:
  - models/points_regression_quantile_10.pkl (10th percentile)
  - models/points_regression_quantile_50.pkl (50th percentile / median)
  - models/points_regression_quantile_90.pkl (90th percentile)
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.artifact_metadata import (
    build_model_bundle_metadata,
    build_training_window,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

FEATURES_CSV = Path("data/player_points_features_with_vegas.csv")
MODELS_DIR = Path("models")
TARGET_COL = "target_pts"

# Quantiles to predict
QUANTILES = [0.10, 0.50, 0.90]  # 10th, 50th (median), 90th percentiles


def load_base_model() -> Dict:
    """Load the base regression model to get feature columns."""
    base_model_path = MODELS_DIR / "points_regression.pkl"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    with open(base_model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def train_quantile_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    quantile: float,
    model_type: str = "xgboost",
) -> tuple[object, float]:
    """
    Train a quantile regression model for a specific quantile.
    
    Returns:
        (model, validation_mae)
    """
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[TARGET_COL].to_numpy()
    X_val = df_val[feature_cols].to_numpy() if len(df_val) > 0 else None
    y_val = df_val[TARGET_COL].to_numpy() if len(df_val) > 0 else None
    
    if model_type == "xgboost":
        if xgb is None:
            raise ImportError("xgboost not installed")
        
        # XGBoost quantile regression
        model = xgb.XGBRegressor(
            objective=f"reg:quantileerror",
            quantile_alpha=quantile,
            random_state=42,
            n_jobs=-1,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=300,
            subsample=1.0,
            colsample_bytree=0.8,
            min_child_weight=1.0,
        )
    else:
        # For other models, we'd need to use quantile loss
        # For now, use XGBoost
        raise ValueError("Only xgboost supported for quantile regression")
    
    print(f"\n[Quantile {quantile:.2f}] Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    if X_val is not None and len(df_val) > 0:
        y_pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        # Calculate quantile-specific metrics
        # For 10th percentile: should under-predict (most values above)
        # For 90th percentile: should over-predict (most values below)
        # For 50th percentile: should be balanced
        
        coverage = (y_val <= y_pred_val).mean()  # What % of actuals are below prediction
        expected_coverage = quantile  # Should match quantile
        
        print(f"[Quantile {quantile:.2f}] Validation Metrics:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Coverage: {coverage:.1%} (expected: {expected_coverage:.1%})")
        
        return model, mae
    else:
        # Use training metrics if no validation
        y_pred_train = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred_train)
        print(f"[Quantile {quantile:.2f}] Training MAE: {mae:.3f}")
        return model, mae


def main():
    parser = argparse.ArgumentParser(
        description="Train quantile regression models (10th, 50th, 90th percentiles)"
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV),
        help="Path to features CSV",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=2024,
        help="Max season for training",
    )
    parser.add_argument(
        "--val-min-season",
        type=int,
        default=2025,
        help="Min season for validation",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost"],
        help="Model type (default: xgboost)",
    )
    args = parser.parse_args()
    
    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")
    
    print(f"Loading features from {features_csv}...")
    df = pd.read_csv(features_csv, low_memory=False)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Features CSV must contain '{TARGET_COL}' column")
    
    # Load base model to get feature columns - use EXACT same features
    base_bundle = load_base_model()
    base_feature_cols = base_bundle.get("feature_cols", [])
    
    # Handle missing features gracefully (same as base model)
    vegas_cols = ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]
    for col in vegas_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    injury_cols = ["is_injured", "days_since_last_dnp", "dnp_count_last_10"]
    for col in injury_cols:
        if col not in df.columns:
            if col == "is_injured":
                df[col] = 0
            elif col == "days_since_last_dnp":
                df[col] = 999
            else:
                df[col] = 0.0
    
    # Phase 4A features
    game_script_cols = ["blowout_prob", "is_likely_blowout", "garbage_time_minutes_est", "vegas_spread_abs_normalized"]
    for col in game_script_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    player_opp_cols = [
        "player_vs_opp_pts_avg_career", "player_vs_opp_pts_avg_last_5",
        "player_vs_opp_minutes_avg_career", "player_vs_opp_minutes_avg_last_5",
        "player_vs_opp_games_count"
    ]
    for col in player_opp_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    enhanced_dvp_cols = [
        "opp_fg_pct_allowed_vs_pos_roll5", "opp_fg_pct_allowed_vs_pos_roll15",
        "opp_3pt_pct_allowed_vs_pos_roll5", "opp_3pt_pct_allowed_vs_pos_roll15"
    ]
    for col in enhanced_dvp_cols:
        if col not in df.columns:
            if "fg_pct" in col:
                df[col] = 0.45
            elif "3pt_pct" in col:
                df[col] = 0.35
            else:
                df[col] = 0.0
    
    # Add minutes_pred if it's in base model features
    if "minutes_pred" in base_feature_cols:
        from build_points_regression import add_minutes_pred_feature
        MINUTES_MODEL_PATH = Path("models/minutes_regression.pkl")
        ok = add_minutes_pred_feature(df, MINUTES_MODEL_PATH)
        if not ok:
            print("[WARN] Could not add minutes_pred; will fill with 0.0")
            df["minutes_pred"] = 0.0
    
    # Add prop features if they're in base model features
    prop_base_cols = ["prop_pts_line", "prop_over_odds_best", "prop_under_odds_best"]
    prop_indicator = "has_prop_line"
    prop_derived_cols = [
        "prop_minus_pts_roll5", "prop_minus_pts_roll15",
        "prop_minus_season_mean", "prop_minus_career_mean", "prop_minus_model_baseline"
    ]
    
    # Check if base model uses prop features
    uses_props = any(c in base_feature_cols for c in prop_base_cols + [prop_indicator] + prop_derived_cols)
    
    if uses_props:
        # Add prop indicator
        if prop_indicator in base_feature_cols:
            if "prop_pts_line" in df.columns:
                df[prop_indicator] = (~df["prop_pts_line"].isna()).astype(float)
            else:
                df[prop_indicator] = 0.0
        
        # Fill prop base cols
        for col in prop_base_cols:
            if col in base_feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)
        
        # Fill prop derived cols
        for col in prop_derived_cols:
            if col in base_feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)
    
    # Use EXACT same feature columns as base model (don't filter)
    feature_cols = base_feature_cols.copy()
    
    # Ensure all features exist in df (fill missing with 0.0)
    for col in feature_cols:
        if col not in df.columns:
            print(f"[WARN] Feature '{col}' from base model not in data; filling with 0.0")
            if col == "has_prop_line":
                df[col] = 0.0
            elif "pct" in col:
                if "fg_pct" in col:
                    df[col] = 0.45
                elif "3pt_pct" in col:
                    df[col] = 0.35
                else:
                    df[col] = 0.0
            else:
                df[col] = 0.0
    
    print(f"\nUsing {len(feature_cols)} features")
    
    # Split by season
    df_train = df[df["season"] <= args.train_max_season].copy()
    df_val = df[df["season"] >= args.val_min_season].copy()
    
    print(f"\nTrain: {len(df_train):,} games")
    print(f"Val:   {len(df_val):,} games")
    
    if len(df_train) < 1000:
        raise ValueError(f"Insufficient training data: {len(df_train):,} rows")
    
    # Train quantile models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    quantile_models = {}
    
    for quantile in QUANTILES:
        model, val_mae = train_quantile_model(
            df_train=df_train,
            df_val=df_val,
            feature_cols=feature_cols,
            quantile=quantile,
            model_type=args.model_type,
        )
        
        # Save model
        model_path = MODELS_DIR / f"points_regression_quantile_{int(quantile*100)}.pkl"
        bundle = {
            "model": model,
            "quantile": quantile,
            "feature_cols": feature_cols,
            "val_mae": val_mae,
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
                extra={"quantile": quantile},
            ),
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        
        print(f"[Quantile {quantile:.2f}] Saved model to {model_path}")
        quantile_models[quantile] = bundle
    
    print(f"\n{'='*60}")
    print("QUANTILE REGRESSION TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Trained models for quantiles: {QUANTILES}")
    print("\nModel files:")
    for quantile in QUANTILES:
        model_path = MODELS_DIR / f"points_regression_quantile_{int(quantile*100)}.pkl"
        print(f"  Quantile {quantile:.2f}: {model_path}")


if __name__ == "__main__":
    main()
