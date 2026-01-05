#!/usr/bin/env python
"""
validate_quantile_models.py

Compare quantile regression ensemble vs base model on validation set.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.load_quantile_model import predict_with_quantile_ensemble, load_quantile_models, load_base_model

FEATURES_CSV = Path("data/player_points_features_with_vegas.csv")
VAL_PREDS_CSV = Path("data/points_regression_val_preds.csv")


def main():
    print("="*70)
    print("QUANTILE REGRESSION VALIDATION")
    print("="*70)
    print()
    
    # Load features (validation set: season >= 2025)
    print("Loading features...")
    df_features = pd.read_csv(FEATURES_CSV, low_memory=False)
    df_val = df_features[df_features["season"] >= 2025].copy()
    print(f"Loaded {len(df_val):,} validation rows (season >= 2025)")
    print()
    
    # Load models
    print("Loading models...")
    base_model = load_base_model()
    quantile_models = load_quantile_models()
    
    if not base_model or not quantile_models:
        print("❌ Failed to load models")
        return
    
    feature_cols = base_model["feature_cols"]
    print(f"Using {len(feature_cols)} features")
    print()
    
    # Fill missing features with defaults (same as model training)
    missing_feature_cols = [c for c in feature_cols if c not in df_val.columns]
    if missing_feature_cols:
        print(f"⚠ Missing features in data (will fill with defaults): {missing_feature_cols}")
    
    for col in missing_feature_cols:
        if col == "has_prop_line":
            df_val[col] = 0.0
        elif col == "minutes_pred":
            df_val[col] = 0.0
        elif "pct" in col:
            if "fg_pct" in col:
                df_val[col] = 0.45
            elif "3pt_pct" in col:
                df_val[col] = 0.35
            else:
                df_val[col] = 0.0
        else:
            df_val[col] = 0.0
    
    # Get feature vectors and true values
    X_val = df_val[feature_cols].to_numpy()
    y_true = df_val["target_pts"].to_numpy()
    
    # Base model predictions
    print("Computing base model predictions...")
    base_model_obj = base_model["model"]
    y_pred_base = base_model_obj.predict(X_val)
    
    # Quantile ensemble predictions
    print("Computing quantile ensemble predictions...")
    y_pred_quantile = []
    quantile_used_counts = {"10th (low)": 0, "50th (medium)": 0, "90th (high)": 0}
    
    for i in range(len(X_val)):
        x = X_val[i:i+1]
        pred, quantile_name = predict_with_quantile_ensemble(
            X=x,
            feature_cols=feature_cols,
            quantile_models=quantile_models,
            base_model=base_model,
        )
        y_pred_quantile.append(pred)
        quantile_used_counts[quantile_name] = quantile_used_counts.get(quantile_name, 0) + 1
    
    y_pred_quantile = np.array(y_pred_quantile)
    
    print("Quantile usage:")
    for name, count in quantile_used_counts.items():
        print(f"  {name}: {count:,} ({100*count/len(y_pred_quantile):.1f}%)")
    print()
    
    # Compare metrics
    print("="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print()
    
    mae_base = mean_absolute_error(y_true, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
    r2_base = r2_score(y_true, y_pred_base)
    
    mae_quantile = mean_absolute_error(y_true, y_pred_quantile)
    rmse_quantile = np.sqrt(mean_squared_error(y_true, y_pred_quantile))
    r2_quantile = r2_score(y_true, y_pred_quantile)
    
    print("Metric          | Base Model    | Quantile Ensemble | Improvement")
    print("-"*70)
    print(f"MAE             | {mae_base:13.3f} | {mae_quantile:17.3f} | {mae_base-mae_quantile:+11.3f} ({100*(mae_base-mae_quantile)/mae_base:+.2f}%)")
    print(f"RMSE            | {rmse_base:13.3f} | {rmse_quantile:17.3f} | {rmse_base-rmse_quantile:+11.3f} ({100*(rmse_base-rmse_quantile)/rmse_base:+.2f}%)")
    print(f"R²              | {r2_base:13.3f} | {r2_quantile:17.3f} | {r2_quantile-r2_base:+11.3f} ({100*(r2_quantile-r2_base)/r2_base:+.2f}%)")
    print()
    
    # Analyze by score range
    print("="*70)
    print("IMPROVEMENT BY SCORE RANGE")
    print("="*70)
    print()
    
    low_mask = y_true < 5
    medium_mask = (y_true >= 5) & (y_true <= 25)
    high_mask = y_true > 25
    
    for name, mask in [("Low (<5)", low_mask), ("Medium (5-25)", medium_mask), ("High (>25)", high_mask)]:
        if mask.sum() > 0:
            mae_b = mean_absolute_error(y_true[mask], y_pred_base[mask])
            mae_q = mean_absolute_error(y_true[mask], y_pred_quantile[mask])
            improvement = mae_b - mae_q
            print(f"{name:15s}: Base={mae_b:.3f}, Quantile={mae_q:.3f}, Improvement={improvement:+.3f} ({100*improvement/mae_b:+.2f}%)")
    print()
    
    # Bias analysis
    print("="*70)
    print("BIAS REDUCTION")
    print("="*70)
    print()
    
    for name, mask in [("Low (<5)", low_mask), ("Medium (5-25)", medium_mask), ("High (>25)", high_mask)]:
        if mask.sum() > 0:
            bias_base = y_pred_base[mask].mean() - y_true[mask].mean()
            bias_quantile = y_pred_quantile[mask].mean() - y_true[mask].mean()
            bias_reduction = abs(bias_base) - abs(bias_quantile)
            print(f"{name:15s}:")
            print(f"  Base bias:     {bias_base:+.2f}")
            print(f"  Quantile bias: {bias_quantile:+.2f}")
            print(f"  Bias reduction: {bias_reduction:+.2f} points")
            print()


if __name__ == "__main__":
    main()

