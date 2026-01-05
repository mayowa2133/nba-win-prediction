#!/usr/bin/env python
"""
load_quantile_model.py

Helper functions to load and use quantile regression models.
Uses base model prediction to select appropriate quantile.

Logic:
- Low expected scores (<8 pts) -> use 10th percentile (conservative, avoid over-prediction)
- Medium expected scores (8-18 pts) -> use 50th percentile (median)
- High expected scores (>18 pts) -> use 90th percentile (optimistic, avoid under-prediction)
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

QUANTILE_MODELS_DIR = Path("models")
BASE_MODEL_PATH = Path("models/points_regression.pkl")

# Thresholds for selecting quantile based on base model prediction
# Adjusted to be more conservative - only use extreme quantiles for very low/high scores
LOW_THRESHOLD = 5.0   # Use 10th percentile for scores < 5
HIGH_THRESHOLD = 22.0  # Use 90th percentile for scores > 22


def load_quantile_models() -> Dict[float, Dict]:
    """Load all quantile models (10th, 50th, 90th percentiles)."""
    quantiles = [0.10, 0.50, 0.90]
    models = {}
    
    for quantile in quantiles:
        model_path = QUANTILE_MODELS_DIR / f"points_regression_quantile_{int(quantile*100)}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            models[quantile] = bundle
        else:
            print(f"[WARN] Quantile model not found: {model_path}")
    
    return models


def load_base_model() -> Optional[Dict]:
    """Load the base regression model."""
    if not BASE_MODEL_PATH.exists():
        return None
    
    with open(BASE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict_with_quantile_ensemble(
    X: np.ndarray,
    feature_cols: list,
    quantile_models: Dict[float, Dict],
    base_model: Optional[Dict] = None,
    use_weighted_blend: bool = True,
) -> Tuple[float, str]:
    """
    Predict using quantile ensemble.
    
    Args:
        use_weighted_blend: If True, blend quantiles based on base prediction.
                          If False, use hard thresholds.
    
    Returns:
        (predicted_score, quantile_used)
    """
    # First, get base model prediction to determine which quantile to use
    if base_model is None:
        base_model = load_base_model()
    
    if base_model is None:
        # Fallback: use median (50th percentile)
        median_model = quantile_models.get(0.50)
        if median_model is None:
            raise ValueError("No quantile models available")
        pred = float(median_model["model"].predict(X)[0])
        return pred, "50th (fallback)"
    
    base_pred = float(base_model["model"].predict(X)[0])
    
    # Get predictions from all quantile models
    pred_10 = float(quantile_models[0.10]["model"].predict(X)[0])
    pred_50 = float(quantile_models[0.50]["model"].predict(X)[0])
    pred_90 = float(quantile_models[0.90]["model"].predict(X)[0])
    
    if use_weighted_blend:
        # Bias correction approach: use quantile models to adjust base prediction
        # Only apply correction for extreme cases where we know base model is biased
        
        if base_pred < 4.0:
            # Very low: base over-predicts, use 10th percentile to correct down
            # Blend: 70% base + 30% 10th percentile (to reduce but not eliminate base)
            pred = 0.7 * base_pred + 0.3 * pred_10
            quantile_name = "bias-correct low (<4)"
        elif base_pred < 7.0:
            # Low: slight correction
            t = (base_pred - 4.0) / (7.0 - 4.0)
            pred = (1.0 - 0.3 * (1-t)) * base_pred + 0.3 * (1-t) * pred_10
            quantile_name = f"bias-correct low-med (t={t:.2f})"
        elif base_pred < 24.0:
            # Medium: use base model (no correction)
            pred = base_pred
            quantile_name = "base (medium)"
        elif base_pred < 28.0:
            # High: slight correction up
            t = (base_pred - 24.0) / (28.0 - 24.0)
            pred = (1.0 - 0.3 * t) * base_pred + 0.3 * t * pred_90
            quantile_name = f"bias-correct med-high (t={t:.2f})"
        else:
            # Very high: base under-predicts, use 90th percentile to correct up
            # Blend: 70% base + 30% 90th percentile
            pred = 0.7 * base_pred + 0.3 * pred_90
            quantile_name = "bias-correct high (>28)"
    else:
        # Hard thresholds (original approach)
        if base_pred < LOW_THRESHOLD:
            quantile = 0.10
            quantile_name = "10th (low)"
            pred = pred_10
        elif base_pred > HIGH_THRESHOLD:
            quantile = 0.90
            quantile_name = "90th (high)"
            pred = pred_90
        else:
            quantile = 0.50
            quantile_name = "50th (medium)"
            pred = pred_50
    
    return pred, quantile_name

