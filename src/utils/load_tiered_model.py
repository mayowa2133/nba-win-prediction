#!/usr/bin/env python
"""
load_tiered_model.py

Helper functions to load and use tiered models based on player's star_tier_pts.

Usage:
    from load_tiered_model import load_tiered_model_for_player, predict_with_tiered_model
    
    model_bundle = load_tiered_model_for_player(player_tier, fallback_to_unified=True)
    mu, sigma = predict_with_tiered_model(model_bundle, X, feature_cols)
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

TIER_MODEL_DIR = Path("models")
UNIFIED_MODEL_PATH = Path("models/points_regression.pkl")


def load_tier_model(tier: int) -> Optional[Dict]:
    """Load a tier-specific model."""
    model_path = TIER_MODEL_DIR / f"points_regression_tier_{tier}.pkl"
    if not model_path.exists():
        return None
    
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_unified_model() -> Optional[Dict]:
    """Load the unified (non-tiered) model as fallback."""
    if not UNIFIED_MODEL_PATH.exists():
        return None
    
    with open(UNIFIED_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_tiered_model_for_player(
    star_tier_pts: int,
    fallback_to_unified: bool = True
) -> Optional[Dict]:
    """
    Load the appropriate model for a player based on their star_tier_pts.
    
    Args:
        star_tier_pts: Player's star tier (0, 1, 2, or 3)
        fallback_to_unified: If True, fall back to unified model if tier model not found
    
    Returns:
        Model bundle dict or None
    """
    # Clamp tier to valid range
    tier = max(0, min(3, int(star_tier_pts)))
    
    bundle = load_tier_model(tier)
    if bundle is not None:
        return bundle
    
    if fallback_to_unified:
        return load_unified_model()
    
    return None


def predict_with_tiered_model(
    model_bundle: Dict,
    X: np.ndarray,
    feature_cols: list,
) -> Tuple[float, float]:
    """
    Predict using a tiered model bundle.
    
    Returns:
        (mu, sigma) - predicted mean and sigma
    """
    model = model_bundle["model"]
    sigma = model_bundle.get("sigma", 6.0)
    
    mu = float(model.predict(X)[0])
    return mu, sigma

