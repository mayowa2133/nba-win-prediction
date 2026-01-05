#!/usr/bin/env python
"""
minutes_utils.py

Helper utilities for working with the minutes regression model.

Main entrypoint:
    add_minutes_predictions(df, model_path="models/minutes_regression.pkl")

This:
  - loads models/minutes_regression.pkl
  - uses the stored feature_cols to build X
  - adds a 'min_pred' column to the dataframe with predicted minutes
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import pickle

MINUTES_MODEL_PATH = Path("models") / "minutes_regression.pkl"

# Simple module-level cache so we don't reload the model repeatedly
_minutes_model_cache = None
_minutes_feature_cols_cache: List[str] | None = None


def _load_minutes_model(
    model_path: str | Path = MINUTES_MODEL_PATH,
) -> Tuple[object, List[str]]:
    """
    Load the minutes model bundle from disk and cache it.

    Returns:
        (model, feature_cols)
    """
    global _minutes_model_cache, _minutes_feature_cols_cache

    model_path = Path(model_path)

    if _minutes_model_cache is not None and _minutes_feature_cols_cache is not None:
        return _minutes_model_cache, _minutes_feature_cols_cache

    if not model_path.exists():
        raise FileNotFoundError(
            f"Minutes model bundle not found at {model_path}. "
            f"Did you run build_minutes_regression.py?"
        )

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle.get("model")
    feature_cols = bundle.get("feature_cols")

    if model is None or feature_cols is None:
        raise ValueError(
            f"Minutes model bundle at {model_path} is missing 'model' or 'feature_cols'."
        )

    _minutes_model_cache = model
    _minutes_feature_cols_cache = list(feature_cols)
    return model, _minutes_feature_cols_cache


def add_minutes_predictions(
    df: pd.DataFrame,
    model_path: str | Path = MINUTES_MODEL_PATH,
    col_name: str = "min_pred",
) -> pd.DataFrame:
    """
    Add a predicted-minutes column to a dataframe of features.

    Args:
        df: DataFrame that already contains the minutes feature columns
            used by build_minutes_regression.py.
        model_path: Path to models/minutes_regression.pkl.
        col_name: Name of the column to add (default: 'min_pred').

    Returns:
        A copy of df with a new column 'min_pred' (or col_name).
    """
    model, feature_cols = _load_minutes_model(model_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataframe is missing required minutes features for prediction: "
            f"{missing}"
        )

    X = df[feature_cols].to_numpy()
    preds = model.predict(X)

    # Optional: clamp to a reasonable range [0, 48]
    preds = np.clip(preds, 0.0, 48.0)

    df_out = df.copy()
    df_out[col_name] = preds
    return df_out