#!/usr/bin/env python
"""
sigma_features.py

Shared helpers for sigma-model feature engineering.

The sigma model predicts per-row uncertainty (|residual|) and benefits from
non-linear transforms of mu_hat and interactions with game context.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


SIGMA_DERIVED_COLS: List[str] = [
    "mu_hat_sq",
    "mu_hat_x_blowout_prob",
    "mu_hat_x_teammate_out_count",
    "mu_hat_x_minutes_pred",
]


def add_sigma_derived_features_df(df: pd.DataFrame) -> None:
    """
    Adds derived sigma features in-place. Requires `mu_hat` column.
    Missing base columns are treated as 0.0.
    """
    if "mu_hat" not in df.columns:
        raise ValueError("add_sigma_derived_features_df requires 'mu_hat' column")

    mu = pd.to_numeric(df["mu_hat"], errors="coerce").fillna(0.0)
    df["mu_hat_sq"] = mu * mu

    blowout = pd.to_numeric(df.get("blowout_prob", 0.0), errors="coerce").fillna(0.0)
    df["mu_hat_x_blowout_prob"] = mu * blowout

    out_ct = pd.to_numeric(df.get("teammate_out_count", 0.0), errors="coerce").fillna(0.0)
    df["mu_hat_x_teammate_out_count"] = mu * out_ct

    mins_pred = pd.to_numeric(df.get("minutes_pred", 0.0), errors="coerce").fillna(0.0)
    df["mu_hat_x_minutes_pred"] = mu * mins_pred


def add_sigma_derived_features_map(feat_map: Dict[str, float]) -> None:
    """
    Adds derived sigma features to a feature mapping in-place.
    Requires feat_map['mu_hat'].
    """
    mu = float(feat_map.get("mu_hat", 0.0))
    feat_map["mu_hat_sq"] = mu * mu
    feat_map["mu_hat_x_blowout_prob"] = mu * float(feat_map.get("blowout_prob", 0.0))
    feat_map["mu_hat_x_teammate_out_count"] = mu * float(feat_map.get("teammate_out_count", 0.0))
    feat_map["mu_hat_x_minutes_pred"] = mu * float(feat_map.get("minutes_pred", 0.0))


