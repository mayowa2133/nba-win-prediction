from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import build_points_regression as regression_module


class SpyRegressor:
    def __init__(self, seen_targets: list[np.ndarray]):
        self.seen_targets = seen_targets
        self.mean_ = 0.0

    def fit(self, X, y, sample_weight=None):
        self.seen_targets.append(np.array(y, copy=True))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


def test_walk_forward_tune_uses_supplied_target_col(monkeypatch):
    seen_targets: list[np.ndarray] = []

    monkeypatch.setattr(
        regression_module,
        "build_model",
        lambda model_type, params: SpyRegressor(seen_targets),
    )
    monkeypatch.setattr(
        regression_module,
        "generate_histgb_param_grid",
        lambda: [{"stub": 1}],
    )

    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2024, 2024, 2024],
            "feature_a": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "target_pts": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
            "target_reb": [5.0, 7.0, 9.0, 6.0, 8.0, 10.0],
        }
    )

    best_params, sigma = regression_module.walk_forward_tune(
        df=df,
        feature_cols=["feature_a"],
        target_col="target_reb",
        train_min=2023,
        train_max=2023,
        val_min=2024,
        val_max=2024,
        model_type="histgb",
        n_tune_iter=1,
        sample_weights=None,
    )

    assert best_params == {"stub": 1}
    assert sigma >= 0.0
    assert seen_targets
    assert all(np.max(targets) <= 10.0 for targets in seen_targets)

