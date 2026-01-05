#!/usr/bin/env python
"""
predict_player_props_from_features.py

Train an XGBRegressor to predict player points from rolling features.

Usage:
  python predict_player_props_from_features.py

This script:
  - loads data/player_points_features.csv
  - splits into train / test by season (simple walk-forward: train <= split_year)
  - trains XGBRegressor
  - prints basic metrics
  - saves model to models/player_points_xgb.json
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATA_DIR = Path("data")
MODELS_DIR = Path("models")

FEATURES_CSV = DATA_DIR / "player_points_features.csv"
MODEL_PATH = MODELS_DIR / "player_points_xgb.json"

# You can tune this
TEST_START_SEASON = 2023  # train on seasons < 2023, test on >= 2023


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    return df


def train_test_split_by_season(df: pd.DataFrame, test_start_season: int):
    train_df = df[df["season"] < test_start_season].copy()
    test_df = df[df["season"] >= test_start_season].copy()

    id_cols = [
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_team_abbrev",
        "is_home",
    ]
    target = "pts"

    feature_cols = [c for c in df.columns if c not in id_cols + [target]]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target].values

    return X_train, y_train, X_test, y_test, feature_cols, train_df, test_df


def train_model(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def eval_model(model, X, y, split_name: str):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"{split_name} MAE: {mae:.3f}")
    print(f"{split_name} R^2: {r2:.3f}")

    # You can also print correlation
    corr = np.corrcoef(y, preds)[0, 1]
    print(f"{split_name} corr(actual, pred): {corr:.3f}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        train_df,
        test_df,
    ) = train_test_split_by_season(df, TEST_START_SEASON)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    model = train_model(X_train, y_train)

    print("\n=== Train performance ===")
    eval_model(model, X_train, y_train, "Train")

    print("\n=== Test performance ===")
    eval_model(model, X_test, y_test, "Test")

    # Save model
    model.save_model(str(MODEL_PATH))
    print(f"\nSaved XGBRegressor model to {MODEL_PATH}")
    print(f"Feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()
