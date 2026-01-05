#!/usr/bin/env python
"""
build_points_model.py

Train:
  1) A logistic regression model to predict P(points > 15.5)
  2) A regression model to predict expected points

Outputs:
  - models/points_over_15_5.pkl
  - models/points_regression.pkl
"""

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DATA_CSV = Path("data/player_points_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Feature columns we’ll feed into both models
FEATURE_COLS = [
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
    "is_home",
    "days_since_last_game",
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
]


def main():
    print(f"Loading features from {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)

    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    # Drop rows with any NaNs in our feature or target columns
    needed_cols = FEATURE_COLS + ["pts", "season"]
    before = len(df)
    df = df.dropna(subset=needed_cols)
    after = len(df)
    if after < before:
        print(f"Dropped {before - after:,} rows due to NaNs in features/target.")

    if df.empty:
        print("No data left after cleaning. Aborting.")
        return

    seasons = sorted(df["season"].unique())
    print("Seasons in dataset:", seasons)

    # Same logic as before: train on <= 2022, validate on > 2022
    train_df = df[df["season"] <= 2022].copy()
    val_df = df[df["season"] > 2022].copy()

    print(f"Using seasons <= 2022 for TRAIN")
    print(f"Using seasons  > 2022 for HOLDOUT / VAL")
    print(f"Train rows: {len(train_df):,}")
    print(f"Val   rows: {len(val_df):,}")

    if train_df.empty or val_df.empty:
        print("Train or validation split is empty. Check your season logic.")
        return

    # ------------------------------------------------------------------
    # 1) Logistic regression: P(points > 15.5)
    # ------------------------------------------------------------------
    print("\nTraining LogisticRegression model for OVER 15.5 ...")

    X_train = train_df[FEATURE_COLS]  # keep as DataFrame (fixes feature-name warning)
    X_val = val_df[FEATURE_COLS]

    y_train_class = (train_df["pts"] > 15.5).astype(int)
    y_val_class = (val_df["pts"] > 15.5).astype(int)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train_class)

    # Evaluate on holdout seasons
    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val_class, val_probs)
    preds = (val_probs >= 0.5).astype(int)
    acc = accuracy_score(y_val_class, preds)
    brier = brier_score_loss(y_val_class, val_probs)

    print("\nEvaluating logistic model on holdout seasons...")
    print(f"AUC:          {auc:.3f}")
    print(f"Accuracy:     {acc:.3f}")
    print(f"Brier score:  {brier:.3f} (lower is better)")
    print(f"Positives in val (over 15.5 pts): {y_val_class.mean() * 100:.3f}%\n")
    print("Classification report (threshold = 0.5):")
    print(classification_report(y_val_class, preds))

    # Save logistic model bundle
    clf_bundle = {
        "model_type": "logistic_over_15_5",
        "threshold_points": 15.5,
        "features": FEATURE_COLS,
        "model": clf,
    }

    out_logistic = MODEL_DIR / "points_over_15_5.pkl"
    joblib.dump(clf_bundle, out_logistic)
    print(f"\nSaved logistic model to {out_logistic}")

    # ------------------------------------------------------------------
    # 2) Regression model: predict expected points
    # ------------------------------------------------------------------
    print("\nTraining Ridge regression model for expected points ...")

    y_train_reg = train_df["pts"].astype(float)
    y_val_reg = val_df["pts"].astype(float)

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X_train, y_train_reg)

    val_pred_reg = reg.predict(X_val)

    mae = mean_absolute_error(y_val_reg, val_pred_reg)
    rmse = math.sqrt(mean_squared_error(y_val_reg, val_pred_reg))
    r2 = r2_score(y_val_reg, val_pred_reg)

    print("\nEvaluating regression model on holdout seasons...")
    print(f"MAE:   {mae:.3f}")
    print(f"RMSE:  {rmse:.3f}")
    print(f"R^2:   {r2:.3f}")

    # Estimate residual std dev for a simple normal approximation later
    residuals = y_val_reg - val_pred_reg
    sigma = float(residuals.std(ddof=1))
    print(f"Estimated residual sigma: {sigma:.3f}")

    reg_bundle = {
        "model_type": "ridge_points_regression",
        "features": FEATURE_COLS,
        "model": reg,
        "sigma": sigma,
    }

    out_reg = MODEL_DIR / "points_regression.pkl"
    joblib.dump(reg_bundle, out_reg)
    print(f"Saved regression model to {out_reg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
