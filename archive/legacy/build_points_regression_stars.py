#!/usr/bin/env python
"""
build_points_regression_stars.py

Train a regression model to predict player points using the engineered features in
data/player_points_features.csv, but restricted to *star scorers*:

    star_tier_pts >= 2

This is intended to complement the global model by focusing on the players
who matter most for props (primary/secondary options and stars).

Outputs:
  - models/points_regression_stars.pkl: {
        "model": regressor (HistGradientBoostingRegressor or XGBRegressor),
        "sigma": float (std of val residuals),
        "feature_cols": [list of feature names]
    }
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import xgboost lazily; it's only needed if model_type = "xgboost"
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None  # we will check this if the user requests xgboost


FEATURES_CSV = Path("data") / "player_points_features.csv"
MODEL_PATH = Path("models") / "points_regression_stars.pkl"

# Optional upstream minutes model
MINUTES_MODEL_PATH = Path("models") / "minutes_regression.pkl"

TARGET_COL = "target_pts"

# Must match build_player_points_features.py
BASE_FEATURE_COLS = [
    # core rolling features
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

    # NEW: usage events (approx USG-style volume)
    "usg_events_roll5",
    "usg_events_roll15",

    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
    "days_since_last_game",
    "is_home",

    # matchup/env features
    "opp_dvp_pos_pts_roll5",
    "opp_dvp_pos_pts_roll15",
    "team_pace_roll5",
    "team_pace_roll15",

    # player baselines
    "player_pts_career_mean",
    "player_pts_season_mean",
    "player_minutes_career_mean",
    "player_minutes_season_mean",

    # role vs career / star tiers
    "rel_minutes_vs_career",
    "rel_pts_vs_career",
    "star_tier_pts",
    "star_tier_minutes",

    # trends (form / usage)
    "pts_trend_5_15",
    "minutes_trend_5_15",
    "fga_trend_5_15",

    # volatility
    "pts_std5",
    "minutes_std5",
    "fga_std5",

    # usage ratios
    "pts_per_min_roll5",
    "fga_per_min_roll5",
    "fta_per_min_roll5",

    # rest flags
    "is_b2b",
    "is_long_rest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a star-only regression model for player points "
            "(star_tier_pts >= 2) with season-based train/val splits."
        )
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV),
        help="Path to player_points_features.csv (default: data/player_points_features.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Where to save the regression model bundle (default: models/points_regression_stars.pkl)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="histgb",
        choices=["histgb", "xgboost"],
        help="Which regressor to train: 'histgb' or 'xgboost'. Default: histgb.",
    )
    parser.add_argument(
        "--train-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include in training. Default: min season in the data.",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include in training. Default: second-most-recent season.",
    )
    parser.add_argument(
        "--val-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include in validation. Default: train_max_season + 1.",
    )
    parser.add_argument(
        "--val-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include in validation. Default: max season.",
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="If set, perform walk-forward hyperparameter tuning over the validation seasons.",
    )
    parser.add_argument(
        "--n-tune-iter",
        type=int,
        default=20,
        help="Number of hyperparameter configs to evaluate during tuning (default: 20).",
    )
    parser.add_argument(
        "--use-minutes-pred",
        action="store_true",
        help=(
            "If set, load models/minutes_regression.pkl and add its predictions as "
            "'minutes_pred' feature for all rows."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Minutes prediction helper
# ---------------------------------------------------------------------


def add_minutes_pred_feature(df: pd.DataFrame, minutes_model_path: Path) -> bool:
    if not minutes_model_path.exists():
        print(f"[WARN] Minutes model not found at {minutes_model_path}; skipping minutes_pred feature.")
        return False

    try:
        with open(minutes_model_path, "rb") as f:
            minutes_bundle = pickle.load(f)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Failed to load minutes model from {minutes_model_path}: {e}")
        return False

    minutes_model = minutes_bundle.get("model")
    minutes_feature_cols = minutes_bundle.get("feature_cols")

    if minutes_model is None or minutes_feature_cols is None:
        print("[WARN] Minutes model bundle missing 'model' or 'feature_cols'; skipping minutes_pred.")
        return False

    missing = [c for c in minutes_feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Some minutes feature columns missing in df: {missing}; skipping minutes_pred.")
        return False

    X_min = df[minutes_feature_cols].to_numpy()
    print(f"[INFO] Computing minutes_pred for {len(df):,} rows using minutes model...")
    df["minutes_pred"] = minutes_model.predict(X_min)
    print("[INFO] Added 'minutes_pred' column to features DataFrame.")
    return True


# ---------------------------------------------------------------------
# Model factories + param grids (same as main script)
# ---------------------------------------------------------------------


def make_histgb_model(params: Dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=params.get("max_iter", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("max_leaf_nodes", 63),
        min_samples_leaf=params.get("min_samples_leaf", 50),
        l2_regularization=params.get("l2_regularization", 0.0),
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


def make_xgb_model(params: Dict):
    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install it with 'pip install xgboost' or use --model-type histgb."
        )

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        random_state=42,
        n_jobs=-1,
    )


def generate_histgb_param_grid() -> List[Dict]:
    learning_rates = [0.03, 0.05, 0.08]
    max_leaf_nodes_list = [31, 63, 127]
    min_samples_leaf_list = [20, 50, 100]
    l2_regs = [0.0, 0.01, 0.1]
    max_iters = [300, 400, 600]

    configs = []
    for lr in learning_rates:
        for leaf_nodes in max_leaf_nodes_list:
            for min_leaf in min_samples_leaf_list:
                for l2 in l2_regs:
                    for max_iter in max_iters:
                        configs.append(
                            {
                                "learning_rate": lr,
                                "max_leaf_nodes": leaf_nodes,
                                "min_samples_leaf": min_leaf,
                                "l2_regularization": l2,
                                "max_iter": max_iter,
                            }
                        )
    return configs


def generate_xgb_param_grid() -> List[Dict]:
    learning_rates = [0.03, 0.05, 0.08]
    max_depths = [3, 4, 6]
    n_estimators_list = [300, 500]
    subsamples = [0.8, 1.0]
    colsample_bytree_list = [0.6, 0.8, 1.0]
    min_child_weights = [1.0, 5.0]

    configs = []
    for lr in learning_rates:
        for depth in max_depths:
            for n_est in n_estimators_list:
                for subs in subsamples:
                    for col in colsample_bytree_list:
                        for mcw in min_child_weights:
                            configs.append(
                                {
                                    "learning_rate": lr,
                                    "max_depth": depth,
                                    "n_estimators": n_est,
                                    "subsample": subs,
                                    "colsample_bytree": col,
                                    "min_child_weight": mcw,
                                    "reg_lambda": 1.0,
                                    "reg_alpha": 0.0,
                                }
                            )
    return configs


def build_model(model_type: str, params: Dict):
    if model_type == "histgb":
        return make_histgb_model(params)
    elif model_type == "xgboost":
        return make_xgb_model(params)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------
# Walk-forward tuning (unchanged, but now runs on star-only df)
# ---------------------------------------------------------------------


def walk_forward_tune(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_min: int,
    train_max: int,
    val_min: int,
    val_max: int,
    model_type: str,
    n_tune_iter: int,
) -> Tuple[Dict, float]:
    val_seasons = sorted(
        int(s) for s in df[(df["season"] >= val_min) & (df["season"] <= val_max)]["season"].unique()
    )
    if not val_seasons:
        raise RuntimeError(
            f"[WFV] No validation seasons found in range [{val_min}, {val_max}] for walk-forward tuning."
        )

    if model_type == "histgb":
        all_configs = generate_histgb_param_grid()
    else:
        all_configs = generate_xgb_param_grid()

    total_configs = len(all_configs)
    n_tune = min(n_tune_iter, total_configs)

    rng = np.random.RandomState(42)
    if n_tune < total_configs:
        idx = rng.choice(total_configs, size=n_tune, replace=False)
        configs = [all_configs[i] for i in idx]
    else:
        configs = all_configs

    print(f"\n[WFV] Running walk-forward tuning over validation seasons: {val_seasons}")
    print(f"[WFV] Evaluating {len(configs)} hyperparameter configs (out of {total_configs} total).")

    best_mae = float("inf")
    best_params: Optional[Dict] = None
    best_residuals: List[float] = []

    for i, params in enumerate(configs, start=1):
        fold_maes: List[float] = []
        fold_residuals: List[float] = []

        for season_val in val_seasons:
            train_upper = min(train_max, season_val - 1)
            if train_upper < train_min:
                continue

            train_mask = (df["season"] >= train_min) & (df["season"] <= train_upper)
            val_mask = df["season"] == season_val

            df_train_f = df[train_mask]
            df_val_f = df[val_mask]

            if df_train_f.empty or df_val_f.empty:
                continue

            X_train_f = df_train_f[feature_cols].to_numpy()
            y_train_f = df_train_f[TARGET_COL].to_numpy()
            X_val_f = df_val_f[feature_cols].to_numpy()
            y_val_f = df_val_f[TARGET_COL].to_numpy()

            model = build_model(model_type, params)
            model.fit(X_train_f, y_train_f)

            y_pred_f = model.predict(X_val_f)
            mae_f = mean_absolute_error(y_val_f, y_pred_f)
            fold_maes.append(mae_f)
            fold_residuals.extend(list(y_val_f - y_pred_f))

        if not fold_maes:
            avg_mae = float("inf")
        else:
            avg_mae = float(np.mean(fold_maes))

        print(f"[WFV] Config {i}/{len(configs)}: avg MAE={avg_mae:6.3f}  params={params}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params
            best_residuals = fold_residuals

    if best_params is None:
        raise RuntimeError("[WFV] Failed to evaluate any hyperparameter configs.")

    sigma_wfv = float(np.std(best_residuals, ddof=1)) if best_residuals else 0.0
    print(f"\n[WFV] Best hyperparams by avg MAE={best_mae:6.3f}: {best_params}")
    print(f"[WFV] Sigma estimated from walk-forward residuals: {sigma_wfv:7.3f}")

    return best_params, sigma_wfv


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    features_csv = Path(args.features_csv)
    model_path = Path(args.model_path)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")

    print(f"Loading features from {features_csv} ...")
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in features CSV.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected a '{TARGET_COL}' target column in features CSV.")
    if "star_tier_pts" not in df.columns:
        raise ValueError("Expected 'star_tier_pts' column to build star-only model.")

    # ------------------------------------------------------------------
    # Filter to star-only rows
    # ------------------------------------------------------------------
    df = df[df["star_tier_pts"] >= 2].copy()
    if df.empty:
        raise RuntimeError("No rows with star_tier_pts >= 2; cannot train star-only model.")

    print(f"Filtered to star-tier rows (star_tier_pts >= 2): {len(df):,} rows.\n")

    seasons = sorted(df["season"].unique())
    print("Seasons in star-only dataset:", seasons)
    if not seasons:
        raise ValueError("No seasons found in star-only data.")

    min_season = int(min(seasons))
    max_season = int(max(seasons))

    # ------------------------------------------------------------------
    # Optionally add minutes_pred feature
    # ------------------------------------------------------------------
    feature_cols: List[str] = BASE_FEATURE_COLS.copy()
    if args.use_minutes_pred:
        ok = add_minutes_pred_feature(df, MINUTES_MODEL_PATH)
        if ok:
            feature_cols.append("minutes_pred")
            print("[INFO] Using 'minutes_pred' as an additional feature for star model.")
        else:
            print("[WARN] Proceeding without 'minutes_pred' feature.")

    # ------------------------------------------------------------------
    # Resolve train/val seasons
    # ------------------------------------------------------------------
    train_min = args.train_min_season if args.train_min_season is not None else min_season

    if args.train_max_season is not None:
        train_max = args.train_max_season
    else:
        if min_season == max_season:
            train_max = max_season
        else:
            sorted_seasons = sorted(seasons)
            train_max = int(sorted_seasons[-2])

    if args.val_min_season is not None:
        val_min = args.val_min_season
    else:
        val_min = train_max + 1 if train_max < max_season else train_max

    val_max = args.val_max_season if args.val_max_season is not None else max_season

    train_min = max(train_min, min_season)
    train_max = min(train_max, max_season)
    val_min = max(val_min, min_season)
    val_max = min(val_max, max_season)

    if train_min > train_max:
        raise ValueError(
            f"Invalid train season range: [{train_min}, {train_max}] "
            f"given star-only seasons [{min_season}, {max_season}]"
        )
    if val_min > val_max:
        raise ValueError(
            f"Invalid val season range: [{val_min}, {val_max}] "
            f"given star-only seasons [{min_season}, {max_season}]"
        )

    print("\n=== Season split configuration (star-only) ===")
    print(f"Train seasons: [{train_min}, {train_max}]")
    print(f"Val   seasons: [{val_min}, {val_max}]")

    train_mask = (df["season"] >= train_min) & (df["season"] <= train_max)
    val_mask = (df["season"] >= val_min) & (df["season"] <= val_max)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    print(f"\nTrain rows (stars): {len(df_train):,}")
    print(f"Val   rows (stars): {len(df_val):,}")

    if df_train.empty:
        raise RuntimeError("Training set is empty with the chosen star-only season range.")

    missing_feats = [c for c in feature_cols if c not in df_train.columns]
    if missing_feats:
        raise ValueError(f"Training data is missing expected feature columns: {missing_feats}")

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[TARGET_COL].to_numpy()

    if not df_val.empty:
        X_val = df_val[feature_cols].to_numpy()
        y_val = df_val[TARGET_COL].to_numpy()
    else:
        X_val = None
        y_val = None

    # ------------------------------------------------------------------
    # Optional walk-forward tuning
    # ------------------------------------------------------------------
    best_params: Dict
    sigma_wfv: Optional[float] = None

    if args.tune_hyperparams:
        best_params, sigma_wfv = walk_forward_tune(
            df=df,
            feature_cols=feature_cols,
            train_min=train_min,
            train_max=train_max,
            val_min=val_min,
            val_max=val_max,
            model_type=args.model_type,
            n_tune_iter=args.n_tune_iter,
        )
    else:
        if args.model_type == "histgb":
            best_params = {
                "max_iter": 600,
                "learning_rate": 0.05,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
            }
        else:  # xgboost
            best_params = {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            }

    print(f"\nTraining star-only model_type={args.model_type} with params={best_params}")

    model = build_model(args.model_type, best_params)

    # ------------------------------------------------------------------
    # Train final model
    # ------------------------------------------------------------------
    print("\nTraining star-only model...")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate on star-only holdout
    # ------------------------------------------------------------------
    if X_val is not None and y_val is not None and len(df_val) > 0:
        print(f"\nEvaluating STAR MODEL on holdout seasons ([{val_min}, {val_max}])...")
        y_pred_val = model.predict(X_val)

        mae_model = mean_absolute_error(y_val, y_pred_val)
        rmse_model = math.sqrt(mean_squared_error(y_val, y_pred_val))
        r2_model = r2_score(y_val, y_pred_val)

        print(f"STAR MODEL  - MAE:  {mae_model:6.3f}")
        print(f"STAR MODEL  - RMSE: {rmse_model:6.3f}")
        print(f"STAR MODEL  - R^2:  {r2_model:6.3f}")

        residuals = y_val - y_pred_val
        sigma_holdout = float(np.std(residuals, ddof=1))
        print(f"\nEstimated sigma (std of star-only holdout residuals): {sigma_holdout:6.3f}")

        if sigma_wfv is not None:
            print(f"[INFO] Using sigma from walk-forward validation instead of single holdout: {sigma_wfv:7.3f}")
            sigma = sigma_wfv
        else:
            sigma = sigma_holdout
    else:
        print("\nNo star-only validation set; estimating sigma from training residuals (NOT ideal).")
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        sigma = float(np.std(residuals, ddof=1))
        print(f"Sigma (star-only) estimated from train residuals: {sigma:6.3f}")

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    bundle = {
        "model": model,
        "sigma": sigma,
        "feature_cols": feature_cols,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved STAR regression model bundle to {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()