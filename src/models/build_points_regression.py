#!/usr/bin/env python
"""
build_points_regression.py

Train a regression model to predict player points using the engineered features in
data/player_points_features.csv (or a merged version that includes props),
and compare it to a couple of simple baselines.

Outputs:
  - models/points_regression.pkl: {
        "model": regressor (HistGradientBoostingRegressor or XGBRegressor),
        "sigma": float (std of val residuals),
        "feature_cols": [list of feature names]
    }

New (2025-11):
  - Flexible season splitting via CLI
  - Uses richer feature set including player baselines, trend, volatility, rest,
    and matchup/env features (pace, DvP vs position).
  - Optional walk-forward hyperparameter tuning
  - Supports model_type: "histgb" (default) or "xgboost"
  - (Optional) Can include predicted minutes (minutes_pred) from minutes_regression.pkl

New (sample weights):
  - Optional star-based sample weights so the model focuses more on high-usage scorers.

New (market props):
  - Optional inclusion of prop-based features (prop_pts_line + odds) via --use-prop-features.
    Assumes features CSV already has columns from join_market_lines_to_features.py.

NEW (prop-only + prop-derived features):
  - --prop-only: restrict train/val to rows that have a prop line
  - --use-prop-derived-features: adds simple deltas like (prop_pts_line - pts_roll5)
  - --model-baseline-path: optional no-props model to compute (prop_pts_line - baseline_pred)
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

from src.utils.artifact_metadata import (
    build_model_bundle_metadata,
    build_training_window,
)

# Try to import xgboost lazily; it's only needed if model_type = "xgboost"
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None  # we will check this if the user requests xgboost


FEATURES_CSV = Path("data/player_points_features.csv")
MODEL_PATH = Path("models/points_regression.pkl")

# Optional upstream minutes model
MINUTES_MODEL_PATH = Path("models/minutes_regression.pkl")

# NEW: target column name from build_player_points_features.py
TARGET_COL = "target_pts"

# These must match what predict_points_over_line_next_game.py / star_best_bets_screener.py expects
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

    # NEW: rolling usage volume
    "usg_events_roll5",
    "usg_events_roll15",

    # opponent scoring environment
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",

    # team scoring margin environment (NEW)
    "team_margin_roll5",
    "team_margin_roll15",

    # rest / basic context
    "days_since_last_game",
    "is_home",

    # matchup/env features (pace + DvP vs position)
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

    # Vegas lines features (game-level context from sportsbooks)
    # These capture forward-looking information: game script, injuries priced in, pace expectations
    "vegas_game_total",   # O/U total points - predicts scoring environment
    "vegas_spread",       # spread for this team - predicts game script / blowout risk
    "vegas_abs_spread",   # absolute spread - blowout risk regardless of side
    
    # PHASE 4A: Game script features
    "blowout_prob",       # Probability of blowout (from spread)
    "is_likely_blowout",  # Binary: spread > 12
    "garbage_time_minutes_est",  # Estimated garbage time minutes
    "vegas_spread_abs_normalized",  # Normalized abs spread (0-1)
    
    # Injury/availability features (Phase 2)
    "is_injured",         # binary: 1 if injured/DNP, 0 otherwise
    "days_since_last_dnp", # days since last DNP (999 if never)
    "dnp_count_last_10",  # number of DNPs in last 10 games

    # Phase 4B: Lineup context / shorthandedness (team-level, derived from injuries)
    "teammate_out_count",
    "teammate_out_star_count",
    "teammate_out_usg15_sum",
    "teammate_out_min15_sum",
    "team_available_players",
    "is_team_shorthanded",
    
    # PHASE 4A: Player vs Opponent History
    "player_vs_opp_pts_avg_career",
    "player_vs_opp_pts_avg_last_5",
    "player_vs_opp_minutes_avg_career",
    "player_vs_opp_minutes_avg_last_5",
    "player_vs_opp_games_count",
    
    # PHASE 4A: Enhanced DvP (if available)
    "opp_fg_pct_allowed_vs_pos_roll5",
    "opp_fg_pct_allowed_vs_pos_roll15",
    "opp_3pt_pct_allowed_vs_pos_roll5",
    "opp_3pt_pct_allowed_vs_pos_roll15",
]

# Optional prop-based features (when using a CSV that has props joined)
PROP_BASE_COLS = [
    "prop_pts_line",
    "prop_over_odds_best",
    "prop_under_odds_best",
]
# We'll also add a binary indicator
PROP_INDICATOR_COL = "has_prop_line"

# Optional derived prop-vs-signal features
PROP_DERIVED_COLS = [
    "prop_minus_pts_roll5",
    "prop_minus_pts_roll15",
    "prop_minus_season_mean",
    "prop_minus_career_mean",
    "prop_minus_model_baseline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression model for player points with flexible "
            "season-based train/val splits and optional hyperparameter tuning."
        )
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(FEATURES_CSV),
        help="Path to player_points_features.csv (or merged with props). "
             "Default: data/player_points_features.csv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Where to save the regression model bundle (default: models/points_regression.pkl)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=TARGET_COL,
        help="Target column in features CSV (default: target_pts). "
             "Examples: target_reb, target_ast, target_fg3m",
    )
    parser.add_argument(
        "--val-preds-out",
        type=str,
        default="data/points_regression_val_preds.csv",
        help="Where to write validation predictions CSV (default: data/points_regression_val_preds.csv)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="histgb",
        choices=["histgb", "xgboost"],
        help=(
            "Which regressor to train: 'histgb' (HistGradientBoostingRegressor) "
            "or 'xgboost' (XGBRegressor). Default: histgb."
        ),
    )
    parser.add_argument(
        "--train-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include in training. "
             "Default: min season in the data.",
    )
    parser.add_argument(
        "--train-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include in training. "
             "Default: second-most-recent season in the data.",
    )
    parser.add_argument(
        "--val-min-season",
        type=int,
        default=None,
        help="Minimum season (start year) to include in validation. "
             "Default: train_max_season + 1.",
    )
    parser.add_argument(
        "--val-max-season",
        type=int,
        default=None,
        help="Maximum season (start year) to include in validation. "
             "Default: max season in the data.",
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
    parser.add_argument(
        "--use-star-weights",
        action="store_true",
        help=(
            "If set, use higher sample weights for higher 'star_tier_pts' players "
            "when training (focus the model on stars)."
        ),
    )
    parser.add_argument(
        "--use-prop-features",
        action="store_true",
        help=(
            "If set, include market prop features "
            "(prop_pts_line, prop_over_odds_best, prop_under_odds_best, has_prop_line) "
            "as inputs. Assumes these columns exist in the features CSV "
            "(e.g. from join_market_lines_to_features.py)."
        ),
    )

    # NEW: prop-only training and derived features
    parser.add_argument(
        "--prop-only",
        action="store_true",
        help="If set, restrict training/validation rows to those with a prop line (has_prop_line == 1).",
    )
    parser.add_argument(
        "--use-prop-derived-features",
        action="store_true",
        help=(
            "If set, add derived prop-vs-signal features like (prop_pts_line - pts_roll5). "
            "Requires --use-prop-features."
        ),
    )
    parser.add_argument(
        "--model-baseline-path",
        type=str,
        default="models/points_regression_no_props.pkl",
        help=(
            "Optional: path to a 'no-props' points model bundle used to compute "
            "prop_minus_model_baseline (prop_pts_line - baseline_pred)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Minutes prediction helper
# ---------------------------------------------------------------------


def add_minutes_pred_feature(df: pd.DataFrame, minutes_model_path: Path) -> bool:
    """
    Load the minutes regression model and add a 'minutes_pred' column to df.

    Returns True if the feature was successfully added, False otherwise.
    """
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
# Sample weight helper
# ---------------------------------------------------------------------


def compute_star_sample_weights(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Build star-based sample weights from 'star_tier_pts'.

    Mapping:
      tier 0 -> 0.5
      tier 1 -> 1.0
      tier 2 -> 2.0
      tier 3 -> 3.0

    Returns a Series aligned with df.index, or None if star_tier_pts missing.
    """
    if "star_tier_pts" not in df.columns:
        print("[WARN] 'star_tier_pts' column not found; cannot use star-based sample weights.")
        return None

    mapping = {0: 0.5, 1: 1.0, 2: 2.0, 3: 3.0}
    # Fill NaNs with 1 before mapping, then backfill any unmapped with 1.0
    tiers = df["star_tier_pts"].fillna(1)
    weights = tiers.map(mapping).fillna(1.0).astype(float)

    print("\n[STAR WEIGHTS] Using star-based sample weights:")
    print("[STAR WEIGHTS] Mapping: tier 0 -> 0.5, 1 -> 1.0, 2 -> 2.0, 3 -> 3.0")
    print("[STAR WEIGHTS] Weight summary:")
    print(weights.describe())

    # For sanity, show counts by tier if available
    try:
        tier_counts = df["star_tier_pts"].value_counts(dropna=False).sort_index()
        print("\n[STAR WEIGHTS] Star tier distribution:")
        print(tier_counts.to_string())
    except Exception:
        pass

    return weights


# ---------------------------------------------------------------------
# Model factories + param grids
# ---------------------------------------------------------------------


def make_histgb_model(params: Dict) -> HistGradientBoostingRegressor:
    """Create a HistGradientBoostingRegressor with given params plus fixed settings."""
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
    """Create an XGBRegressor with given params."""
    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install it with 'pip install xgboost' or use --model-type histgb."
        )

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",   # fast on CPUs
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
    """Grid of histgb hyperparameters; we will sample from this."""
    learning_rates = [0.03, 0.05, 0.08]
    max_leaf_nodes_list = [31, 63, 127]
    min_samples_leaf_list = [20, 50, 100]
    l2_regs = [0.0, 0.01, 0.1]
    # Include 600 here so walk-forward tuning can also try deeper training
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
    """Grid of xgboost hyperparameters; we will sample from this."""
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
                                    # regularization can be fixed or tuned later
                                    "reg_lambda": 1.0,
                                    "reg_alpha": 0.0,
                                }
                            )
    return configs


def build_model(model_type: str, params: Dict):
    """Factory that returns the correct model based on model_type."""
    if model_type == "histgb":
        return make_histgb_model(params)
    elif model_type == "xgboost":
        return make_xgb_model(params)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------
# Walk-forward tuning
# ---------------------------------------------------------------------


def walk_forward_tune(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_min: int,
    train_max: int,
    val_min: int,
    val_max: int,
    model_type: str,
    n_tune_iter: int,
    sample_weights: Optional[pd.Series] = None,
) -> Tuple[Dict, float]:
    """
    Walk-forward tuning over the validation seasons.

    For each candidate hyperparameter config:
      - For each validation season s in [val_min, val_max]:
          Train on seasons in [train_min, min(train_max, s-1)]
          Validate on season == s
      - Aggregate MAE across seasons

    If sample_weights is provided (aligned with df.index), it will be used as
    sample_weight for training folds.
    """
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
            # Train on seasons <= min(train_max, season_val - 1)
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
            y_train_f = df_train_f[target_col].to_numpy()
            X_val_f = df_val_f[feature_cols].to_numpy()
            y_val_f = df_val_f[target_col].to_numpy()

            model = build_model(model_type, params)

            if sample_weights is not None:
                w_train_f = sample_weights.loc[df_train_f.index].to_numpy()
                model.fit(X_train_f, y_train_f, sample_weight=w_train_f)
            else:
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


def main():
    args = parse_args()

    features_csv = Path(args.features_csv)
    model_path = Path(args.model_path)
    target_col = str(args.target_col)
    val_preds_out = Path(args.val_preds_out)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    print(f"Loading features from {features_csv} ...")
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in features CSV.")
    if target_col not in df.columns:
        raise ValueError(f"Expected a '{target_col}' target column in features CSV.")

    seasons = sorted(df["season"].unique())
    print("Seasons in dataset:", seasons)
    if not seasons:
        raise ValueError("No seasons found in data.")

    min_season = int(min(seasons))
    max_season = int(max(seasons))

    # ------------------------------------------------------------------
    # Handle missing Vegas features gracefully (for backward compatibility)
    # ------------------------------------------------------------------
    vegas_cols = ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]
    for col in vegas_cols:
        if col not in df.columns:
            print(f"[INFO] Vegas column '{col}' not in features; filling with 0.0")
            df[col] = 0.0

    # ------------------------------------------------------------------
    # Handle missing injury features gracefully (for backward compatibility)
    # ------------------------------------------------------------------
    injury_cols = ["is_injured", "days_since_last_dnp", "dnp_count_last_10"]
    for col in injury_cols:
        if col not in df.columns:
            if col == "is_injured":
                print(f"[INFO] Injury column '{col}' not in features; filling with 0 (healthy)")
                df[col] = 0
            elif col == "days_since_last_dnp":
                print(f"[INFO] Injury column '{col}' not in features; filling with 999 (never injured)")
                df[col] = 999
            else:
                print(f"[INFO] Injury column '{col}' not in features; filling with 0")
                df[col] = 0

    # ------------------------------------------------------------------
    # Phase 4B: Handle missing lineup context features gracefully
    # ------------------------------------------------------------------
    lineup_cols = [
        "teammate_out_count",
        "teammate_out_star_count",
        "teammate_out_usg15_sum",
        "teammate_out_min15_sum",
        "team_available_players",
        "is_team_shorthanded",
    ]
    for col in lineup_cols:
        if col not in df.columns:
            print(f"[INFO] Lineup context column '{col}' not in features; filling with 0.0")
            df[col] = 0.0

    # ------------------------------------------------------------------
    # Handle missing Phase 4A features gracefully (for backward compatibility)
    # ------------------------------------------------------------------
    # Game script features
    game_script_cols = ["blowout_prob", "is_likely_blowout", "garbage_time_minutes_est", "vegas_spread_abs_normalized"]
    for col in game_script_cols:
        if col not in df.columns:
            print(f"[INFO] Game script column '{col}' not in features; filling with 0.0")
            df[col] = 0.0
    
    # Player vs opponent history
    player_opp_cols = [
        "player_vs_opp_pts_avg_career", "player_vs_opp_pts_avg_last_5",
        "player_vs_opp_minutes_avg_career", "player_vs_opp_minutes_avg_last_5",
        "player_vs_opp_games_count"
    ]
    for col in player_opp_cols:
        if col not in df.columns:
            print(f"[INFO] Player vs opponent column '{col}' not in features; filling with 0.0")
            df[col] = 0.0
    
    # Enhanced DvP
    enhanced_dvp_cols = [
        "opp_fg_pct_allowed_vs_pos_roll5", "opp_fg_pct_allowed_vs_pos_roll15",
        "opp_3pt_pct_allowed_vs_pos_roll5", "opp_3pt_pct_allowed_vs_pos_roll15"
    ]
    for col in enhanced_dvp_cols:
        if col not in df.columns:
            print(f"[INFO] Enhanced DvP column '{col}' not in features; filling with league average")
            # Use league averages as defaults
            if "fg_pct" in col:
                df[col] = 0.45  # League average FG%
            elif "3pt_pct" in col:
                df[col] = 0.35  # League average 3PT%
            else:
                df[col] = 0.0

    # ------------------------------------------------------------------
    # Optionally add minutes_pred feature from minutes model
    # ------------------------------------------------------------------
    feature_cols: List[str] = BASE_FEATURE_COLS.copy()
    if args.use_minutes_pred:
        ok = add_minutes_pred_feature(df, MINUTES_MODEL_PATH)
        if ok:
            feature_cols.append("minutes_pred")
            print("[INFO] Using 'minutes_pred' as an additional feature for points model.")
        else:
            print("[WARN] Proceeding without 'minutes_pred' feature.")

    # ------------------------------------------------------------------
    # Optional prop-based features (market lines & odds)
    # ------------------------------------------------------------------
    prop_features_enabled = False
    if args.use_prop_features:
        missing_prop_cols = [c for c in PROP_BASE_COLS if c not in df.columns]
        if missing_prop_cols:
            print(
                f"[WARN] --use-prop-features was set, but the following prop columns "
                f"are missing in the features CSV: {missing_prop_cols}. "
                "Skipping prop features."
            )
        else:
            prop_features_enabled = True

            # Binary indicator: does this row have a prop line?
            df[PROP_INDICATOR_COL] = (~df["prop_pts_line"].isna()).astype(float)

            # Fill NaNs on prop numeric cols with 0.0 so tree models can handle them.
            # The model can distinguish "real" vs "fake" via has_prop_line.
            for col in PROP_BASE_COLS:
                df[col] = df[col].fillna(0.0)

            feature_cols.extend(PROP_BASE_COLS)
            feature_cols.append(PROP_INDICATOR_COL)

            print("[INFO] Using prop-based features as additional inputs:")
            print(f"       {PROP_BASE_COLS + [PROP_INDICATOR_COL]}")

    # ------------------------------------------------------------------
    # Optional prop-derived features
    # ------------------------------------------------------------------
    if args.use_prop_derived_features and not prop_features_enabled:
        print("[WARN] --use-prop-derived-features was set, but prop features are not enabled/available. Skipping.")
    elif args.use_prop_derived_features and prop_features_enabled:
        # Needed columns for deltas
        needed = [
            "prop_pts_line",
            "pts_roll5",
            "pts_roll15",
            "player_pts_season_mean",
            "player_pts_career_mean",
            PROP_INDICATOR_COL,
        ]
        missing_needed = [c for c in needed if c not in df.columns]
        if missing_needed:
            print(f"[WARN] Cannot compute prop-derived features; missing: {missing_needed}")
        else:
            df["prop_minus_pts_roll5"] = (df["prop_pts_line"] - df["pts_roll5"]) * df[PROP_INDICATOR_COL]
            df["prop_minus_pts_roll15"] = (df["prop_pts_line"] - df["pts_roll15"]) * df[PROP_INDICATOR_COL]
            df["prop_minus_season_mean"] = (df["prop_pts_line"] - df["player_pts_season_mean"]) * df[PROP_INDICATOR_COL]
            df["prop_minus_career_mean"] = (df["prop_pts_line"] - df["player_pts_career_mean"]) * df[PROP_INDICATOR_COL]

            # Optional: compare market line vs a frozen no-props model prediction
            df["prop_minus_model_baseline"] = 0.0
            baseline_path = Path(args.model_baseline_path)
            if baseline_path.exists():
                try:
                    with open(baseline_path, "rb") as f:
                        base_bundle = pickle.load(f)
                    base_model = base_bundle.get("model")
                    base_cols = base_bundle.get("feature_cols")

                    if base_model is not None and base_cols is not None:
                        missing_base = [c for c in base_cols if c not in df.columns]
                        if missing_base:
                            print(
                                f"[WARN] Baseline model features missing in df ({len(missing_base)} cols). "
                                "Skipping prop_minus_model_baseline."
                            )
                        else:
                            base_pred = base_model.predict(df[base_cols].to_numpy())
                            df["prop_minus_model_baseline"] = (df["prop_pts_line"] - base_pred) * df[PROP_INDICATOR_COL]
                            print("[INFO] Computed prop_minus_model_baseline using no-props model.")
                    else:
                        print("[WARN] Baseline bundle missing 'model' or 'feature_cols'; skipping prop_minus_model_baseline.")
                except Exception as e:
                    print(f"[WARN] Failed computing prop_minus_model_baseline: {e}; leaving as 0.0")
            else:
                print(f"[INFO] Baseline model not found at {baseline_path}; leaving prop_minus_model_baseline as 0.0")

            # Ensure no NaNs
            for c in PROP_DERIVED_COLS:
                if c in df.columns:
                    df[c] = df[c].fillna(0.0)

            # Add to feature columns
            for c in PROP_DERIVED_COLS:
                if c in df.columns:
                    feature_cols.append(c)

            print("[INFO] Using prop-derived features:", [c for c in PROP_DERIVED_COLS if c in df.columns])

    # ------------------------------------------------------------------
    # Optional star-based sample weights
    # ------------------------------------------------------------------
    sample_weights: Optional[pd.Series] = None
    if args.use_star_weights:
        sample_weights = compute_star_sample_weights(df)
        if sample_weights is None:
            print("[WARN] Falling back to uniform sample weights (1.0) since star_tier_pts was unavailable.")

    # ------------------------------------------------------------------
    # Resolve train/val season ranges from CLI + defaults
    # ------------------------------------------------------------------
    train_min = args.train_min_season if args.train_min_season is not None else min_season

    # Default train_max: second-most-recent season (reserve the latest for validation)
    if args.train_max_season is not None:
        train_max = args.train_max_season
    else:
        if min_season == max_season:
            train_max = max_season
        else:
            sorted_seasons = sorted(seasons)
            train_max = int(sorted_seasons[-2])

    # Default val_min: one season after train_max (if possible)
    if args.val_min_season is not None:
        val_min = args.val_min_season
    else:
        val_min = train_max + 1 if train_max < max_season else train_max

    # Default val_max: latest season in data
    val_max = args.val_max_season if args.val_max_season is not None else max_season

    # Clamp to data range
    train_min = max(train_min, min_season)
    train_max = min(train_max, max_season)
    val_min = max(val_min, min_season)
    val_max = min(val_max, max_season)

    if train_min > train_max:
        raise ValueError(
            f"Invalid train season range: [{train_min}, {train_max}] "
            f"given data seasons [{min_season}, {max_season}]"
        )
    if val_min > val_max:
        raise ValueError(
            f"Invalid val season range: [{val_min}, {val_max}] "
            "given data seasons "
            f"[{min_season}, {max_season}]"
        )

    print("\n=== Season split configuration ===")
    print(f"Train seasons: [{train_min}, {train_max}]")
    print(f"Val   seasons: [{val_min}, {val_max}]")

    # ------------------------------------------------------------------
    # Create train / val splits (for final training + evaluation)
    # ------------------------------------------------------------------
    train_mask = (df["season"] >= train_min) & (df["season"] <= train_max)
    val_mask = (df["season"] >= val_min) & (df["season"] <= val_max)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    # ------------------------------------------------------------------
    # Optional: restrict to rows with props (prop-only training)
    # ------------------------------------------------------------------
    if args.prop_only:
        if PROP_INDICATOR_COL not in df.columns:
            print("[WARN] --prop-only set but has_prop_line not present. Did you forget --use-prop-features?")
        else:
            before_tr = len(df_train)
            before_va = len(df_val)
            df_train = df_train[df_train[PROP_INDICATOR_COL] == 1.0].copy()
            df_val = df_val[df_val[PROP_INDICATOR_COL] == 1.0].copy()
            print(f"[INFO] Prop-only enabled. Train rows: {before_tr:,} -> {len(df_train):,}")
            print(f"[INFO] Prop-only enabled. Val rows:   {before_va:,} -> {len(df_val):,}")

    print(f"\nTrain rows: {len(df_train):,}")
    print(f"Val   rows: {len(df_val):,}")

    if df_train.empty:
        raise RuntimeError("Training set is empty with the chosen season range.")
    if df_val.empty:
        print(
            "WARNING: Validation set is empty with the chosen season range. "
            "Model will still train, but evaluation metrics will be skipped."
        )

    # Features / targets
    missing_feats = [c for c in feature_cols if c not in df_train.columns]
    if missing_feats:
        raise ValueError(f"Training data is missing expected feature columns: {missing_feats}")

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy()

    if not df_val.empty:
        X_val = df_val[feature_cols].to_numpy()
        y_val = df_val[target_col].to_numpy()
    else:
        X_val = None
        y_val = None

    # Build training sample weights for final fit if enabled
    if sample_weights is not None:
        w_train = sample_weights.loc[df_train.index].to_numpy()
    else:
        w_train = None

    # ------------------------------------------------------------------
    # Optional walk-forward hyperparameter tuning
    # ------------------------------------------------------------------
    best_params: Dict
    sigma_wfv: Optional[float] = None

    if args.tune_hyperparams:
        best_params, sigma_wfv = walk_forward_tune(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            train_min=train_min,
            train_max=train_max,
            val_min=val_min,
            val_max=val_max,
            model_type=args.model_type,
            n_tune_iter=args.n_tune_iter,
            sample_weights=sample_weights,
        )
    else:
        # Defaults based on tune_points_regression.py (best dev config on 2024)
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

    print(f"\nTraining model_type={args.model_type} with params={best_params}")

    model = build_model(args.model_type, best_params)

    # ------------------------------------------------------------------
    # Train final model on full train seasons
    # ------------------------------------------------------------------
    print("\nTraining model...")
    if w_train is not None:
        print("[INFO] Training with star-based sample weights.")
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate model on holdout seasons
    # ------------------------------------------------------------------
    if X_val is not None and y_val is not None and len(df_val) > 0:
        print(
            f"\nEvaluating MODEL on holdout seasons "
            f"([{val_min}, {val_max}])..."
        )
        y_pred_val = model.predict(X_val)

        mae_model = mean_absolute_error(y_val, y_pred_val)
        rmse_model = math.sqrt(mean_squared_error(y_val, y_pred_val))
        r2_model = r2_score(y_val, y_pred_val)

        print(f"MODEL  - MAE:  {mae_model:6.3f}")
        print(f"MODEL  - RMSE: {rmse_model:6.3f}")
        print(f"MODEL  - R^2:  {r2_model:6.3f}")

        residuals = y_val - y_pred_val
        sigma_holdout = float(np.std(residuals, ddof=1))
        print(f"\nEstimated sigma (std of holdout residuals): {sigma_holdout:6.3f}")

        # NEW: prop-only slice metrics (if available)
        if PROP_INDICATOR_COL in df_val.columns:
            prop_mask = df_val[PROP_INDICATOR_COL] == 1.0
            if prop_mask.any():
                y_val_prop = y_val[prop_mask.to_numpy()]
                y_pred_prop = y_pred_val[prop_mask.to_numpy()]
                mae_prop = mean_absolute_error(y_val_prop, y_pred_prop)
                rmse_prop = math.sqrt(mean_squared_error(y_val_prop, y_pred_prop))
                print(f"\n[PROP SLICE] Val rows w/ props: {int(prop_mask.sum()):,}")
                print(f"[PROP SLICE] MAE:  {mae_prop:6.3f}")
                print(f"[PROP SLICE] RMSE: {rmse_prop:6.3f}")

        # --------------------------------------------------------------
        # NEW: save detailed validation predictions & residuals
        # --------------------------------------------------------------
        df_val_export = df_val.copy()
        df_val_export["y_true"] = y_val
        df_val_export["y_pred"] = y_pred_val
        df_val_export["residual"] = df_val_export["y_true"] - df_val_export["y_pred"]

        # Helpful key columns to keep at the front if they exist
        cols_front = [
            c
            for c in [
                "season",
                "game_date",
                "game_id",
                "player_id",
                "player_name",
                "team_abbrev",
                "opp_abbrev",
            ]
            if c in df_val_export.columns
        ]
        other_cols = [c for c in df_val_export.columns if c not in cols_front]
        df_val_export = df_val_export[cols_front + other_cols]

        out_path = val_preds_out
        df_val_export.to_csv(out_path, index=False)
        print(f"\n[DEBUG] Saved validation predictions & residuals to {out_path}")

        # --------------------------------------------------------------
        # OPTIONAL: quick residual slices in the console
        # --------------------------------------------------------------
        df_val_export["abs_residual"] = df_val_export["residual"].abs()

        def print_group_stats(col: str, bins: List[float], label: Optional[str] = None) -> None:
            label = label or col
            tmp = df_val_export.copy()
            tmp["_bin"] = pd.cut(tmp[col], bins=bins, include_lowest=True)
            grp = tmp.groupby("_bin")["abs_residual"].agg(["count", "mean"])
            print(f"\n[RESIDUALS by {label}]")
            print(grp.to_string())

        # Example slices: tweak bins as needed
        if "pts_roll5" in df_val_export.columns:
            print_group_stats(
                "pts_roll5",
                [0, 10, 20, 30, 40, 60],
                label="pts_roll5 (recent scoring)",
            )

        if "minutes_roll5" in df_val_export.columns:
            print_group_stats(
                "minutes_roll5",
                [0, 10, 20, 30, 40, 60],
                label="minutes_roll5 (recent role)",
            )

        if "is_home" in df_val_export.columns:
            grp = df_val_export.groupby("is_home")["abs_residual"].agg(["count", "mean"])
            print("\n[RESIDUALS by home/away]")
            print(grp.to_string())

        if sigma_wfv is not None:
            print(f"[INFO] Using sigma from walk-forward validation instead of single holdout: {sigma_wfv:7.3f}")
            sigma = sigma_wfv
        else:
            sigma = sigma_holdout
    else:
        print("\nNo validation set; estimating sigma from training residuals (NOT ideal).")
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        sigma = float(np.std(residuals, ddof=1))
        print(f"Sigma estimated from train residuals: {sigma:6.3f}")

    # ------------------------------------------------------------------
    # Baseline 1: global mean (using train set only)
    # ------------------------------------------------------------------
    if y_val is not None and len(df_val) > 0:
        print("\nBaseline 1: GLOBAL MEAN (train-set average points)")
        global_mean = float(y_train.mean())
        y_pred_mean = np.full_like(y_val, fill_value=global_mean, dtype=float)

        mae_mean = mean_absolute_error(y_val, y_pred_mean)
        rmse_mean = math.sqrt(mean_squared_error(y_val, y_pred_mean))
        r2_mean = r2_score(y_val, y_pred_mean)

        print(f"MEAN   - MAE:  {mae_mean:6.3f}")
        print(f"MEAN   - RMSE: {rmse_mean:6.3f}")
        print(f"MEAN   - R^2:  {r2_mean:6.3f}")
    else:
        print("\nBaseline 1 (GLOBAL MEAN) skipped: no validation set.")

    # ------------------------------------------------------------------
    # Baseline 2: last-5-games average (stat_roll5 matching the target)
    # ------------------------------------------------------------------
    if y_val is not None and len(df_val) > 0:
        base_roll5_map = {
            "target_pts": "pts_roll5",
            "target_reb": "reb_roll5",
            "target_ast": "ast_roll5",
            "target_fg3m": "fg3m_roll5",
        }
        roll5_col = base_roll5_map.get(target_col)
        if roll5_col is None and target_col.startswith("target_"):
            roll5_col = target_col.replace("target_", "") + "_roll5"

        print(f"\nBaseline 2: LAST-5-GAMES AVERAGE ({roll5_col})")
        if roll5_col in df_val.columns:
            y_pred_roll5 = df_val[roll5_col].to_numpy()

            mae_roll5 = mean_absolute_error(y_val, y_pred_roll5)
            rmse_roll5 = math.sqrt(mean_squared_error(y_val, y_pred_roll5))
            r2_roll5 = r2_score(y_val, y_pred_roll5)

            print(f"ROLL5  - MAE:  {mae_roll5:6.3f}")
            print(f"ROLL5  - RMSE: {rmse_roll5:6.3f}")
            print(f"ROLL5  - R^2:  {r2_roll5:6.3f}")
        else:
            print(f"ROLL5  - skipped ({roll5_col} column missing)")
    else:
        print("\nBaseline 2 (ROLL5) skipped: no validation set.")

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    validation_metrics = {
        "holdout_mae": float(mae_model) if "mae_model" in locals() else None,
        "baseline_mae": float(mae_roll5) if "mae_roll5" in locals() else None,
        "baseline_mean_mae": float(mae_mean) if "mae_mean" in locals() else None,
        "sample_size": int(len(df_val)) if "df_val" in locals() else 0,
    }
    bundle = {
        "model": model,
        "sigma": sigma,
        "feature_cols": feature_cols,
        **build_model_bundle_metadata(
            target=target_col,
            training_window=build_training_window(
                train_min=train_min,
                train_max=train_max,
                val_min=val_min,
                val_max=val_max,
            ),
            readiness_status=None,
            model_type=args.model_type,
            extra={
                "uses_minutes_pred": bool(args.use_minutes_pred),
                "uses_prop_features": bool(args.use_prop_features),
                "uses_prop_derived_features": bool(args.use_prop_derived_features),
                "uses_star_weights": bool(args.use_star_weights),
                "sigma_source": "walk_forward" if sigma_wfv is not None else "holdout",
                "validation_metrics": validation_metrics,
            },
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved regression model bundle to {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
