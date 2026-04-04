#!/usr/bin/env python
"""
build_over_prob_calibrator.py

Fit a probability calibration model for P(OVER line) using the same setup as
backtest_over_line.py, and save it to models/over_prob_calibrator.pkl.

We:
  - Load data/player_points_features.csv
  - Restrict to seasons >= 2023
  - Define a synthetic line = pts_roll5 rounded (approximate "book" line)
  - Compute P(OVER line) using the regression model + normal approximation
    (with either a global sigma or a per-row sigma model if available)
  - Observe actual OVER/UNDER outcome for that synthetic line
  - Fit an IsotonicRegression mapping from raw P(OVER) -> calibrated P(OVER)

Usage:
  python build_over_prob_calibrator.py
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.utils.artifact_metadata import build_model_bundle_metadata, build_training_window

# Paths
FEATURES_CSV = Path("data/player_points_features.csv")
MODEL_PATH = Path("models/points_regression.pkl")
SIGMA_MODEL_PATH = Path("models/points_sigma_model.pkl")
CALIBRATOR_PATH = Path("models/over_prob_calibrator.pkl")

# These must match what you used when training points_regression.pkl
FEATURE_COLS_DEFAULT = [
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
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
    "days_since_last_game",
    "is_home",
]

SEASON_MIN = 2023
MIN_LINE = 8.0  # ignore tiny lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a probability calibration model for P(OVER line)."
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
        help="Path to points regression model bundle (default: models/points_regression.pkl)",
    )
    parser.add_argument(
        "--sigma-model-path",
        type=str,
        default=str(SIGMA_MODEL_PATH),
        help="Path to sigma model bundle (default: models/points_sigma_model.pkl)",
    )
    parser.add_argument(
        "--calibrator-path",
        type=str,
        default=str(CALIBRATOR_PATH),
        help="Where to save calibrator bundle (default: models/over_prob_calibrator.pkl)",
    )
    return parser.parse_args()


def normal_over_probs(mu, sigma, line):
    """
    Vectorized normal approximation: P(OVER line), P(UNDER line)
    for arrays mu, line, and scalar/array sigma.
    """
    mu_arr = np.asarray(mu, dtype=float)
    line_arr = np.asarray(line, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    # Handle scalar sigma separately to keep old behaviour
    if sigma_arr.shape == ():
        if sigma_arr <= 0:
            p_over = (mu_arr > line_arr).astype(float)
            p_under = 1.0 - p_over
            return p_over, p_under
        z = (line_arr - mu_arr) / float(sigma_arr)
    else:
        # Per-row sigma: clamp to small positive to avoid div-by-zero
        sigma_arr = np.where(sigma_arr <= 0, 1e-6, sigma_arr)
        z = (line_arr - mu_arr) / sigma_arr

    # Use math.erf via vectorization to avoid np.erf (not present)
    def _phi(zz):
        return 0.5 * (1.0 + math.erf(zz / math.sqrt(2.0)))

    vphi = np.vectorize(_phi)
    p_under = vphi(z)
    p_over = 1.0 - p_under
    return p_over, p_under


def load_regression_model(path: Path):
    """
    Load the regression bundle {model, sigma, feature_cols} from pickle.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle["model"]
        sigma = float(bundle.get("sigma", 7.0))
        feature_cols = bundle.get("feature_cols", FEATURE_COLS_DEFAULT)
    else:
        # Fallback if only model was pickled
        model = bundle
        sigma = 7.0
        feature_cols = FEATURE_COLS_DEFAULT

    return model, sigma, feature_cols


def load_sigma_model(path: Path, fallback_feature_cols):
    """
    Load the sigma model bundle from pickle, if present.

    Expected (new-style) bundle structure, e.g.:
      {
        "model": <regressor>,
        "feature_cols": [...],
        "config": {
            "use_log_target": True/False,
            "eps": 1e-3,
            "sigma_scale": 1.0,
        },
      }

    Also remains backwards compatible with older flat dicts.
    """
    if not path.exists():
        print(f"[INFO] Sigma model file not found at {path}; using global sigma only.")
        return None, None, {}

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        sigma_model = bundle.get("sigma_model") or bundle.get("model") or bundle.get("regressor")
        if sigma_model is None:
            # Last-resort: treat the whole bundle as the model
            sigma_model = bundle

        feature_cols = bundle.get("feature_cols", fallback_feature_cols)

        # New-style: config dict
        if "config" in bundle and isinstance(bundle["config"], dict):
            cfg = bundle["config"]
        else:
            # Backwards-compatible flat config
            cfg = {
                "use_log_target": bool(bundle.get("use_log_target", False)),
                "eps": float(bundle.get("eps", 1e-3)),
                "sigma_scale": float(bundle.get("sigma_scale", 1.0)),
            }
    else:
        sigma_model = bundle
        feature_cols = fallback_feature_cols
        cfg = {
            "use_log_target": False,
            "eps": 1e-3,
            "sigma_scale": 1.0,
        }

    print(f"[INFO] Loaded sigma model from {path}")
    print(f"[INFO] Sigma model feature_cols: {feature_cols}")
    print(f"[INFO] Sigma config: {cfg}")
    return sigma_model, feature_cols, cfg


def main():
    args = parse_args()
    features_csv = Path(args.features_csv)
    model_path = Path(args.model_path)
    sigma_model_path = Path(args.sigma_model_path)
    calibrator_path = Path(args.calibrator_path)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    print(f"Loading features from {features_csv} ...")
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    # NEW: handle schema where actual points are stored as 'target_pts'
    if "pts" not in df.columns:
        if "target_pts" in df.columns:
            print("[INFO] 'pts' column not found; using 'target_pts' as alias for calibration.")
            df["pts"] = df["target_pts"]
        else:
            raise ValueError("Expected 'pts' or 'target_pts' column in features CSV.")

    # Restrict to seasons >= SEASON_MIN (same as backtest)
    df = df[df["season"] >= SEASON_MIN].copy()
    print(f"Using seasons in [{SEASON_MIN}, 9999] -> {len(df):,} rows for calibration.")

    # Synthetic line: approximate naive "book" as last-5-games avg points, rounded
    df["line"] = df["pts_roll5"].round(0)  # integer rounding; adjust to .5 if you prefer

    # Focus on lines that are somewhat realistic (>= 8 points)
    df = df[df["line"] >= MIN_LINE].copy()
    print(f"After filtering to line >= {MIN_LINE:.1f}, we have {len(df):,} rows.")

    # Drop rows with missing actual points or line
    df = df.dropna(subset=["pts", "line"])
    print(f"After dropping rows with NaN pts/line: {len(df):,} rows.")

    if len(df) == 0:
        raise RuntimeError("No data remaining after filtering; cannot fit calibrator.")

    # Actual outcome: did player go over this synthetic line?
    y_over = (df["pts"].values > df["line"].values).astype(int)

    # Load regression model
    print(f"\nLoading regression model from {model_path} ...")
    model, global_sigma, model_feature_cols = load_regression_model(model_path)

    used_cols = model_feature_cols if model_feature_cols else FEATURE_COLS_DEFAULT
    print("Model uses feature columns:")
    for c in used_cols:
        print(f"  - {c}")

    # Handle missing Vegas features gracefully (for backward compatibility)
    vegas_cols = ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]
    for col in vegas_cols:
        if col in used_cols and col not in df.columns:
            print(f"[INFO] Vegas column '{col}' not in features; filling with 0.0")
            df[col] = 0.0

    # Handle missing prop features gracefully (for historical data without props)
    prop_cols = [
        "prop_pts_line",
        "prop_over_odds_best",
        "prop_under_odds_best",
        "has_prop_line",
        "prop_minus_pts_roll5",
        "prop_minus_pts_roll15",
        "prop_minus_season_mean",
        "prop_minus_career_mean",
        "prop_minus_model_baseline",
    ]
    for col in prop_cols:
        if col in used_cols and col not in df.columns:
            if col == "has_prop_line":
                print(f"[INFO] Prop column '{col}' not in features; filling with 0.0 (no prop)")
                df[col] = 0.0
            else:
                print(f"[INFO] Prop column '{col}' not in features; filling with 0.0")
                df[col] = 0.0

    # Handle missing injury features gracefully (for backward compatibility)
    injury_cols = ["is_injured", "days_since_last_dnp", "dnp_count_last_10"]
    for col in injury_cols:
        if col in used_cols and col not in df.columns:
            if col == "is_injured":
                print(f"[INFO] Injury column '{col}' not in features; filling with 0 (healthy)")
                df[col] = 0
            elif col == "days_since_last_dnp":
                print(f"[INFO] Injury column '{col}' not in features; filling with 999 (never injured)")
                df[col] = 999
            else:
                print(f"[INFO] Injury column '{col}' not in features; filling with 0")
                df[col] = 0

    # Handle minutes_pred feature (computed dynamically from minutes model)
    if "minutes_pred" in used_cols and "minutes_pred" not in df.columns:
        print("[INFO] Computing 'minutes_pred' feature using minutes model...")
        minutes_model_path = Path("models/minutes_regression.pkl")
        if minutes_model_path.exists():
            try:
                import pickle
                with open(minutes_model_path, "rb") as f:
                    minutes_bundle = pickle.load(f)
                minutes_model = minutes_bundle.get("model")
                minutes_feature_cols = minutes_bundle.get("feature_cols")
                
                if minutes_model is not None and minutes_feature_cols is not None:
                    missing_min = [c for c in minutes_feature_cols if c not in df.columns]
                    if not missing_min:
                        X_min = df[minutes_feature_cols].to_numpy()
                        df["minutes_pred"] = minutes_model.predict(X_min)
                        print("[INFO] Successfully computed 'minutes_pred' feature.")
                    else:
                        print(f"[WARN] Missing minutes feature columns: {missing_min}; filling minutes_pred with 0.0")
                        df["minutes_pred"] = 0.0
                else:
                    print("[WARN] Minutes model bundle invalid; filling minutes_pred with 0.0")
                    df["minutes_pred"] = 0.0
            except Exception as e:
                print(f"[WARN] Failed to load minutes model: {e}; filling minutes_pred with 0.0")
                df["minutes_pred"] = 0.0
        else:
            print(f"[WARN] Minutes model not found at {minutes_model_path}; filling minutes_pred with 0.0")
            df["minutes_pred"] = 0.0

    X = df[used_cols].to_numpy()
    print("\nPredicting expected points (mu) for each game...")
    mu = model.predict(X)

    # Expose mu_hat as a column so the sigma model can consume it if needed
    df["mu_hat"] = mu

    # Try to load sigma model for per-row sigma
    sigma_model, sigma_feature_cols, sigma_cfg = load_sigma_model(sigma_model_path, used_cols)

    # If sigma model expects derived sigma features, add them now
    try:
        from src.utils.sigma_features import SIGMA_DERIVED_COLS, add_sigma_derived_features_df
        if sigma_feature_cols is not None and any(c in sigma_feature_cols for c in SIGMA_DERIVED_COLS):
            add_sigma_derived_features_df(df)
    except Exception:
        # We'll fall back to dropping missing cols below.
        pass

    if sigma_model is not None:
        print("\nComputing per-row sigma_hat using sigma model...")

        # Drop any sigma features that are missing in df, to avoid KeyError
        missing_for_sigma = [c for c in sigma_feature_cols if c not in df.columns]
        if missing_for_sigma:
            print(f"[WARN] Dropping missing sigma feature columns: {missing_for_sigma}")
            sigma_feature_cols = [c for c in sigma_feature_cols if c in df.columns]

        if not sigma_feature_cols:
            raise RuntimeError(
                "[ERROR] After dropping missing sigma features, no columns remain "
                "for sigma model input. Check points_sigma_model.pkl configuration."
            )

        X_sigma = df[sigma_feature_cols].to_numpy()
        sigma_raw = sigma_model.predict(X_sigma)

        use_log_target = bool(sigma_cfg.get("use_log_target", False))
        eps = float(sigma_cfg.get("eps", 1e-3))
        scale = float(sigma_cfg.get("sigma_scale", 1.0))

        # In the current sigma model, the target is |residual| (i.e., sigma), or log(|residual|+eps)
        if use_log_target:
            sigma_pred = np.exp(sigma_raw)
        else:
            sigma_pred = sigma_raw

        sigma_arr = np.maximum(sigma_pred * float(scale), eps)

        print(
            "Sigma_hat stats (before clipping): "
            f"mean={sigma_arr.mean():.3f}, "
            f"median={np.median(sigma_arr):.3f}, "
            f"min={sigma_arr.min():.3f}, max={sigma_arr.max():.3f}"
        )

        # Clamp to a sane range to avoid insane tails
        sigma_arr = np.clip(sigma_arr, 1.0, 20.0)
        print(
            "Sigma_hat stats (after clipping to [1,20]): "
            f"mean={sigma_arr.mean():.3f}, "
            f"median={np.median(sigma_arr):.3f}"
        )
    else:
        # Fall back to global sigma
        sigma_arr = np.full_like(mu, fill_value=global_sigma, dtype=float)
        print(f"\n[INFO] Using global sigma={global_sigma:.3f} from regression bundle for all rows.")

    print("Computing raw P(OVER line) using normal approximation...")
    lines = df["line"].values
    p_over_raw, p_under_raw = normal_over_probs(mu, sigma_arr, lines)

    # Safety clamp
    p_over_raw = np.clip(p_over_raw, 0.0, 1.0)

    # Basic check: overall Brier before calibration
    brier_raw = np.mean((p_over_raw - y_over) ** 2)
    print(f"\nRaw probabilities Brier score (lower better): {brier_raw:.4f}")

    # Fit isotonic regression calibrator
    print("\nFitting IsotonicRegression calibrator (raw p_over -> calibrated p_over)...")
    iso = IsotonicRegression()
    iso.fit(p_over_raw, y_over)

    # Evaluate calibrated probabilities on the same set (for sanity)
    p_over_cal = iso.predict(p_over_raw)
    p_over_cal = np.clip(p_over_cal, 0.0, 1.0)
    brier_cal = np.mean((p_over_cal - y_over) ** 2)
    print(f"Calibrated probabilities Brier score:         {brier_cal:.4f}")

    # A small bin table to see the effect
    print("\nCalibration bins (using calibrated p_over):")
    bins = np.linspace(0.0, 1.0, 11)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_over_cal > lo) & (p_over_cal <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        avg_pred = float(p_over_cal[mask].mean())
        avg_actual = float(y_over[mask].mean())
        print(
            f"  ({lo:4.2f}, {hi:4.2f}] | n={cnt:5d} | "
            f"avg_pred={avg_pred:5.3f} | actual={avg_actual:5.3f}"
        )

    # Save calibrator
    bundle = {
        "calibrator": iso,
        "info": {
            "brier_raw": float(brier_raw),
            "brier_calibrated": float(brier_cal),
            "season_min": SEASON_MIN,
            "min_line": MIN_LINE,
            "uses_sigma_model": sigma_model is not None,
        },
        **build_model_bundle_metadata(
            target="target_pts_over_probability",
            training_window=build_training_window(
                train_min=SEASON_MIN,
                train_max=int(df_train["season"].max()) if not df_train.empty else None,
                val_min=int(df_eval["season"].min()) if not df_eval.empty else None,
                val_max=int(df_eval["season"].max()) if not df_eval.empty else None,
            ),
            readiness_status="experimental",
            model_type="isotonic_regression",
        ),
    }

    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibrator_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved calibration model to {calibrator_path}")
    print("Done.")


if __name__ == "__main__":
    main()
