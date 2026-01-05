#!/usr/bin/env python
"""
evaluate_over_prob_holdout.py

Out-of-sample evaluation for OVER-probability modeling.

We evaluate 4 variants on a holdout season (default: 2025):
  A) Global sigma, raw normal approx
  B) Sigma model, raw normal approx
  C) Global sigma + isotonic calibrator fit on train seasons
  D) Sigma model + isotonic calibrator fit on train seasons

This lets us decide whether enabling the sigma model + calibrator in inference
actually improves probability quality (Brier / LogLoss) without touching MAE/R².
"""

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss


DEFAULT_FEATURES = Path("data/player_points_features_with_vegas.csv")
MODEL_PATH = Path("models/points_regression.pkl")
SIGMA_MODEL_PATH = Path("models/points_sigma_model.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate P(OVER) variants on a holdout season.")
    p.add_argument("--features-csv", type=str, default=str(DEFAULT_FEATURES))
    p.add_argument("--season-min", type=int, default=2023)
    p.add_argument("--train-max-season", type=int, default=2024)
    p.add_argument("--eval-season", type=int, default=2025)
    p.add_argument("--min-line", type=float, default=8.0)
    p.add_argument("--sigma-model-path", type=str, default=str(SIGMA_MODEL_PATH))
    return p.parse_args()


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_p_over(mu: np.ndarray, sigma: np.ndarray, line: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    line = np.asarray(line, dtype=float)
    sigma = np.where(sigma <= 0, 1e-6, sigma)
    z = (line - mu) / sigma
    vphi = np.vectorize(_phi)
    p_under = vphi(z)
    p_over = 1.0 - p_under
    return np.clip(p_over, 0.0, 1.0)


def load_points_model_bundle(path: Path) -> Dict:
    with open(path, "rb") as f:
        b = pickle.load(f)
    if not isinstance(b, dict) or "model" not in b or "feature_cols" not in b:
        raise ValueError(f"Unexpected model bundle format at {path}")
    return b


def load_sigma_bundle(path: Path) -> Tuple[Optional[object], Optional[List[str]], Dict]:
    if not path.exists():
        return None, None, {}
    with open(path, "rb") as f:
        b = pickle.load(f)
    if not isinstance(b, dict):
        return b, None, {"use_log_target": False, "eps": 1e-3, "sigma_scale": 1.0}
    m = b.get("model") or b.get("sigma_model") or b.get("regressor")
    cols = b.get("feature_cols")
    cfg = b.get("config") if isinstance(b.get("config"), dict) else {}
    cfg_out = {
        "use_log_target": bool(cfg.get("use_log_target", b.get("use_log_target", False))),
        "eps": float(cfg.get("eps", b.get("eps", 1e-3))),
        "sigma_scale": float(cfg.get("sigma_scale", b.get("sigma_scale", 1.0))),
    }
    return m, cols, cfg_out


def ensure_minutes_pred(df: pd.DataFrame) -> None:
    # Reuse the same helper as training
    from build_points_regression import add_minutes_pred_feature

    minutes_path = Path("models/minutes_regression.pkl")
    if "minutes_pred" in df.columns:
        return
    ok = add_minutes_pred_feature(df, minutes_path)
    if not ok:
        df["minutes_pred"] = 0.0


def fill_missing_features(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            continue
        if c == "has_prop_line":
            df[c] = 0.0
        elif c == "is_injured":
            df[c] = 0
        elif c == "days_since_last_dnp":
            df[c] = 999
        elif c == "dnp_count_last_10":
            df[c] = 0.0
        elif c in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
            df[c] = 0.0
        elif "fg_pct" in c:
            df[c] = 0.45
        elif "3pt_pct" in c:
            df[c] = 0.35
        else:
            df[c] = 0.0


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(Path(args.features_csv), low_memory=False)

    # Target column normalization
    if "pts" not in df.columns:
        if "target_pts" in df.columns:
            df["pts"] = df["target_pts"]
        else:
            raise ValueError("Expected pts or target_pts column in features CSV.")

    df = df[df["season"] >= int(args.season_min)].copy()
    df["line"] = df["pts_roll5"].round(0)
    df = df[df["line"] >= float(args.min_line)].copy()
    df = df.dropna(subset=["pts", "line"])

    train_mask = df["season"] <= int(args.train_max_season)
    eval_mask = df["season"] == int(args.eval_season)
    df_train = df[train_mask].copy()
    df_eval = df[eval_mask].copy()
    if df_train.empty or df_eval.empty:
        raise RuntimeError("Train/eval split resulted in empty dataset.")

    y_train = (df_train["pts"].to_numpy() > df_train["line"].to_numpy()).astype(int)
    y_eval = (df_eval["pts"].to_numpy() > df_eval["line"].to_numpy()).astype(int)

    bundle = load_points_model_bundle(MODEL_PATH)
    model = bundle["model"]
    global_sigma = float(bundle.get("sigma", 7.0))
    feature_cols: List[str] = list(bundle["feature_cols"])

    # Ensure minutes_pred if the model uses it
    if "minutes_pred" in feature_cols:
        ensure_minutes_pred(df_train)
        ensure_minutes_pred(df_eval)

    # Fill missing features in both splits
    fill_missing_features(df_train, feature_cols)
    fill_missing_features(df_eval, feature_cols)

    X_train = df_train[feature_cols].to_numpy()
    X_eval = df_eval[feature_cols].to_numpy()
    mu_train = model.predict(X_train)
    mu_eval = model.predict(X_eval)

    # Variant A: global sigma
    sigma_train_A = np.full_like(mu_train, fill_value=global_sigma, dtype=float)
    sigma_eval_A = np.full_like(mu_eval, fill_value=global_sigma, dtype=float)
    p_train_A = normal_p_over(mu_train, sigma_train_A, df_train["line"].to_numpy())
    p_eval_A = normal_p_over(mu_eval, sigma_eval_A, df_eval["line"].to_numpy())

    # Variant B: sigma model if available
    sigma_model, sigma_cols, sigma_cfg = load_sigma_bundle(Path(args.sigma_model_path))
    use_sigma_model = sigma_model is not None
    if sigma_cols is None:
        sigma_cols = feature_cols + ["mu_hat"]

    def sigma_from_model(df_part: pd.DataFrame, mu_part: np.ndarray) -> np.ndarray:
        eps = float(sigma_cfg.get("eps", 1e-3))
        scale = float(sigma_cfg.get("sigma_scale", 1.0))
        use_log = bool(sigma_cfg.get("use_log_target", False))
        df_sigma = df_part.copy()
        df_sigma["mu_hat"] = mu_part
        # If sigma model expects derived sigma features, compute them now
        try:
            from src.utils.sigma_features import SIGMA_DERIVED_COLS, add_sigma_derived_features_df
            if any(c in list(sigma_cols) for c in SIGMA_DERIVED_COLS):
                add_sigma_derived_features_df(df_sigma)
        except Exception:
            pass
        fill_missing_features(df_sigma, list(sigma_cols))
        Xs = df_sigma[list(sigma_cols)].to_numpy()
        raw = sigma_model.predict(Xs)
        pred = np.exp(raw) if use_log else raw
        out = np.maximum(pred * scale, eps)
        return np.clip(out, 1.0, 20.0)

    if use_sigma_model:
        sigma_train_B = sigma_from_model(df_train, mu_train)
        sigma_eval_B = sigma_from_model(df_eval, mu_eval)
        p_train_B = normal_p_over(mu_train, sigma_train_B, df_train["line"].to_numpy())
        p_eval_B = normal_p_over(mu_eval, sigma_eval_B, df_eval["line"].to_numpy())
    else:
        sigma_train_B = sigma_train_A
        sigma_eval_B = sigma_eval_A
        p_train_B = p_train_A
        p_eval_B = p_eval_A

    # Calibrators fit on TRAIN only, evaluated on EVAL only (out-of-sample)
    iso_A = IsotonicRegression(out_of_bounds="clip")
    iso_A.fit(p_train_A, y_train)
    p_eval_C = np.clip(iso_A.predict(p_eval_A), 0.0, 1.0)

    iso_B = IsotonicRegression(out_of_bounds="clip")
    iso_B.fit(p_train_B, y_train)
    p_eval_D = np.clip(iso_B.predict(p_eval_B), 0.0, 1.0)

    # LogLoss expects probs for both classes; we provide p_over
    def ll(p, y) -> float:
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))

    print("=" * 70)
    print("P(OVER) HOLDOUT EVALUATION")
    print("=" * 70)
    print(f"Train seasons: <= {int(args.train_max_season)}  (n={len(df_train):,})")
    print(f"Eval  season : {int(args.eval_season)}        (n={len(df_eval):,})")
    print(f"Sigma model available: {use_sigma_model}")
    print()

    rows = [
        ("A) global sigma, raw", p_eval_A),
        ("B) sigma model, raw", p_eval_B),
        ("C) global sigma + isotonic", p_eval_C),
        ("D) sigma model + isotonic", p_eval_D),
    ]
    print("Variant                         |   Brier |  LogLoss")
    print("-" * 70)
    for name, p in rows:
        print(f"{name:30s} | {brier(p, y_eval):7.4f} | {ll(p, y_eval):8.4f}")


if __name__ == "__main__":
    main()


