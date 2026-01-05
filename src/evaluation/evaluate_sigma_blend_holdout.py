#!/usr/bin/env python
"""
evaluate_sigma_blend_holdout.py

Holdout evaluation for sigma strategies, using the CURRENT mean prediction setup:
  mu = w_mu * mu_tiered + (1-w_mu) * mu_unified   (default w_mu=0.45)

We compare probability quality (Brier / LogLoss) on eval season (default 2025):
  1) Sigma-model + isotonic (current best)
  2) Blended tier/unified sigma + isotonic

Blended sigma:
  sigma_blend = w_sigma * sigma_tier + (1-w_sigma) * sigma_unified

We fit isotonic on TRAIN seasons only (<= train_max_season) and evaluate on EVAL season only.
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss


FEATURES_CSV_DEFAULT = Path("data/player_points_features_with_vegas.csv")
UNIFIED_MODEL_PATH = Path("models/points_regression.pkl")
TIER_MODEL_TEMPLATE = Path("models/points_regression_tier_{tier}.pkl")
SIGMA_MODEL_PATH = Path("models/points_sigma_model.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate sigma blending vs sigma-model on holdout.")
    p.add_argument("--features-csv", type=str, default=str(FEATURES_CSV_DEFAULT))
    p.add_argument("--train-max-season", type=int, default=2024)
    p.add_argument("--eval-season", type=int, default=2025)
    p.add_argument("--min-line", type=float, default=8.0)
    p.add_argument("--w-mu", type=float, default=0.45)
    p.add_argument("--w-sigma-grid-step", type=float, default=0.05)
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


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def ll(p: np.ndarray, y: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def load_bundle(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


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
    if "minutes_pred" in df.columns:
        return
    from build_points_regression import add_minutes_pred_feature

    ok = add_minutes_pred_feature(df, Path("models/minutes_regression.pkl"))
    if not ok:
        df["minutes_pred"] = 0.0


def fill_defaults(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            continue
        if c in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
            df[c] = 0.0
        elif c == "is_injured":
            df[c] = 0
        elif c == "days_since_last_dnp":
            df[c] = 999
        elif c == "dnp_count_last_10":
            df[c] = 0
        elif c == "has_prop_line":
            df[c] = 0.0
        elif "fg_pct" in c:
            df[c] = 0.45
        elif "3pt_pct" in c:
            df[c] = 0.35
        else:
            df[c] = 0.0


def main() -> None:
    args = parse_args()
    w_mu = max(0.0, min(1.0, float(args.w_mu)))
    step = float(args.w_sigma_grid_step)
    if step <= 0 or step > 1:
        raise ValueError("--w-sigma-grid-step must be in (0,1].")

    df = pd.read_csv(Path(args.features_csv), low_memory=False)
    if "target_pts" not in df.columns:
        raise ValueError("features CSV missing target_pts")
    if "star_tier_pts" not in df.columns:
        raise ValueError("features CSV missing star_tier_pts")

    # Synthetic line (same idea as calibrator): rounded pts_roll5
    df["line"] = df["pts_roll5"].round(0)
    df = df[df["line"] >= float(args.min_line)].copy()
    df = df.dropna(subset=["target_pts", "line"])

    train_mask = df["season"] <= int(args.train_max_season)
    eval_mask = df["season"] == int(args.eval_season)
    df_train = df[train_mask].copy()
    df_eval = df[eval_mask].copy()
    if df_train.empty or df_eval.empty:
        raise RuntimeError("Train/eval split resulted in empty dataset.")

    y_train = (df_train["target_pts"].to_numpy() > df_train["line"].to_numpy()).astype(int)
    y_eval = (df_eval["target_pts"].to_numpy() > df_eval["line"].to_numpy()).astype(int)

    unified = load_bundle(UNIFIED_MODEL_PATH)
    model_u = unified["model"]
    cols_u: List[str] = list(unified["feature_cols"])
    sigma_u = float(unified.get("sigma", 7.0))

    tier_models = {}
    sigma_tier = {}
    for t in [0, 1, 2, 3]:
        b = load_bundle(TIER_MODEL_TEMPLATE.with_name(f"points_regression_tier_{t}.pkl"))
        tier_models[t] = b
        sigma_tier[t] = float(b.get("sigma", sigma_u))

    # Ensure minutes_pred if required
    if "minutes_pred" in cols_u:
        ensure_minutes_pred(df_train)
        ensure_minutes_pred(df_eval)

    # Ensure has_prop_line if required
    if "has_prop_line" in cols_u:
        for d in (df_train, df_eval):
            if "has_prop_line" not in d.columns:
                if "prop_pts_line" in d.columns:
                    d["has_prop_line"] = (~d["prop_pts_line"].isna()).astype(float)
                else:
                    d["has_prop_line"] = 0.0

    fill_defaults(df_train, cols_u)
    fill_defaults(df_eval, cols_u)

    X_train_u = df_train[cols_u].to_numpy()
    X_eval_u = df_eval[cols_u].to_numpy()
    mu_train_u = model_u.predict(X_train_u).astype(float)
    mu_eval_u = model_u.predict(X_eval_u).astype(float)

    # Tiered mu computed per tier (feature cols exclude star_tier_pts)
    def mu_tiered(df_part: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df_part), dtype=float)
        tiers = df_part["star_tier_pts"].clip(0, 3).astype(int).to_numpy()
        for t in [0, 1, 2, 3]:
            idx = np.where(tiers == t)[0]
            if idx.size == 0:
                continue
            b = tier_models[t]
            cols_t = list(b["feature_cols"])
            fill_defaults(df_part, cols_t)
            Xt = df_part.iloc[idx][cols_t].to_numpy()
            out[idx] = b["model"].predict(Xt).astype(float)
        return out

    mu_train_t = mu_tiered(df_train)
    mu_eval_t = mu_tiered(df_eval)

    mu_train = w_mu * mu_train_t + (1.0 - w_mu) * mu_train_u
    mu_eval = w_mu * mu_eval_t + (1.0 - w_mu) * mu_eval_u

    # Sigma-model strategy (uses mu_hat)
    sigma_model, sigma_cols, sigma_cfg = load_sigma_bundle(SIGMA_MODEL_PATH)
    if sigma_model is None:
        raise RuntimeError("Sigma model not available; cannot run this comparison.")
    if sigma_cols is None:
        sigma_cols = cols_u + ["mu_hat"]

    def sigma_from_model(df_part: pd.DataFrame, mu_part: np.ndarray) -> np.ndarray:
        eps = float(sigma_cfg.get("eps", 1e-3))
        scale = float(sigma_cfg.get("sigma_scale", 1.0))
        use_log = bool(sigma_cfg.get("use_log_target", False))
        tmp = df_part.copy()
        tmp["mu_hat"] = mu_part
        fill_defaults(tmp, list(sigma_cols))
        Xs = tmp[list(sigma_cols)].to_numpy()
        raw = sigma_model.predict(Xs)
        pred = np.exp(raw) if use_log else raw
        out = np.maximum(pred * scale, eps)
        return np.clip(out, 1.0, 20.0)

    sigma_train_sm = sigma_from_model(df_train, mu_train)
    sigma_eval_sm = sigma_from_model(df_eval, mu_eval)

    p_train_sm = normal_p_over(mu_train, sigma_train_sm, df_train["line"].to_numpy())
    p_eval_sm = normal_p_over(mu_eval, sigma_eval_sm, df_eval["line"].to_numpy())
    iso_sm = IsotonicRegression(out_of_bounds="clip")
    iso_sm.fit(p_train_sm, y_train)
    p_eval_sm_cal = np.clip(iso_sm.predict(p_eval_sm), 0.0, 1.0)

    # Sigma blend grid (tier sigma vs unified sigma), then calibrate
    tiers_eval = df_eval["star_tier_pts"].clip(0, 3).astype(int).to_numpy()
    sigma_eval_tier = np.array([sigma_tier[int(t)] for t in tiers_eval], dtype=float)
    sigma_train_tier = np.array([sigma_tier[int(t)] for t in df_train["star_tier_pts"].clip(0, 3).astype(int).to_numpy()], dtype=float)

    best_blend = None
    for w_sig in np.arange(0.0, 1.0 + 1e-9, step):
        w_sig = float(w_sig)
        sigma_train_bl = np.clip(w_sig * sigma_train_tier + (1.0 - w_sig) * sigma_u, 1.0, 20.0)
        sigma_eval_bl = np.clip(w_sig * sigma_eval_tier + (1.0 - w_sig) * sigma_u, 1.0, 20.0)

        p_train_bl = normal_p_over(mu_train, sigma_train_bl, df_train["line"].to_numpy())
        p_eval_bl = normal_p_over(mu_eval, sigma_eval_bl, df_eval["line"].to_numpy())

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_train_bl, y_train)
        p_eval_bl_cal = np.clip(iso.predict(p_eval_bl), 0.0, 1.0)

        metrics = {
            "w_sigma": w_sig,
            "brier": brier(p_eval_bl_cal, y_eval),
            "logloss": ll(p_eval_bl_cal, y_eval),
        }
        if best_blend is None or metrics["brier"] < best_blend["brier"]:
            best_blend = metrics

    assert best_blend is not None

    print("=" * 70)
    print("SIGMA STRATEGY HOLDOUT (mu uses tiered+unified ensemble)")
    print("=" * 70)
    print(f"Train seasons: <= {int(args.train_max_season)}  (n={len(df_train):,})")
    print(f"Eval  season : {int(args.eval_season)}        (n={len(df_eval):,})")
    print(f"mu blend weight w_mu={w_mu:.2f}")
    print()
    print("Strategy                         |   Brier |  LogLoss")
    print("-" * 70)
    print(f"Sigma-model + isotonic          | {brier(p_eval_sm_cal, y_eval):7.4f} | {ll(p_eval_sm_cal, y_eval):8.4f}")
    print(f"Best sigma_blend + isotonic     | {best_blend['brier']:7.4f} | {best_blend['logloss']:8.4f}   (w_sigma={best_blend['w_sigma']:.2f})")
    print()
    print("Delta (blend - sigma-model):")
    print(f"  Brier:   {best_blend['brier'] - brier(p_eval_sm_cal, y_eval):+.4f}")
    print(f"  LogLoss: {best_blend['logloss'] - ll(p_eval_sm_cal, y_eval):+.4f}")


if __name__ == "__main__":
    main()


