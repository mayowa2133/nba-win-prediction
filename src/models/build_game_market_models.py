#!/usr/bin/env python
"""Train game-level NBA betting models for moneyline, spread, and total."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from src.utils.artifact_metadata import build_model_bundle_metadata, build_training_window
from src.utils.game_market_features import (
    GAME_MARKET_FEATURE_COLUMNS,
    build_historical_game_market_frame,
)


LOGS_CSV = Path("data/player_game_logs.csv")
INJURIES_CSV = Path("data/injury_reports.csv")
LINEUPS_CSV = Path("data/lineup_projections.csv")
STARTERS_CSV = Path("data/starter_history.csv")
MODELS_DIR = Path("models")
METRICS_CSV = Path("data/game_market_model_metrics.csv")


def calibration_error(probabilities: np.ndarray, actuals: np.ndarray, *, bins: int = 10) -> float:
    if len(probabilities) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    weight = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not mask.any():
            continue
        bucket_prob = float(probabilities[mask].mean())
        bucket_actual = float(actuals[mask].mean())
        bucket_weight = float(mask.mean())
        total += abs(bucket_prob - bucket_actual) * bucket_weight
        weight += bucket_weight
    if weight == 0:
        return math.nan
    return total / weight


def _sigma_from_residuals(actual: pd.Series, predicted: np.ndarray) -> float:
    residuals = pd.to_numeric(actual, errors="coerce").fillna(0.0).to_numpy() - np.asarray(predicted, dtype=float)
    sigma = float(np.nanstd(residuals))
    return sigma if sigma > 1e-6 else 1.0


def _regression_side_metrics(
    *,
    actual: pd.Series,
    predicted: np.ndarray,
    market_line: pd.Series,
    side: str,
    sigma: float,
) -> tuple[float, float]:
    line = pd.to_numeric(market_line, errors="coerce")
    mask = line.notna()
    if not mask.any():
        return math.nan, math.nan

    pred = np.asarray(predicted, dtype=float)[mask.to_numpy()]
    actual_values = pd.to_numeric(actual, errors="coerce").fillna(0.0).to_numpy()[mask.to_numpy()]
    line_values = line[mask].to_numpy(dtype=float)

    if side == "home":
        probs = 1.0 - norm.cdf(line_values, loc=pred, scale=sigma)
        outcomes = (actual_values > line_values).astype(float)
    else:
        probs = 1.0 - norm.cdf(line_values, loc=pred, scale=sigma)
        outcomes = (actual_values > line_values).astype(float)

    return float(brier_score_loss(outcomes, probs)), float(calibration_error(probs, outcomes))


def _prepare_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in feature_cols:
        if column not in frame.columns:
            frame[column] = 0.0
    frame = frame[list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame


def _season_bounds(df: pd.DataFrame, train_max: Optional[int], val_min: Optional[int]) -> tuple[int, int]:
    seasons = sorted(int(value) for value in pd.to_numeric(df["season_start"], errors="coerce").dropna().unique().tolist())
    if not seasons:
        raise RuntimeError("No season_start values available for game-market training data")
    resolved_train_max = train_max if train_max is not None else seasons[-2] if len(seasons) > 1 else seasons[-1]
    resolved_val_min = val_min if val_min is not None else seasons[-1]
    return resolved_train_max, resolved_val_min


def _filter_split(df: pd.DataFrame, *, train_max: int, val_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_start = pd.to_numeric(df["season_start"], errors="coerce")
    train_df = df[season_start <= train_max].copy()
    val_df = df[season_start >= val_min].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError(
            f"Could not build non-empty train/val split for game markets (train_max={train_max}, val_min={val_min})"
        )
    return train_df, val_df


def _train_moneyline_model(df: pd.DataFrame, *, feature_cols: list[str], train_max: int, val_min: int) -> tuple[dict, dict]:
    df = df[
        pd.to_numeric(df["market_home_ml_implied"], errors="coerce").notna()
        & pd.to_numeric(df["market_away_ml_implied"], errors="coerce").notna()
    ].copy()
    if df.empty:
        raise RuntimeError("No historical moneyline rows with populated market odds were available for training")

    train_df, val_df = _filter_split(df, train_max=train_max, val_min=val_min)
    X_train = _prepare_features(train_df, feature_cols)
    X_val = _prepare_features(val_df, feature_cols)
    y_train = train_df["home_win"].astype(int)
    y_val = val_df["home_win"].astype(int)

    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    baseline_probs = baseline.predict_proba(X_val)[:, 1]

    candidate = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=25)
    candidate.fit(X_train, y_train)
    candidate_probs = candidate.predict_proba(X_val)[:, 1]

    baseline_log = float(log_loss(y_val, baseline_probs))
    candidate_log = float(log_loss(y_val, candidate_probs))
    baseline_brier = float(brier_score_loss(y_val, baseline_probs))
    candidate_brier = float(brier_score_loss(y_val, candidate_probs))
    baseline_cal = float(calibration_error(baseline_probs, y_val.to_numpy(dtype=float)))
    candidate_cal = float(calibration_error(candidate_probs, y_val.to_numpy(dtype=float)))

    use_candidate = candidate_log < baseline_log or (math.isclose(candidate_log, baseline_log) and candidate_brier <= baseline_brier)
    chosen_model = candidate if use_candidate else baseline
    chosen_probs = candidate_probs if use_candidate else baseline_probs

    bundle = {
        "model": chosen_model,
        "feature_cols": feature_cols,
        "metadata": build_model_bundle_metadata(
            target="game_moneyline",
            training_window=build_training_window(train_max=train_max, val_min=val_min),
            readiness_status="experimental",
            model_type="histgb_classifier" if use_candidate else "logistic_regression",
            extra={
                "validation_metrics": {
                    "log_loss": candidate_log if use_candidate else baseline_log,
                    "brier": candidate_brier if use_candidate else baseline_brier,
                    "calibration_error": candidate_cal if use_candidate else baseline_cal,
                    "sample_size": int(len(val_df)),
                }
            },
        ),
    }
    metrics = {
        "market": "game_moneyline",
        "holdout_brier": candidate_brier if use_candidate else baseline_brier,
        "baseline_brier": baseline_brier,
        "holdout_log_loss": candidate_log if use_candidate else baseline_log,
        "baseline_log_loss": baseline_log,
        "calibration_error": candidate_cal if use_candidate else baseline_cal,
        "sample_size": int(len(val_df)),
        "holdout_mae": math.nan,
        "baseline_mae": math.nan,
        "vig_aware_roi": math.nan,
        "clv": math.nan,
        "trained": 1,
    }
    return bundle, metrics


def _train_regression_market(
    df: pd.DataFrame,
    *,
    market: str,
    target_col: str,
    line_col: str,
    feature_cols: list[str],
    train_max: int,
    val_min: int,
) -> tuple[dict, dict]:
    df = df[pd.to_numeric(df[line_col], errors="coerce").notna()].copy()
    if df.empty:
        raise RuntimeError(f"No historical rows with populated {line_col} were available for {market} training")

    train_df, val_df = _filter_split(df, train_max=train_max, val_min=val_min)
    X_train = _prepare_features(train_df, feature_cols)
    X_val = _prepare_features(val_df, feature_cols)
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0)
    y_val = pd.to_numeric(val_df[target_col], errors="coerce").fillna(0.0)

    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_val)

    candidate = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=25)
    candidate.fit(X_train, y_train)
    candidate_pred = candidate.predict(X_val)

    baseline_mae = float(mean_absolute_error(y_val, baseline_pred))
    candidate_mae = float(mean_absolute_error(y_val, candidate_pred))
    baseline_sigma = _sigma_from_residuals(y_val, baseline_pred)
    candidate_sigma = _sigma_from_residuals(y_val, candidate_pred)
    baseline_brier, baseline_cal = _regression_side_metrics(
        actual=y_val,
        predicted=baseline_pred,
        market_line=val_df[line_col],
        side="home" if market == "game_spread" else "over",
        sigma=baseline_sigma,
    )
    candidate_brier, candidate_cal = _regression_side_metrics(
        actual=y_val,
        predicted=candidate_pred,
        market_line=val_df[line_col],
        side="home" if market == "game_spread" else "over",
        sigma=candidate_sigma,
    )

    use_candidate = candidate_mae < baseline_mae or (math.isclose(candidate_mae, baseline_mae) and candidate_brier <= baseline_brier)
    chosen_model = candidate if use_candidate else baseline
    chosen_pred = candidate_pred if use_candidate else baseline_pred
    chosen_sigma = candidate_sigma if use_candidate else baseline_sigma
    chosen_brier = candidate_brier if use_candidate else baseline_brier
    chosen_cal = candidate_cal if use_candidate else baseline_cal
    chosen_mae = candidate_mae if use_candidate else baseline_mae

    bundle = {
        "model": chosen_model,
        "sigma": chosen_sigma,
        "feature_cols": feature_cols,
        "metadata": build_model_bundle_metadata(
            target=market,
            training_window=build_training_window(train_max=train_max, val_min=val_min),
            readiness_status="experimental",
            model_type="histgb_regressor" if use_candidate else "ridge_regression",
            extra={
                "validation_metrics": {
                    "mae": chosen_mae,
                    "brier": chosen_brier,
                    "calibration_error": chosen_cal,
                    "sample_size": int(len(val_df)),
                }
            },
        ),
    }
    metrics = {
        "market": market,
        "holdout_mae": chosen_mae,
        "baseline_mae": baseline_mae,
        "holdout_brier": chosen_brier,
        "baseline_brier": baseline_brier,
        "holdout_log_loss": math.nan,
        "baseline_log_loss": math.nan,
        "calibration_error": chosen_cal,
        "sample_size": int(len(val_df)),
        "vig_aware_roi": math.nan,
        "clv": math.nan,
        "trained": 1,
    }
    return bundle, metrics


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train game-market models from free NBA historical data.")
    parser.add_argument("--logs-csv", default=str(LOGS_CSV))
    parser.add_argument("--injuries-csv", default=str(INJURIES_CSV))
    parser.add_argument("--lineups-csv", default=str(LINEUPS_CSV))
    parser.add_argument("--starters-csv", default=str(STARTERS_CSV))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--metrics-out", default=str(METRICS_CSV))
    parser.add_argument("--train-max-season", type=int, default=None)
    parser.add_argument("--val-min-season", type=int, default=None)
    parser.add_argument(
        "--train-cutoff-date",
        default=None,
        help="Optional YYYY-MM-DD cutoff; only games strictly before this date are used for training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs = pd.read_csv(args.logs_csv)
    injuries = load_optional_csv(Path(args.injuries_csv))
    lineups = load_optional_csv(Path(args.lineups_csv))
    starters = load_optional_csv(Path(args.starters_csv))

    history = build_historical_game_market_frame(
        logs,
        injuries_df=injuries if not injuries.empty else None,
        lineup_df=lineups if not lineups.empty else None,
        starter_history_df=starters if not starters.empty else None,
    )
    if args.train_cutoff_date:
        history = history[pd.to_datetime(history["game_date"], errors="coerce") < pd.to_datetime(args.train_cutoff_date)]
    if history.empty:
        raise RuntimeError("Historical game-market feature frame is empty")

    train_max, val_min = _season_bounds(history, args.train_max_season, args.val_min_season)
    feature_cols = list(GAME_MARKET_FEATURE_COLUMNS)
    spread_coverage_rate = float(pd.to_numeric(history["market_home_spread_line"], errors="coerce").notna().mean())
    total_coverage_rate = float(pd.to_numeric(history["market_total_line"], errors="coerce").notna().mean())
    moneyline_coverage_rate = float(
        (
            pd.to_numeric(history["market_home_ml_implied"], errors="coerce").notna()
            & pd.to_numeric(history["market_away_ml_implied"], errors="coerce").notna()
        ).mean()
    )

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    market_trainers = [
        (
            "game_moneyline",
            moneyline_coverage_rate,
            lambda: _train_moneyline_model(history, feature_cols=feature_cols, train_max=train_max, val_min=val_min),
        ),
        (
            "game_spread",
            spread_coverage_rate,
            lambda: _train_regression_market(
                history,
                market="game_spread",
                target_col="home_margin_target",
                line_col="market_home_spread_line",
                feature_cols=feature_cols,
                train_max=train_max,
                val_min=val_min,
            ),
        ),
        (
            "game_total",
            total_coverage_rate,
            lambda: _train_regression_market(
                history,
                market="game_total",
                target_col="game_total_target",
                line_col="market_total_line",
                feature_cols=feature_cols,
                train_max=train_max,
                val_min=val_min,
            ),
        ),
    ]
    trained_any = False
    for market, coverage_rate, trainer in market_trainers:
        try:
            bundle, metrics = trainer()
        except RuntimeError as exc:
            print(f"[WARN] Skipping {market} training: {exc}")
            metrics = {
                "market": market,
                "holdout_brier": math.nan,
                "baseline_brier": math.nan,
                "holdout_log_loss": math.nan,
                "baseline_log_loss": math.nan,
                "holdout_mae": math.nan,
                "baseline_mae": math.nan,
                "calibration_error": math.nan,
                "sample_size": 0,
                "vig_aware_roi": math.nan,
                "clv": math.nan,
                "trained": 0,
                "skip_reason": str(exc),
            }
            metrics["source_coverage"] = coverage_rate
            metrics["moneyline_coverage_rate"] = moneyline_coverage_rate
            metrics_rows.append(metrics)
            continue

        trained_any = True
        metrics["source_coverage"] = coverage_rate
        metrics["moneyline_coverage_rate"] = moneyline_coverage_rate
        output = models_dir / f"{market}_model.pkl"
        with output.open("wb") as handle:
            pickle.dump(bundle, handle)
        metrics_rows.append(metrics)
        print(f"[INFO] Wrote {market} model bundle: {output}")

    if not trained_any:
        raise RuntimeError("No game-market models were trained successfully")

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"[INFO] Wrote game-market metrics: {metrics_path}")


if __name__ == "__main__":
    main()
