#!/usr/bin/env python
"""Build market-readiness snapshots from settled recommendations and model metrics."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.contracts.market_readiness import DEFAULT_MARKET_READINESS
from src.evaluation.market_readiness import MarketMetrics, calibration_error, evaluate_market_readiness
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import MarketReadinessSnapshotRecord, RecommendationRecord


RECOMMENDATIONS_CSV = Path("data/settled_recommendations.csv")
TRAINING_METRICS_CSV = Path("data/game_market_model_metrics.csv")
OUTPUT_CSV = Path("data/market_readiness.csv")
MODELS_DIR = Path("models")


MODEL_BUNDLE_PATHS = {
    "player_points": "points_regression.pkl",
    "player_rebounds": "rebounds_regression.pkl",
    "player_assists": "assists_regression.pkl",
    "player_threes": "threes_regression.pkl",
    "game_moneyline": "game_moneyline_model.pkl",
    "game_spread": "game_spread_model.pkl",
    "game_total": "game_total_model.pkl",
}


def _load_recommendations(database_url: Optional[str], csv_path: Path) -> pd.DataFrame:
    if database_url:
        with session_scope(database_url) as session:
            rows = session.query(RecommendationRecord).all()
        return pd.DataFrame(
            [
                {
                    "recommendation_id": row.id,
                    "game_date": row.game_date,
                    "market": row.market,
                    "recommendation_origin": row.recommendation_origin,
                    "fair_line": row.fair_line,
                    "selected_probability": row.selected_probability,
                    "actual_value": row.actual_value,
                    "result": row.result,
                    "clv": row.clv,
                    "roi": row.roi,
                    "quote_source_provider": row.quote_source_provider,
                }
                for row in rows
            ]
        )
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _load_training_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _bundle_payload(bundle: dict) -> dict:
    if "metadata" in bundle and isinstance(bundle["metadata"], dict):
        return bundle["metadata"]
    return bundle


def _bundle_training_metrics(models_dir: Path) -> pd.DataFrame:
    rows = []
    for market, relative_path in MODEL_BUNDLE_PATHS.items():
        bundle_path = models_dir / relative_path
        if not bundle_path.exists():
            continue
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
        payload = _bundle_payload(bundle)
        validation = payload.get("validation_metrics") or {}
        if not validation:
            continue
        rows.append(
            {
                "market": market,
                "holdout_mae": validation.get("holdout_mae") or validation.get("mae"),
                "baseline_mae": validation.get("baseline_mae"),
                "holdout_brier": validation.get("holdout_brier") or validation.get("brier"),
                "baseline_brier": validation.get("baseline_brier"),
                "holdout_log_loss": validation.get("holdout_log_loss") or validation.get("log_loss"),
                "baseline_log_loss": validation.get("baseline_log_loss"),
                "calibration_error": validation.get("calibration_error"),
                "sample_size": validation.get("sample_size"),
                "trained": 1,
            }
        )
    return pd.DataFrame(rows)


def load_training_metrics(*, metrics_csv: Path, models_dir: Path) -> pd.DataFrame:
    frames = []
    bundle_metrics = _bundle_training_metrics(models_dir)
    if not bundle_metrics.empty:
        frames.append(bundle_metrics)
    csv_metrics = _load_training_metrics(metrics_csv)
    if not csv_metrics.empty:
        frames.append(csv_metrics)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("market").drop_duplicates(subset=["market"], keep="last")
    return combined


def _normalized_outcome(series: pd.Series) -> np.ndarray:
    mapping = {"win": 1.0, "loss": 0.0}
    return series.astype(str).str.lower().map(mapping).dropna().to_numpy(dtype=float)


def _window_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {"sample_size": int(len(df))}
    settled = df[df["result"].astype(str).str.lower().isin({"win", "loss", "push"})].copy()
    metrics["vig_aware_roi"] = float(pd.to_numeric(settled.get("roi"), errors="coerce").dropna().mean()) if not settled.empty else math.nan
    metrics["clv"] = float(pd.to_numeric(settled.get("clv"), errors="coerce").dropna().mean()) if not settled.empty else math.nan

    line_actual = pd.to_numeric(settled.get("actual_value"), errors="coerce")
    line_pred = pd.to_numeric(settled.get("fair_line"), errors="coerce")
    mask_line = line_actual.notna() & line_pred.notna()
    metrics["mae"] = float((line_actual[mask_line] - line_pred[mask_line]).abs().mean()) if mask_line.any() else math.nan

    prob_df = settled[settled["result"].astype(str).str.lower().isin({"win", "loss"})].copy()
    probs = pd.to_numeric(prob_df.get("selected_probability"), errors="coerce")
    mask_prob = probs.notna()
    if mask_prob.any():
        actual = prob_df.loc[mask_prob, "result"].astype(str).str.lower().map({"win": 1.0, "loss": 0.0}).to_numpy(dtype=float)
        pred = probs[mask_prob].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
        metrics["brier"] = float(brier_score_loss(actual, pred))
        metrics["log_loss"] = float(log_loss(actual, pred, labels=[0.0, 1.0]))
        metrics["calibration_error"] = float(calibration_error(pred, actual))
    else:
        metrics["brier"] = math.nan
        metrics["log_loss"] = math.nan
        metrics["calibration_error"] = math.nan
    return metrics


def _season_start(dt: pd.Timestamp) -> pd.Timestamp:
    year = dt.year if dt.month >= 10 else dt.year - 1
    return pd.Timestamp(year=year, month=10, day=1)


def build_readiness_rows(
    recommendations_df: pd.DataFrame,
    training_metrics_df: Optional[pd.DataFrame] = None,
) -> list[dict]:
    recommendations = recommendations_df.copy()
    if not recommendations.empty:
        recommendations["game_date"] = pd.to_datetime(recommendations["game_date"], errors="coerce")
    training_metrics = training_metrics_df.copy() if training_metrics_df is not None else pd.DataFrame()
    training_lookup = {
        str(row["market"]): row
        for _, row in training_metrics.iterrows()
    }
    markets = set(DEFAULT_MARKET_READINESS)
    markets.update(str(value) for value in recommendations.get("market", pd.Series(dtype=str)).dropna().unique().tolist())
    markets.update(str(value) for value in training_metrics.get("market", pd.Series(dtype=str)).dropna().unique().tolist())

    rows = []
    max_game_date = recommendations["game_date"].max() if not recommendations.empty else None
    for market in sorted(markets):
        settled = recommendations[recommendations["market"].astype(str) == market].copy() if not recommendations.empty else pd.DataFrame()
        full_window = _window_metrics(settled) if not settled.empty else {"sample_size": 0}
        trailing_30 = {"sample_size": 0}
        season_to_date = {"sample_size": 0}
        historical_sample_size = 0
        live_clv_sample_size = 0
        evidence_mode = "historical_only"
        live_quote_source = None
        if not settled.empty and "recommendation_origin" in settled.columns:
            historical_sample_size = int((settled["recommendation_origin"].astype(str) == "historical_replay").sum())
            live_clv_sample_size = int(
                (
                    (settled["recommendation_origin"].astype(str) == "live_daily")
                    & pd.to_numeric(settled.get("clv"), errors="coerce").notna()
                    & settled["result"].astype(str).str.lower().isin({"win", "loss", "push"})
                ).sum()
            )
            if live_clv_sample_size > 0:
                evidence_mode = "historical_plus_live"
            if "quote_source_provider" in settled.columns:
                live_sources = (
                    settled.loc[
                        settled["recommendation_origin"].astype(str) == "live_daily",
                        "quote_source_provider",
                    ]
                    .astype(str)
                    .replace("", pd.NA)
                    .dropna()
                )
                if not live_sources.empty:
                    live_quote_source = str(live_sources.value_counts().idxmax())
        if max_game_date is not None and not settled.empty and pd.notna(max_game_date):
            trailing_30_start = max_game_date - timedelta(days=30)
            trailing_30 = _window_metrics(settled[settled["game_date"] >= trailing_30_start].copy())
            season_to_date = _window_metrics(settled[settled["game_date"] >= _season_start(max_game_date)].copy())

        training_row = training_lookup.get(market)
        metrics = MarketMetrics(
            market=market,
            holdout_mae=training_row.get("holdout_mae") if training_row is not None else full_window.get("mae"),
            baseline_mae=training_row.get("baseline_mae") if training_row is not None else math.nan,
            holdout_brier=training_row.get("holdout_brier") if training_row is not None else full_window.get("brier"),
            baseline_brier=training_row.get("baseline_brier") if training_row is not None else math.nan,
            holdout_log_loss=training_row.get("holdout_log_loss") if training_row is not None else full_window.get("log_loss"),
            baseline_log_loss=training_row.get("baseline_log_loss") if training_row is not None else math.nan,
            calibration_error=full_window.get("calibration_error"),
            vig_aware_roi=full_window.get("vig_aware_roi"),
            clv=full_window.get("clv"),
            sample_size=full_window.get("sample_size"),
            trained=training_row.get("trained") if training_row is not None else int(full_window.get("sample_size", 0) > 0),
            live_clv_sample_size=live_clv_sample_size,
            evidence_mode=evidence_mode,
        )
        readiness = evaluate_market_readiness(metrics)
        rows.append(
            {
                **readiness,
                "metrics_json": json.dumps(
                    {
                        "full_window": full_window,
                        "trailing_30d": trailing_30,
                        "season_to_date": season_to_date,
                        "evidence_mode": evidence_mode,
                        "historical_sample_size": historical_sample_size,
                        "live_clv_sample_size": live_clv_sample_size,
                        "live_quote_source": live_quote_source,
                        "source_coverage": training_row.get("source_coverage") if training_row is not None else math.nan,
                        "moneyline_coverage_rate": training_row.get("moneyline_coverage_rate") if training_row is not None else math.nan,
                        "training_metrics": training_row.to_dict() if training_row is not None else {},
                    }
                ),
            }
        )
    return rows


def persist_readiness(rows: Iterable[dict], *, database_url: Optional[str]) -> int:
    if database_url is None:
        return 0
    init_database(database_url)
    with session_scope(database_url) as session:
        session.query(MarketReadinessSnapshotRecord).delete()
        count = 0
        for row in rows:
            session.add(
                MarketReadinessSnapshotRecord(
                    market=str(row["market"]),
                    status=str(row["status"]),
                    tier=str(row["tier"]),
                    label=str(row["label"]),
                    summary=str(row["summary"]),
                    metrics_json=json.loads(str(row["metrics_json"])),
                    as_of_timestamp=pd.Timestamp.utcnow().replace(microsecond=0).isoformat(),
                )
            )
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a market-readiness snapshot from settled recommendations.")
    parser.add_argument("--recommendations-csv", default=str(RECOMMENDATIONS_CSV))
    parser.add_argument("--training-metrics-csv", default=str(TRAINING_METRICS_CSV))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--database-url", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recommendations = _load_recommendations(args.database_url, Path(args.recommendations_csv))
    training_metrics = load_training_metrics(
        metrics_csv=Path(args.training_metrics_csv),
        models_dir=Path(args.models_dir),
    )
    rows = build_readiness_rows(recommendations, training_metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    persisted = persist_readiness(rows, database_url=args.database_url)
    print(f"[INFO] Wrote market readiness snapshot: {output}")
    if args.database_url:
        print(f"[INFO] Persisted {persisted} readiness row(s) to the warehouse")


if __name__ == "__main__":
    main()
