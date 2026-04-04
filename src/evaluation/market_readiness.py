"""Market-readiness gating based on holdout and betting-quality metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.contracts.market_readiness import DEFAULT_MARKET_READINESS


@dataclass(frozen=True)
class MarketMetrics:
    market: str
    holdout_mae: Optional[float] = None
    baseline_mae: Optional[float] = None
    holdout_brier: Optional[float] = None
    baseline_brier: Optional[float] = None
    holdout_log_loss: Optional[float] = None
    baseline_log_loss: Optional[float] = None
    calibration_error: Optional[float] = None
    vig_aware_roi: Optional[float] = None
    clv: Optional[float] = None
    sample_size: Optional[int] = None
    trained: Optional[int] = None
    live_clv_sample_size: Optional[int] = None
    evidence_mode: Optional[str] = None


def _metric_is_available(value: Optional[float]) -> bool:
    return value is not None and not pd.isna(value)


def calibration_error(probabilities, actuals, *, bins: int = 10) -> float:
    probabilities = pd.Series(probabilities, dtype=float)
    actuals = pd.Series(actuals, dtype=float)
    if probabilities.empty or actuals.empty:
        return float("nan")
    edges = pd.interval_range(start=0.0, end=1.0, periods=bins)
    bucketed = pd.cut(probabilities.clip(0.0, 1.0), bins=edges)
    total = 0.0
    weight = 0.0
    for _, group in pd.DataFrame({"prob": probabilities, "actual": actuals, "bucket": bucketed}).groupby("bucket", observed=False):
        if group.empty:
            continue
        bucket_weight = len(group) / len(probabilities)
        total += abs(float(group["prob"].mean()) - float(group["actual"].mean())) * bucket_weight
        weight += bucket_weight
    if weight == 0:
        return float("nan")
    return total / weight


def _beats_baseline(metrics: MarketMetrics) -> bool:
    mae_ok = (
        _metric_is_available(metrics.holdout_mae)
        and _metric_is_available(metrics.baseline_mae)
        and float(metrics.holdout_mae) < float(metrics.baseline_mae)
    )
    brier_ok = (
        _metric_is_available(metrics.holdout_brier)
        and _metric_is_available(metrics.baseline_brier)
        and float(metrics.holdout_brier) <= float(metrics.baseline_brier)
    )
    logloss_ok = (
        _metric_is_available(metrics.holdout_log_loss)
        and _metric_is_available(metrics.baseline_log_loss)
        and float(metrics.holdout_log_loss) <= float(metrics.baseline_log_loss)
    )

    has_regression_gate = _metric_is_available(metrics.holdout_mae) and _metric_is_available(metrics.baseline_mae)
    has_probability_gate = (
        _metric_is_available(metrics.holdout_brier)
        and _metric_is_available(metrics.baseline_brier)
    ) or (
        _metric_is_available(metrics.holdout_log_loss)
        and _metric_is_available(metrics.baseline_log_loss)
    )

    if has_regression_gate:
        if has_probability_gate:
            checks = []
            if _metric_is_available(metrics.holdout_brier) and _metric_is_available(metrics.baseline_brier):
                checks.append(brier_ok)
            if _metric_is_available(metrics.holdout_log_loss) and _metric_is_available(metrics.baseline_log_loss):
                checks.append(logloss_ok)
            return mae_ok and all(checks)
        return mae_ok

    if has_probability_gate:
        checks = []
        if _metric_is_available(metrics.holdout_brier) and _metric_is_available(metrics.baseline_brier):
            checks.append(brier_ok)
        if _metric_is_available(metrics.holdout_log_loss) and _metric_is_available(metrics.baseline_log_loss):
            checks.append(logloss_ok)
        return bool(checks) and all(checks)

    return False


def _calibration_ok(metrics: MarketMetrics) -> bool:
    if not _metric_is_available(metrics.calibration_error):
        return False
    return float(metrics.calibration_error) <= 0.05


def _minimum_sample_size(metrics: MarketMetrics) -> int:
    return 250 if str(metrics.market).startswith("game_") else 500


def _minimum_live_sample_size(metrics: MarketMetrics) -> int:
    return _minimum_sample_size(metrics)


def _production_ready(metrics: MarketMetrics) -> bool:
    if not _metric_is_available(metrics.sample_size):
        return False
    if int(float(metrics.sample_size)) < _minimum_sample_size(metrics):
        return False
    if not _metric_is_available(metrics.live_clv_sample_size):
        return False
    if int(float(metrics.live_clv_sample_size)) < _minimum_live_sample_size(metrics):
        return False
    if not _beats_baseline(metrics):
        return False
    if not _calibration_ok(metrics):
        return False
    if not _metric_is_available(metrics.vig_aware_roi) or float(metrics.vig_aware_roi) < 0:
        return False
    if not _metric_is_available(metrics.clv) or float(metrics.clv) <= 0:
        return False
    return True


def _production_blockers(metrics: MarketMetrics) -> list[str]:
    blockers: list[str] = []
    if not _metric_is_available(metrics.sample_size):
        blockers.append("missing settled sample size")
    elif int(float(metrics.sample_size)) < _minimum_sample_size(metrics):
        blockers.append(f"minimum sample size {int(float(metrics.sample_size))}/{_minimum_sample_size(metrics)}")

    if not _metric_is_available(metrics.live_clv_sample_size):
        blockers.append("missing live publish-time CLV sample")
    elif int(float(metrics.live_clv_sample_size)) < _minimum_live_sample_size(metrics):
        blockers.append(
            f"minimum live CLV sample size {int(float(metrics.live_clv_sample_size))}/{_minimum_live_sample_size(metrics)}"
        )

    if not _calibration_ok(metrics):
        blockers.append("calibration")
    if not _metric_is_available(metrics.vig_aware_roi) or float(metrics.vig_aware_roi) < 0:
        blockers.append("non-negative vig-aware ROI")
    if not _metric_is_available(metrics.clv) or float(metrics.clv) <= 0:
        blockers.append("positive CLV")
    return blockers


def evaluate_market_readiness(metrics: MarketMetrics) -> dict:
    default = DEFAULT_MARKET_READINESS.get(
        metrics.market,
        {
            "status": "experimental",
            "tier": "unknown",
            "label": "Experimental",
            "summary": "No explicit readiness rule has been defined for this market.",
        },
    )

    if _production_ready(metrics):
        return {
            "market": metrics.market,
            "status": "production",
            "tier": "beta_primary" if metrics.market == "player_points" else "beta_secondary",
            "label": "Production",
            "summary": (
                "Passes holdout baseline, calibration, vig-aware ROI, and CLV gates "
                f"(ROI={float(metrics.vig_aware_roi):+.3f}, CLV={float(metrics.clv):+.3f}, "
                f"n={int(float(metrics.sample_size or 0))})."
            ),
        }

    if _beats_baseline(metrics):
        blockers = _production_blockers(metrics)
        summary = (
            "Model clears historical baseline gates, but still misses production requirements: "
            + ", ".join(blockers)
            + "."
        )
        return {
            "market": metrics.market,
            "status": "experimental",
            "tier": default["tier"],
            "label": "Experimental",
            "summary": summary,
        }

    any_metrics = any(
        _metric_is_available(value)
        for value in (
            metrics.holdout_mae,
            metrics.holdout_brier,
            metrics.holdout_log_loss,
            metrics.calibration_error,
            metrics.vig_aware_roi,
            metrics.clv,
            metrics.sample_size,
        )
    ) or bool(metrics.trained)
    if any_metrics:
        if str(metrics.evidence_mode or "") == "historical_only":
            summary = (
                "Historical backtests exist, but the market has not yet accumulated enough live publish-time evidence "
                "to move beyond experimental."
            )
        else:
            summary = (
                "Model is trained and/or scored, but does not yet clear baseline and production gates."
                if default["status"] == "planned"
                else "Market remains experimental because it still fails one or more baseline or betting-quality gates."
            )
        return {
            "market": metrics.market,
            "status": "experimental",
            "tier": "beta_secondary" if str(metrics.market).startswith("game_") else default["tier"],
            "label": "Experimental",
            "summary": summary,
        }

    return {
        "market": metrics.market,
        "status": default["status"],
        "tier": default["tier"],
        "label": default["label"],
        "summary": default["summary"],
    }


def build_market_readiness_rows(metrics: Iterable[MarketMetrics]) -> List[dict]:
    return [evaluate_market_readiness(metric) for metric in metrics]


def load_metrics_csv(path: Path) -> List[MarketMetrics]:
    df = pd.read_csv(path)
    rows: List[MarketMetrics] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            MarketMetrics(
                market=str(record["market"]),
                holdout_mae=record.get("holdout_mae"),
                baseline_mae=record.get("baseline_mae"),
                holdout_brier=record.get("holdout_brier"),
                baseline_brier=record.get("baseline_brier"),
                holdout_log_loss=record.get("holdout_log_loss"),
                baseline_log_loss=record.get("baseline_log_loss"),
                calibration_error=record.get("calibration_error"),
                vig_aware_roi=record.get("vig_aware_roi"),
                clv=record.get("clv"),
                sample_size=record.get("sample_size"),
                trained=record.get("trained"),
                live_clv_sample_size=record.get("live_clv_sample_size"),
                evidence_mode=record.get("evidence_mode"),
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a market-readiness snapshot from metric inputs.")
    parser.add_argument("--metrics-csv", required=True, help="CSV with readiness metrics per market.")
    parser.add_argument(
        "--output",
        default="data/market_readiness.csv",
        help="Path to the readiness snapshot CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = load_metrics_csv(Path(args.metrics_csv))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(build_market_readiness_rows(metrics)).to_csv(output, index=False)
    print(f"[INFO] Wrote market readiness snapshot: {output}")


if __name__ == "__main__":
    main()
