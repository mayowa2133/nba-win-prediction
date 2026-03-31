from __future__ import annotations

from src.evaluation.market_readiness import MarketMetrics, evaluate_market_readiness


def test_market_readiness_marks_market_production_when_all_gates_pass():
    metrics = MarketMetrics(
        market="player_points",
        holdout_mae=4.49,
        baseline_mae=4.87,
        holdout_brier=0.2200,
        baseline_brier=0.2231,
        calibration_error=0.03,
        vig_aware_roi=0.02,
        clv=0.01,
        sample_size=600,
    )

    readiness = evaluate_market_readiness(metrics)

    assert readiness["status"] == "production"
    assert readiness["label"] == "Production"


def test_market_readiness_keeps_market_experimental_when_roi_or_clv_fail():
    metrics = MarketMetrics(
        market="player_rebounds",
        holdout_mae=5.0,
        baseline_mae=5.3,
        holdout_brier=0.24,
        baseline_brier=0.25,
        calibration_error=0.03,
        vig_aware_roi=-0.01,
        clv=-0.02,
        sample_size=600,
    )

    readiness = evaluate_market_readiness(metrics)

    assert readiness["status"] == "experimental"
    assert "misses production requirements" in readiness["summary"]


def test_game_market_with_training_metrics_but_low_sample_stays_experimental():
    metrics = MarketMetrics(
        market="game_moneyline",
        holdout_brier=0.19,
        baseline_brier=0.21,
        holdout_log_loss=0.59,
        baseline_log_loss=0.63,
        calibration_error=0.03,
        vig_aware_roi=0.01,
        clv=0.01,
        sample_size=40,
        trained=1,
    )

    readiness = evaluate_market_readiness(metrics)

    assert readiness["status"] == "experimental"
    assert "minimum sample size" in readiness["summary"] or "production gates" in readiness["summary"]
