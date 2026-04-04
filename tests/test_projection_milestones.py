from __future__ import annotations

from src.inference.scan_slate_with_model import build_projection_milestone_summary, norm_cdf


def test_build_projection_milestone_summary_points_market():
    mu = 22.4
    sigma = 5.0
    summary = build_projection_milestone_summary(
        market_key="player_points",
        mu=mu,
        sigma=sigma,
        line=22.5,
        probability_over_line=lambda target_line: 1.0 - norm_cdf((target_line - mu) / sigma),
    )

    assert summary["most_likely_milestone"] == 20.0
    assert round(float(summary["most_likely_milestone_probability"]), 3) == 0.719
    assert round(float(summary["likely_range_low"]), 2) == 19.03
    assert round(float(summary["likely_range_high"]), 2) == 25.77

    thresholds = [row["threshold"] for row in summary["milestone_probabilities"]]
    assert thresholds == [10.0, 15.0, 20.0, 25.0, 30.0]


def test_build_projection_milestone_summary_rebounds_market_uses_two_rebound_steps():
    mu = 10.3
    sigma = 3.1
    summary = build_projection_milestone_summary(
        market_key="player_rebounds",
        mu=mu,
        sigma=sigma,
        line=12.5,
        probability_over_line=lambda target_line: 1.0 - norm_cdf((target_line - mu) / sigma),
    )

    thresholds = [row["threshold"] for row in summary["milestone_probabilities"]]
    assert thresholds == [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    assert summary["most_likely_milestone"] == 10.0
