from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

from src.api.app import create_app


def test_beta_api_serves_precomputed_recommendations(tmp_path, monkeypatch):
    edges_path = tmp_path / "edges.csv"
    pd.DataFrame(
        [
            {
                "recommendation_id": "rec_123",
                "game_id": "game_123",
                "player": "Test Player",
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "market_key": "player_points",
                "best_side": "over",
                "prop_pts_line": 20.0,
                "over_odds_best": -110.0,
                "under_odds_best": -110.0,
                "best_edge": 0.12,
                "model_mean_pts": 23.4,
                "model_p_over": 0.71,
                "model_p_under": 0.29,
                "likely_range_low": 20.0,
                "likely_range_high": 27.0,
                "likely_range_confidence": 0.50,
                "most_likely_milestone": 20.0,
                "most_likely_milestone_probability": 0.71,
                "milestone_probabilities_json": (
                    '[{"threshold": 15, "probability": 0.88, "fair_odds": -733, "line_equivalent": 14.5}, '
                    '{"threshold": 20, "probability": 0.71, "fair_odds": -245, "line_equivalent": 19.5}, '
                    '{"threshold": 25, "probability": 0.39, "fair_odds": 156, "line_equivalent": 24.5}]'
                ),
                "generated_at_utc": "2026-01-10T12:00:00+00:00",
                "model_version": "target_pts:2026-01-10T12:00:00+00:00",
                "market_readiness_status": "production",
            }
        ]
    ).to_csv(edges_path, index=False)

    monkeypatch.setenv("NBA_BETTING_EDGES_PATH", str(edges_path))
    app = create_app()
    client = TestClient(app)

    list_response = client.get("/v1/recommendations")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["count"] == 1
    assert list_payload["items"][0]["id"] == "rec_123"
    assert list_payload["items"][0]["confidence"] == "medium"

    detail_response = client.get("/v1/recommendations/rec_123")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["market"] == "player_points"
    assert detail_payload["status"] == "production"
    assert detail_payload["most_likely_milestone"] == 20.0
    assert detail_payload["likely_range_low"] == 20.0
    assert detail_payload["likely_range_high"] == 27.0
    assert detail_payload["milestone_probabilities"][1]["threshold"] == 20.0
    assert any(reason["label"] == "Most likely milestone" for reason in detail_payload["reasons"])

    slate_response = client.get("/v1/slates/2026-01-10")
    assert slate_response.status_code == 200
    slate_payload = slate_response.json()
    assert slate_payload["date"] == "2026-01-10"
    assert len(slate_payload["games"]) == 1

    readiness_response = client.get("/v1/markets/readiness")
    assert readiness_response.status_code == 200
    readiness_payload = readiness_response.json()
    assert any(item["market"] == "player_points" for item in readiness_payload["items"])
