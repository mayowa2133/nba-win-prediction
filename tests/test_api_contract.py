from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

from src.api.app import create_app
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import (
    GameOddsSnapshotRecord,
    InjuryReportRecord,
    LineupProjectionRecord,
    RecommendationRecord,
)


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


def test_root_serves_dashboard_ui():
    app = create_app()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Crossover Insights" in response.text
    assert "/static/app.js" in response.text


def test_mobile_home_endpoint_serves_live_slate_shape(tmp_path, monkeypatch):
    edges_path = tmp_path / "mobile_edges.csv"
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
                "generated_at_utc": "2026-01-10T12:00:00+00:00",
                "model_version": "target_pts:2026-01-10T12:00:00+00:00",
                "market_readiness_status": "production",
                "commence_time": "2026-01-10T23:00:00Z",
            },
            {
                "recommendation_id": "rec_456",
                "game_id": "game_456",
                "player": "Other Player",
                "game_date": "2026-01-10",
                "home_team": "Miami Heat",
                "away_team": "New York Knicks",
                "market_key": "player_assists",
                "best_side": "under",
                "prop_pts_line": 7.5,
                "over_odds_best": -108.0,
                "under_odds_best": -112.0,
                "best_edge": 0.09,
                "model_mean_pts": 6.2,
                "model_p_over": 0.38,
                "model_p_under": 0.62,
                "generated_at_utc": "2026-01-10T12:00:00+00:00",
                "model_version": "target_ast:2026-01-10T12:00:00+00:00",
                "market_readiness_status": "production",
                "commence_time": "2026-01-11T00:30:00Z",
            },
        ]
    ).to_csv(edges_path, index=False)

    monkeypatch.setenv("NBA_BETTING_EDGES_PATH", str(edges_path))
    monkeypatch.delenv("NBA_BETTING_DATABASE_URL", raising=False)
    app = create_app()
    client = TestClient(app)

    response = client.get("/v1/mobile/home?date=2026-01-10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_date"] == "2026-01-10"
    assert payload["available_dates"] == ["2026-01-10"]
    assert len(payload["featured_recommendations"]) == 2
    assert len(payload["games"]) == 2
    assert payload["games"][0]["top_recommendation"]["id"] == "rec_123"
    assert payload["games"][0]["commence_time"] == "2026-01-10T23:00:00Z"
    assert payload["games"][0]["recommendation_count"] == 1
    assert payload["trending_parlays"][0]["recommendations"][0]["id"] == "rec_123"


def test_mobile_game_detail_and_trends_use_database_artifacts(tmp_path, monkeypatch):
    db_path = tmp_path / "mobile_api.db"
    database_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("NBA_BETTING_DATABASE_URL", database_url)
    monkeypatch.setenv("NBA_BETTING_EDGES_PATH", str(tmp_path / "missing.csv"))
    init_database(database_url)

    with session_scope(database_url) as session:
        session.add_all(
            [
                RecommendationRecord(
                    id="rec_live",
                    game_id="game_live",
                    player="Trae Young",
                    game_date="2026-01-10",
                    home_team="Atlanta Hawks",
                    away_team="Boston Celtics",
                    market="player_points",
                    selection="over",
                    sportsbook_line=26.5,
                    sportsbook_odds=-112.0,
                    fair_line=29.1,
                    fair_odds=-148,
                    edge=0.085,
                    selected_probability=0.63,
                    market_implied_probability=0.53,
                    confidence="high",
                    status="production",
                    model_version="live-model",
                    data_timestamp="2026-01-10T15:00:00Z",
                    published_line=26.5,
                    published_odds=-112.0,
                    published_at="2026-01-10T15:00:00Z",
                    lineup_context_json={
                        "home_projected_returning_starters": 4,
                        "away_projected_returning_starters": 5,
                        "projected_returning_starters": 9,
                        "projected_replacements": 1,
                    },
                    injury_context_json={"summary": "ATL: Trae Young questionable"},
                    reasons_json=[{"label": "Model vs line", "detail": "Projection still clears the market."}],
                ),
                RecommendationRecord(
                    id="rec_win",
                    game_id="game_old",
                    player="Jayson Tatum",
                    game_date="2026-01-08",
                    home_team="Boston Celtics",
                    away_team="Miami Heat",
                    market="player_points",
                    selection="over",
                    sportsbook_line=28.5,
                    sportsbook_odds=-110.0,
                    fair_line=30.0,
                    fair_odds=-140,
                    edge=0.06,
                    selected_probability=0.6,
                    market_implied_probability=0.52,
                    confidence="medium",
                    status="production",
                    model_version="live-model",
                    data_timestamp="2026-01-08T15:00:00Z",
                    published_line=28.5,
                    published_odds=-110.0,
                    published_at="2026-01-08T15:00:00Z",
                    result="win",
                    clv=0.04,
                    roi=0.91,
                    reasons_json=[{"label": "Win", "detail": "Settled winner."}],
                ),
                RecommendationRecord(
                    id="rec_loss",
                    game_id="game_older",
                    player="Jimmy Butler",
                    game_date="2026-01-07",
                    home_team="Miami Heat",
                    away_team="New York Knicks",
                    market="player_assists",
                    selection="under",
                    sportsbook_line=6.5,
                    sportsbook_odds=102.0,
                    fair_line=5.8,
                    fair_odds=-115,
                    edge=0.04,
                    selected_probability=0.57,
                    market_implied_probability=0.49,
                    confidence="low",
                    status="production",
                    model_version="live-model",
                    data_timestamp="2026-01-07T15:00:00Z",
                    published_line=6.5,
                    published_odds=102.0,
                    published_at="2026-01-07T15:00:00Z",
                    result="loss",
                    clv=-0.01,
                    roi=-1.0,
                    reasons_json=[{"label": "Loss", "detail": "Settled loser."}],
                ),
                GameOddsSnapshotRecord(
                    fixture_id="fixture_live",
                    game_id="game_live",
                    game_date="2026-01-10",
                    commence_time="2026-01-10T23:00:00Z",
                    home_team="Atlanta Hawks",
                    away_team="Boston Celtics",
                    market="player_points",
                    side="over",
                    sportsbook="Consensus",
                    line_value=26.5,
                    price=-112.0,
                    captured_at="2026-01-10T14:00:00Z",
                ),
                InjuryReportRecord(
                    game_id="game_live",
                    game_date="2026-01-10",
                    report_date="2026-01-10",
                    player_name="Trae Young",
                    team_abbrev="ATL",
                    report_status="Questionable",
                    normalized_status="questionable",
                    projected_availability="game-time decision",
                    raw_reason="Right ankle soreness",
                    reported_at="2026-01-10T11:00:00Z",
                ),
                InjuryReportRecord(
                    game_id="game_live",
                    game_date="2026-01-10",
                    report_date="2026-01-10",
                    player_name="Kristaps Porzingis",
                    team_abbrev="BOS",
                    report_status="Probable",
                    normalized_status="probable",
                    projected_availability="expected to play",
                    raw_reason="Illness",
                    reported_at="2026-01-10T11:10:00Z",
                ),
                LineupProjectionRecord(
                    projection_id="lineup_live",
                    game_id="game_live",
                    game_date="2026-01-10",
                    team_abbrev="ATL",
                    opponent_abbrev="BOS",
                    projected_starter="Trae Young",
                    projected_position="G",
                    starter_probability=0.82,
                    projection_reason="recent_starter",
                    injury_status="questionable",
                    projection_generated_at="2026-01-10T10:45:00Z",
                ),
                LineupProjectionRecord(
                    projection_id="lineup_live",
                    game_id="game_live",
                    game_date="2026-01-10",
                    team_abbrev="BOS",
                    opponent_abbrev="ATL",
                    projected_starter="Jayson Tatum",
                    projected_position="F",
                    starter_probability=0.98,
                    projection_reason="locked_in_starter",
                    injury_status="available",
                    projection_generated_at="2026-01-10T10:45:00Z",
                ),
            ]
        )

    app = create_app()
    client = TestClient(app)

    game_response = client.get("/v1/mobile/games/game_live")
    assert game_response.status_code == 200
    game_payload = game_response.json()
    assert game_payload["commence_time"] == "2026-01-10T23:00:00Z"
    assert game_payload["recommendations"][0]["id"] == "rec_live"
    assert len(game_payload["injuries"]) == 2
    assert game_payload["injuries"][0]["player_name"] == "Kristaps Porzingis"
    assert len(game_payload["lineup_summary"]) == 2
    assert game_payload["lineup_summary"][0]["team_abbrev"] == "BOS"

    trends_response = client.get("/v1/mobile/trends")
    assert trends_response.status_code == 200
    trends_payload = trends_response.json()
    assert trends_payload["wins"] == 1
    assert trends_payload["losses"] == 1
    assert trends_payload["pushes"] == 0
    assert trends_payload["hit_rate"] == 0.5
    assert len(trends_payload["recent_settlements"]) == 2
    assert trends_payload["recent_settlements"][0]["id"] == "rec_win"
    assert len(trends_payload["chart_points"]) == 2
    assert round(trends_payload["chart_points"][-1]["cumulative_roi"], 2) == -0.09
