from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.api.repository import RecommendationRepository
from src.warehouse.materialize import materialize_edges


def test_repository_reads_materialized_database(tmp_path):
    edges_path = tmp_path / "edges.csv"
    db_path = tmp_path / "nba_betting_beta.db"
    database_url = f"sqlite:///{db_path}"

    pd.DataFrame(
        [
            {
                "player": "Database Player",
                "game_date": "2026-01-12",
                "home_team": "Toronto Raptors",
                "away_team": "Miami Heat",
                "market_key": "player_points",
                "best_side": "over",
                "prop_pts_line": 18.5,
                "over_odds_best": -105.0,
                "under_odds_best": -115.0,
                "best_edge": 0.09,
                "model_mean_pts": 21.8,
                "model_p_over": 0.69,
                "model_p_under": 0.31,
                "model_version": "points:db-test",
            }
        ]
    ).to_csv(edges_path, index=False)

    scored_count, readiness_count = materialize_edges(edges_path, database_url=database_url)

    assert scored_count == 1
    assert readiness_count >= 1

    repository = RecommendationRepository(
        edges_path=Path(tmp_path / "missing.csv"),
        database_url=database_url,
    )
    items = repository.list_recommendations(date="2026-01-12")

    assert len(items) == 1
    assert items[0].player == "Database Player"
    assert items[0].market == "player_points"
    assert items[0].status == "production"


def test_repository_hides_historical_replay_rows_by_default(tmp_path):
    db_path = tmp_path / "nba_betting_beta.db"
    database_url = f"sqlite:///{db_path}"
    live_edges = tmp_path / "live_edges.csv"
    replay_edges = tmp_path / "replay_edges.csv"

    base_row = {
        "game_id": "game_origin",
        "game_date": "2026-01-12",
        "home_team": "Toronto Raptors",
        "away_team": "Miami Heat",
        "market_key": "player_points",
        "best_side": "over",
        "prop_pts_line": 18.5,
        "over_odds_best": -105.0,
        "under_odds_best": -115.0,
        "best_edge": 0.09,
        "model_mean_pts": 21.8,
        "model_p_over": 0.69,
        "model_p_under": 0.31,
        "model_version": "points:origin-test",
    }

    pd.DataFrame(
        [
            {
                **base_row,
                "recommendation_id": "rec_live",
                "player": "Live Player",
            }
        ]
    ).to_csv(live_edges, index=False)
    pd.DataFrame(
        [
            {
                **base_row,
                "recommendation_id": "rec_replay",
                "player": "Replay Player",
            }
        ]
    ).to_csv(replay_edges, index=False)

    materialize_edges(
        live_edges,
        database_url=database_url,
        recommendation_origin="live_daily",
    )
    materialize_edges(
        replay_edges,
        database_url=database_url,
        recommendation_origin="historical_replay",
    )

    repository = RecommendationRepository(
        edges_path=Path(tmp_path / "missing.csv"),
        database_url=database_url,
    )

    default_items = repository.list_recommendations(date="2026-01-12")
    replay_items = repository.list_recommendations(
        date="2026-01-12",
        origins=("historical_replay",),
    )

    assert [item.id for item in default_items] == ["rec_live"]
    assert [item.id for item in replay_items] == ["rec_replay"]
    assert repository.get_recommendation("rec_replay") is None
    assert repository.get_recommendation("rec_replay", origins=("historical_replay",)) is not None
