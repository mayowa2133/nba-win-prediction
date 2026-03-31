from __future__ import annotations

import pandas as pd

from src.evaluation.build_market_readiness_snapshot import build_readiness_rows
from src.evaluation.settle_recommendations import settle_recommendations_frame


def test_settlement_updates_results_and_readiness_metrics():
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": "rec_ml",
                "game_id": "g1",
                "player": "",
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "market": "game_moneyline",
                "selection": "home",
                "sportsbook_line": 0.0,
                "sportsbook_odds": -110.0,
                "published_line": 0.0,
                "published_odds": -110.0,
                "fair_line": 0.0,
                "selected_probability": 0.60,
            },
            {
                "recommendation_id": "rec_total",
                "game_id": "g1",
                "player": "",
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "market": "game_total",
                "selection": "over",
                "sportsbook_line": 205.5,
                "sportsbook_odds": -105.0,
                "published_line": 205.5,
                "published_odds": -105.0,
                "fair_line": 212.0,
                "selected_probability": 0.58,
            },
            {
                "recommendation_id": "rec_prop",
                "game_id": "g1",
                "player": "Trae Young",
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "market": "player_points",
                "selection": "over",
                "sportsbook_line": 25.5,
                "sportsbook_odds": -110.0,
                "published_line": 25.5,
                "published_odds": -110.0,
                "fair_line": 28.0,
                "selected_probability": 0.62,
            },
        ]
    )
    logs = pd.DataFrame(
        [
            {"game_id": "g1", "season": "2025-26", "game_date": "2026-01-10", "player_id": 1, "player_name": "Trae Young", "team_abbrev": "ATL", "opp_abbrev": "BOS", "is_home": 1, "minutes": 35, "pts": 28, "reb": 4, "ast": 11, "stl": 1, "blk": 0, "tov": 3, "fg3m": 4, "fg3a": 10, "fga": 20, "fgm": 10, "fta": 6, "ftm": 4, "pf": 1, "oreb": 1, "team_score": 110, "opp_score": 102, "spread_close": 4.5, "total_close": 206.5, "ml_team": -120, "ml_opp": 100},
            {"game_id": "g1", "season": "2025-26", "game_date": "2026-01-10", "player_id": 2, "player_name": "Jayson Tatum", "team_abbrev": "BOS", "opp_abbrev": "ATL", "is_home": 0, "minutes": 36, "pts": 24, "reb": 8, "ast": 5, "stl": 1, "blk": 1, "tov": 2, "fg3m": 3, "fg3a": 8, "fga": 18, "fgm": 9, "fta": 5, "ftm": 3, "pf": 2, "oreb": 1, "team_score": 102, "opp_score": 110, "spread_close": -4.5, "total_close": 206.5, "ml_team": 100, "ml_opp": -120},
        ]
    )
    closing = pd.DataFrame(
        [
            {"game_id": "g1", "market": "game_moneyline", "side": "home", "line_value": 0.0, "price": -130.0, "captured_at": "2026-01-10T23:55:00Z"},
            {"game_id": "g1", "market": "game_total", "side": "over", "line_value": 207.5, "price": -110.0, "captured_at": "2026-01-10T23:55:00Z"},
        ]
    )

    updated_df, settled_df = settle_recommendations_frame(
        recommendations,
        logs_df=logs,
        closing_lines_df=closing,
    )

    assert set(updated_df["result"].dropna()) == {"win"}
    total_row = updated_df[updated_df["recommendation_id"] == "rec_total"].iloc[0]
    assert total_row["closing_line"] == 207.5
    assert total_row["clv"] > 0
    prop_row = updated_df[updated_df["recommendation_id"] == "rec_prop"].iloc[0]
    assert prop_row["actual_value"] == 28.0

    training_metrics = pd.DataFrame(
        [
            {"market": "game_moneyline", "holdout_brier": 0.19, "baseline_brier": 0.21, "holdout_log_loss": 0.59, "baseline_log_loss": 0.64, "trained": 1},
            {"market": "game_total", "holdout_mae": 10.0, "baseline_mae": 11.0, "holdout_brier": 0.23, "baseline_brier": 0.24, "trained": 1},
        ]
    )
    readiness_rows = build_readiness_rows(updated_df, training_metrics)
    readiness_by_market = {row["market"]: row for row in readiness_rows}

    assert readiness_by_market["game_moneyline"]["status"] == "experimental"
    assert readiness_by_market["game_total"]["status"] == "experimental"
