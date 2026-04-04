from __future__ import annotations

import pandas as pd

from src.data.props_to_market_lines import aggregate_props


def test_aggregate_props_can_require_two_sided_markets():
    frame = pd.DataFrame(
        [
            {
                "sport_key": "basketball_nba",
                "event_id": "evt1",
                "commence_time": "2026-04-04T23:00:00Z",
                "home_team": "Philadelphia 76ers",
                "away_team": "Detroit Pistons",
                "market_key": "player_points",
                "player": "Tyrese Maxey",
                "line": 27.5,
                "side": "over",
                "odds": -112,
                "book": "draftkings",
                "source_provider": "scoresandodds",
            },
            {
                "sport_key": "basketball_nba",
                "event_id": "evt1",
                "commence_time": "2026-04-04T23:00:00Z",
                "home_team": "Philadelphia 76ers",
                "away_team": "Detroit Pistons",
                "market_key": "player_points",
                "player": "Tyrese Maxey",
                "line": 27.5,
                "side": "under",
                "odds": -108,
                "book": "fanduel",
                "source_provider": "covers",
            },
            {
                "sport_key": "basketball_nba",
                "event_id": "evt1",
                "commence_time": "2026-04-04T23:00:00Z",
                "home_team": "Philadelphia 76ers",
                "away_team": "Detroit Pistons",
                "market_key": "player_assists",
                "player": "Kelly Oubre Jr.",
                "line": 1.5,
                "side": "under",
                "odds": -140,
                "book": "draftkings",
                "source_provider": "covers",
            },
        ]
    )

    out = aggregate_props(frame, require_two_sided=True)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["player"] == "Tyrese Maxey"
    assert row["has_two_sided_market"] == 1
    assert row["over_odds_best"] == -112.0
    assert row["under_odds_best"] == -108.0
