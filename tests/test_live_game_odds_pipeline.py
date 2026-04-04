from __future__ import annotations

from datetime import date

from src.data.oddspapi_game_odds import build_closing_lines_frame
from src.data.the_odds_api_game_odds import build_game_odds_snapshot_frame


def test_the_odds_api_live_snapshot_normalization_and_closing_lines():
    payloads = [
        {
            "id": "evt_1",
            "commence_time": "2026-03-31T23:00:00Z",
            "home_team": "Boston Celtics",
            "away_team": "Atlanta Hawks",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-03-31T17:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-03-31T17:00:00Z",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -120},
                                {"name": "Atlanta Hawks", "price": 110},
                            ],
                        },
                        {
                            "key": "spreads",
                            "last_update": "2026-03-31T19:30:00Z",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -108, "point": -6.5},
                                {"name": "Atlanta Hawks", "price": -112, "point": 6.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "last_update": "2026-03-31T18:00:00Z",
                            "outcomes": [
                                {"name": "Over", "price": -105, "point": 228.5},
                                {"name": "Under", "price": -115, "point": 228.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]

    snapshot_df = build_game_odds_snapshot_frame(
        payloads,
        report_date=date(2026, 3, 31),
        source_url="https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
    )

    assert set(snapshot_df["market"]) == {"game_moneyline", "game_spread", "game_total"}
    assert set(snapshot_df[snapshot_df["market"] == "game_spread"]["line_value"]) == {6.5}

    closing_df = build_closing_lines_frame(snapshot_df)
    spread_home = closing_df[(closing_df["market"] == "game_spread") & (closing_df["side"] == "home")].iloc[0]
    assert spread_home["line_value"] == 6.5
    assert spread_home["price"] == -108.0
