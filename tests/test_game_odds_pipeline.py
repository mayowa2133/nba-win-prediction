from __future__ import annotations

import pandas as pd

from src.data.oddspapi_game_odds import build_closing_lines_frame, build_game_odds_snapshot_frame


def test_game_odds_snapshot_normalization_and_closing_lines():
    payloads = [
        {
            "fixtureId": "fx1",
            "startTime": "2026-03-31T23:00:00Z",
            "participant1Name": "Boston Celtics",
            "participant2Name": "Atlanta Hawks",
            "bookmakerOdds": {
                "pinnacle": {
                    "markets": {
                        "1": {
                            "outcomes": {
                                "home_ml": {"players": {"0": [{"bookmakerOutcomeId": "home", "price": -120, "createdAt": "2026-03-31T17:00:00Z"}]}},
                                "away_ml": {"players": {"0": [{"bookmakerOutcomeId": "away", "price": 110, "createdAt": "2026-03-31T17:00:00Z"}]}},
                            }
                        },
                        "2": {
                            "outcomes": {
                                "home_spread": {"players": {"0": [
                                    {"bookmakerOutcomeId": "5.5/home", "price": -110, "createdAt": "2026-03-31T17:00:00Z"},
                                    {"bookmakerOutcomeId": "6.5/home", "price": -108, "createdAt": "2026-03-31T19:30:00Z"},
                                ]}},
                                "away_spread": {"players": {"0": [{"bookmakerOutcomeId": "6.5/away", "price": -112, "createdAt": "2026-03-31T19:30:00Z"}]}},
                            }
                        },
                        "3": {
                            "outcomes": {
                                "over_total": {"players": {"0": [{"bookmakerOutcomeId": "228.5/over", "price": -105, "createdAt": "2026-03-31T18:00:00Z"}]}},
                                "under_total": {"players": {"0": [{"bookmakerOutcomeId": "228.5/under", "price": -115, "createdAt": "2026-03-31T18:00:00Z"}]}},
                            }
                        },
                    }
                }
            },
        }
    ]
    market_catalog = {
        "1": {"marketName": "Moneyline"},
        "2": {"marketName": "Spread"},
        "3": {"marketName": "Total"},
    }

    snapshot_df = build_game_odds_snapshot_frame(
        payloads,
        market_catalog=market_catalog,
        source_url_prefix="https://api.oddspapi.io/v4/odds?fixtureId=",
        is_historical=False,
    )
    assert set(snapshot_df["market"]) == {"game_moneyline", "game_spread", "game_total"}
    assert set(snapshot_df[snapshot_df["market"] == "game_spread"]["line_value"]) == {5.5, 6.5}

    closing_df = build_closing_lines_frame(snapshot_df)
    spread_home = closing_df[(closing_df["market"] == "game_spread") & (closing_df["side"] == "home")].iloc[0]
    assert spread_home["line_value"] == 6.5
    assert spread_home["price"] == -108.0
