from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.public_page_game_odds import (
    build_espn_game_lines_frame,
    build_espn_game_odds_snapshot_frame,
    build_scoresandodds_game_lines_frame,
    build_scoresandodds_games_frame,
    build_scoresandodds_matchup_snapshot_frame,
)
from src.data.public_page_props import build_covers_prop_rows, build_scoresandodds_prop_rows, merge_prop_source_rows
from src.data.sportsgameodds import (
    build_sportsgameodds_game_lines_frame,
    build_sportsgameodds_game_odds_snapshot_frame,
    build_sportsgameodds_prop_rows,
)
from src.evaluation.build_market_readiness_snapshot import build_readiness_rows


FIXTURES = Path(__file__).parent / "fixtures" / "public_pages"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _sportsgameodds_events() -> list[dict]:
    return [
        {
            "eventID": "evt_heat_wizards",
            "status": {"startsAt": "2026-04-04T23:30:00Z"},
            "teams": {
                "home": {"names": {"long": "Miami Heat", "medium": "MIA", "short": "MIA"}},
                "away": {"names": {"long": "Washington Wizards", "medium": "WAS", "short": "WAS"}},
            },
            "odds": {
                "points-home-game-ml-home": {
                    "oddID": "points-home-game-ml-home",
                    "statID": "points",
                    "statEntityID": "home",
                    "periodID": "game",
                    "betTypeID": "ml",
                    "sideID": "home",
                    "bookOdds": "-2400",
                    "byBookmaker": {
                        "draftkings": {"odds": "-2500", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOdds": "-2200"},
                        "fanduel": {"odds": "-2300", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOdds": "-2100"},
                    },
                },
                "points-away-game-ml-away": {
                    "oddID": "points-away-game-ml-away",
                    "statID": "points",
                    "statEntityID": "away",
                    "periodID": "game",
                    "betTypeID": "ml",
                    "sideID": "away",
                    "bookOdds": "+1200",
                    "byBookmaker": {
                        "draftkings": {"odds": "+1250", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOdds": "+1100"},
                        "fanduel": {"odds": "+1150", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOdds": "+1050"},
                    },
                },
                "points-home-game-sp-home": {
                    "oddID": "points-home-game-sp-home",
                    "statID": "points",
                    "statEntityID": "home",
                    "periodID": "game",
                    "betTypeID": "sp",
                    "sideID": "home",
                    "bookSpread": "-18.5",
                    "bookOdds": "-105",
                    "byBookmaker": {
                        "draftkings": {"odds": "-105", "spread": "-18.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openSpread": "-17.5", "openOdds": "-110"},
                        "fanduel": {"odds": "-108", "spread": "-18.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openSpread": "-17.5", "openOdds": "-112"},
                    },
                },
                "points-away-game-sp-away": {
                    "oddID": "points-away-game-sp-away",
                    "statID": "points",
                    "statEntityID": "away",
                    "periodID": "game",
                    "betTypeID": "sp",
                    "sideID": "away",
                    "bookSpread": "+18.5",
                    "bookOdds": "-115",
                    "byBookmaker": {
                        "draftkings": {"odds": "-115", "spread": "+18.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openSpread": "+17.5", "openOdds": "-110"},
                        "fanduel": {"odds": "-112", "spread": "+18.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openSpread": "+17.5", "openOdds": "-108"},
                    },
                },
                "points-all-game-ou-over": {
                    "oddID": "points-all-game-ou-over",
                    "statID": "points",
                    "statEntityID": "all",
                    "periodID": "game",
                    "betTypeID": "ou",
                    "sideID": "over",
                    "bookOverUnder": "224.5",
                    "bookOdds": "-110",
                    "byBookmaker": {
                        "draftkings": {"odds": "-110", "overUnder": "224.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOverUnder": "223.5", "openOdds": "-108"},
                        "fanduel": {"odds": "-112", "overUnder": "224.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOverUnder": "223.5", "openOdds": "-110"},
                    },
                },
                "points-all-game-ou-under": {
                    "oddID": "points-all-game-ou-under",
                    "statID": "points",
                    "statEntityID": "all",
                    "periodID": "game",
                    "betTypeID": "ou",
                    "sideID": "under",
                    "bookOverUnder": "224.5",
                    "bookOdds": "-110",
                    "byBookmaker": {
                        "draftkings": {"odds": "-110", "overUnder": "224.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOverUnder": "223.5", "openOdds": "-112"},
                        "fanduel": {"odds": "-108", "overUnder": "224.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z", "openOverUnder": "223.5", "openOdds": "-110"},
                    },
                },
                "points-TYLER_HERRO_1_NBA-game-ou-over": {
                    "oddID": "points-TYLER_HERRO_1_NBA-game-ou-over",
                    "statID": "points",
                    "statEntityID": "TYLER_HERRO_1_NBA",
                    "periodID": "game",
                    "betTypeID": "ou",
                    "sideID": "over",
                    "byBookmaker": {
                        "draftkings": {"odds": "-115", "overUnder": "24.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z"},
                        "fanduel": {"odds": "-118", "overUnder": "24.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z"},
                    },
                },
                "points-TYLER_HERRO_1_NBA-game-ou-under": {
                    "oddID": "points-TYLER_HERRO_1_NBA-game-ou-under",
                    "statID": "points",
                    "statEntityID": "TYLER_HERRO_1_NBA",
                    "periodID": "game",
                    "betTypeID": "ou",
                    "sideID": "under",
                    "byBookmaker": {
                        "draftkings": {"odds": "-105", "overUnder": "24.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z"},
                        "fanduel": {"odds": "-102", "overUnder": "24.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z"},
                    },
                },
                "assists-TYLER_HERRO_1_NBA-game-ou-over": {
                    "oddID": "assists-TYLER_HERRO_1_NBA-game-ou-over",
                    "statID": "assists",
                    "statEntityID": "TYLER_HERRO_1_NBA",
                    "periodID": "game",
                    "betTypeID": "ou",
                    "sideID": "over",
                    "byBookmaker": {
                        "draftkings": {"odds": "-110", "overUnder": "5.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z"},
                    },
                },
                "assists-TYLER_HERRO_1_NBA-game-ou-under": {
                    "oddID": "assists-TYLER_HERRO_1_NBA-game-ou-under",
                    "statID": "assists",
                    "statEntityID": "TYLER_HERRO_1_NBA",
                    "periodID": "game",
                    "betTypeID": "ou",
                    "sideID": "under",
                    "byBookmaker": {
                        "draftkings": {"odds": "-120", "overUnder": "5.5", "available": True, "lastUpdatedAt": "2026-04-04T14:00:00Z"},
                    },
                },
            },
        }
    ]


def test_scoresandodds_game_pages_normalize_current_lines_and_matchup_rows():
    games_df = build_scoresandodds_games_frame(
        _fixture("scoresandodds_nba.html"),
        report_date=date(2026, 4, 4),
        page_snapshot_at="2026-04-04T14:00:00Z",
    )

    assert len(games_df) == 1
    row = games_df.iloc[0]
    assert row["home_team"] == "Miami Heat"
    assert row["away_team"] == "Washington Wizards"
    assert row["vegas_home_spread"] == -18.5
    assert row["current_home_moneyline"] == -2400.0

    game_lines = build_scoresandodds_game_lines_frame(games_df)
    assert game_lines.iloc[0]["open_home_spread"] == -17.5
    assert game_lines.iloc[0]["open_game_total"] == 247.5

    snapshots = build_scoresandodds_matchup_snapshot_frame(
        _fixture("scoresandodds_matchup.html"),
        game_row=row,
        source_url="https://www.scoresandodds.com/nba/heat-vs-wizards",
        page_snapshot_at="2026-04-04T14:00:00Z",
    )

    assert set(snapshots["market"]) == {"game_moneyline", "game_spread", "game_total"}
    assert set(snapshots["sportsbook"]) == {"draftkings", "fanduel"}
    assert set(snapshots["source_provider"]) == {"scoresandodds"}
    home_spread = snapshots[(snapshots["market"] == "game_spread") & (snapshots["side"] == "home") & (snapshots["sportsbook"] == "draftkings")].iloc[0]
    assert home_spread["line_value"] == 18.5
    assert home_spread["price"] == -105.0


def test_scoresandodds_matchup_props_produce_two_sided_rows_with_provenance():
    game_row = build_scoresandodds_games_frame(
        _fixture("scoresandodds_nba.html"),
        report_date=date(2026, 4, 4),
        page_snapshot_at="2026-04-04T14:00:00Z",
    ).iloc[0]
    rows = build_scoresandodds_prop_rows(
        _fixture("scoresandodds_matchup.html"),
        game_row=game_row,
        source_url="https://www.scoresandodds.com/nba/heat-vs-wizards",
        allowed_markets={"player_points", "player_assists"},
        page_snapshot_at="2026-04-04T14:00:00Z",
    )

    frame = pd.DataFrame(rows)
    assert set(frame["market_key"]) == {"player_points", "player_assists"}
    assert set(frame["side"]) == {"over", "under"}
    assert set(frame["book"]) == {"draftkings", "fanduel"}
    assert set(frame["source_provider"]) == {"scoresandodds"}
    assert set(frame[frame["player"] == "Tyler Herro"]["line"]) == {24.5}


def test_espn_embedded_payload_normalizes_current_game_odds_and_lines():
    html = _fixture("espn_nba_odds.html")
    snapshot_df = build_espn_game_odds_snapshot_frame(
        html,
        report_date=date(2026, 4, 4),
        page_snapshot_at="2026-04-04T14:05:00Z",
    )
    game_lines_df = build_espn_game_lines_frame(
        html,
        report_date=date(2026, 4, 4),
        page_snapshot_at="2026-04-04T14:05:00Z",
    )

    assert len(snapshot_df) == 6
    assert set(snapshot_df["source_provider"]) == {"espn"}
    assert set(snapshot_df["sportsbook"]) == {"draftkings"}
    assert game_lines_df.iloc[0]["vegas_home_spread"] == -18.5
    assert game_lines_df.iloc[0]["current_away_moneyline"] == 1200.0


def test_sportsgameodds_game_frames_normalize_snapshot_rows_and_consensus_lines():
    events = _sportsgameodds_events()
    snapshot_df = build_sportsgameodds_game_odds_snapshot_frame(
        events,
        report_date=date(2026, 4, 4),
        page_snapshot_at="2026-04-04T14:00:00Z",
    )
    game_lines_df = build_sportsgameodds_game_lines_frame(
        events,
        report_date=date(2026, 4, 4),
        page_snapshot_at="2026-04-04T14:00:00Z",
    )

    assert len(snapshot_df) == 12
    assert set(snapshot_df["market"]) == {"game_moneyline", "game_spread", "game_total"}
    assert set(snapshot_df["source_provider"]) == {"sportsgameodds"}
    assert set(snapshot_df["sportsbook"]) == {"draftkings", "fanduel"}
    home_spread = snapshot_df[(snapshot_df["market"] == "game_spread") & (snapshot_df["side"] == "home") & (snapshot_df["sportsbook"] == "draftkings")].iloc[0]
    assert home_spread["line_value"] == 18.5
    assert home_spread["price"] == -105.0

    assert len(game_lines_df) == 1
    line_row = game_lines_df.iloc[0]
    assert line_row["vegas_home_spread"] == -18.5
    assert line_row["vegas_game_total"] == 224.5
    assert line_row["open_home_spread"] == -17.5
    assert line_row["open_game_total"] == 223.5
    assert line_row["current_home_moneyline"] == -2400.0


def test_sportsgameodds_prop_rows_extract_supported_markets():
    rows = build_sportsgameodds_prop_rows(
        _sportsgameodds_events(),
        report_date=date(2026, 4, 4),
        allowed_markets={"player_points", "player_assists"},
        page_snapshot_at="2026-04-04T14:00:00Z",
    )
    frame = pd.DataFrame(rows)

    assert set(frame["market_key"]) == {"player_points", "player_assists"}
    assert set(frame["side"]) == {"over", "under"}
    assert set(frame["book"]) == {"draftkings", "fanduel"}
    assert set(frame["source_provider"]) == {"sportsgameodds"}
    assert set(frame[frame["player"] == "Tyler Herro"]["line"]) == {24.5, 5.5}


def test_covers_props_parser_returns_one_sided_rows():
    rows = build_covers_prop_rows(
        _fixture("covers_props.html"),
        report_date=date(2026, 4, 4),
        allowed_markets={"player_points", "player_assists"},
        page_snapshot_at="2026-04-04T14:10:00Z",
    )
    frame = pd.DataFrame(rows)

    assert set(frame["market_key"]) == {"player_points", "player_assists"}
    assert set(frame["side"]) == {"over", "under"}
    assert "pinnacle" in set(frame["book"])
    assert "fanduel" in set(frame["book"])


def test_covers_props_parser_prefers_trailing_price_token():
    html = """
    <table>
      <tr class="game-projections-container">
        <a class="projection-game-link">GSW @ LAL</a>
        <span class="_badge">POINTS SCORED</span>
        <div class="category-title">
          <span class="category">A. Wiggins (SF)</span>
          <span class="prediction">o15.5 Points Scored</span>
        </div>
        <td class="compare-odds-column">
          <img alt="Tooniebet logo" />
          <a class="book-odds">o15.5 -115</a>
        </td>
        <td class="compare-odds-column">
          <img alt="DraftKings logo" />
          <a class="book-odds">o15.5 EVEN</a>
        </td>
      </tr>
    </table>
    """

    rows = build_covers_prop_rows(
        html,
        report_date=date(2026, 4, 4),
        allowed_markets={"player_points"},
        page_snapshot_at="2026-04-04T14:10:00Z",
    )
    frame = pd.DataFrame(rows).sort_values("book").reset_index(drop=True)

    assert list(frame["book"]) == ["draftkings", "tooniebet"]
    assert list(frame["odds"]) == [100, -115]
    assert list(frame["line"]) == [15.5, 15.5]


def test_covers_props_parser_prefers_full_name_from_player_link():
    html = """
    <table>
      <tr class="game-projections-container">
        <a class="projection-game-link">DET @ PHI</a>
        <span class="_badge">POINTS SCORED</span>
        <div class="category-title">
          <span class="category">
            <a class="player-link" href="/sport/basketball/nba/players/4926/kevin-huerter">K. Huerter</a>
            <span class="player-position">(SG)</span>
          </span>
          <span class="prediction">u10.5 Points Scored</span>
        </div>
        <td class="compare-odds-column">
          <img alt="Bet365 logo" />
          <a class="book-odds">u10.5 -115</a>
        </td>
      </tr>
    </table>
    """

    rows = build_covers_prop_rows(
        html,
        report_date=date(2026, 4, 4),
        allowed_markets={"player_points"},
        page_snapshot_at="2026-04-04T14:10:00Z",
    )
    frame = pd.DataFrame(rows)

    assert list(frame["player"]) == ["Kevin Huerter"]


def test_merge_prop_source_rows_dedupes_exact_overlap_and_keeps_priority_source():
    scores_row = {
        "player": "Tyrese Maxey",
        "line": 27.5,
        "side": "over",
        "odds": -112,
        "book": "draftkings",
        "commence_time": "2026-04-04T23:00:00Z",
        "home_team": "Philadelphia 76ers",
        "away_team": "Detroit Pistons",
        "market_key": "player_points",
        "source_provider": "scoresandodds",
    }
    covers_row = {**scores_row, "source_provider": "covers"}

    merged, sources_used = merge_prop_source_rows(
        {"scoresandodds": [scores_row], "covers": [covers_row]},
        source_priority=("scoresandodds", "covers"),
    )

    assert sources_used == ["scoresandodds", "covers"]
    assert len(merged) == 1
    assert merged[0]["source_provider"] == "scoresandodds"


def test_merge_prop_source_rows_keeps_complementary_sides_from_multiple_sources():
    scores_row = {
        "player": "Tyrese Maxey",
        "line": 27.5,
        "side": "over",
        "odds": -112,
        "book": "draftkings",
        "commence_time": "2026-04-04T23:00:00Z",
        "home_team": "Philadelphia 76ers",
        "away_team": "Detroit Pistons",
        "market_key": "player_points",
        "source_provider": "scoresandodds",
    }
    covers_row = {
        **scores_row,
        "side": "under",
        "odds": -108,
        "source_provider": "covers",
    }

    merged, _ = merge_prop_source_rows(
        {"scoresandodds": [scores_row], "covers": [covers_row]},
        source_priority=("scoresandodds", "covers"),
    )

    frame = pd.DataFrame(merged).sort_values(["side"]).reset_index(drop=True)
    assert list(frame["side"]) == ["over", "under"]
    assert set(frame["source_provider"]) == {"scoresandodds", "covers"}


def test_readiness_rows_include_live_quote_source_metadata():
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": "rec_game",
                "game_date": "2026-04-04",
                "market": "game_spread",
                "recommendation_origin": "live_daily",
                "fair_line": 4.0,
                "selected_probability": 0.57,
                "actual_value": 7.0,
                "result": "win",
                "clv": 0.5,
                "roi": 0.91,
                "quote_source_provider": "scoresandodds",
            },
            {
                "recommendation_id": "rec_hist",
                "game_date": "2026-03-01",
                "market": "game_spread",
                "recommendation_origin": "historical_replay",
                "fair_line": 3.5,
                "selected_probability": 0.55,
                "actual_value": 1.0,
                "result": "loss",
                "clv": None,
                "roi": -1.0,
                "quote_source_provider": None,
            },
        ]
    )
    training = pd.DataFrame(
        [
            {
                "market": "game_spread",
                "holdout_mae": 10.0,
                "baseline_mae": 11.0,
                "holdout_brier": 0.22,
                "baseline_brier": 0.23,
                "trained": 1,
            }
        ]
    )

    rows = build_readiness_rows(recommendations, training)
    metrics = json.loads(next(row["metrics_json"] for row in rows if row["market"] == "game_spread"))
    assert metrics["live_quote_source"] == "scoresandodds"
    assert metrics["evidence_mode"] == "historical_plus_live"
