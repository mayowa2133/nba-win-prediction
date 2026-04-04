from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.historical_game_odds import (
    backfill_player_logs,
    build_historical_snapshot_frame,
    import_historical_odds_sources,
    reconcile_historical_odds,
)


def test_historical_odds_reconciliation_prefers_lower_priority_source(tmp_path):
    source_a = tmp_path / "source_a.csv"
    source_b = tmp_path / "source_b.csv"
    manifest = tmp_path / "source_manifest.json"

    pd.DataFrame(
        [
            {
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "spread_home": -3.5,
                "total_points": 221.5,
                "moneyline_home": -150,
                "moneyline_away": 130,
            }
        ]
    ).to_csv(source_a, index=False)
    pd.DataFrame(
        [
            {
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "spread_home": -4.0,
                "total_points": 221.5,
                "moneyline_home": -155,
                "moneyline_away": 135,
            }
        ]
    ).to_csv(source_b, index=False)

    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "primary",
                    "license": "CC0-1.0",
                    "priority": 10,
                    "source_kind": "mapped_csv",
                    "path": "source_a.csv",
                    "line_phase": "close",
                    "column_map": {
                        "game_date": "game_date",
                        "home_team": "home_team",
                        "away_team": "away_team",
                        "spread_home": "spread_home",
                        "total_points": "total_points",
                        "moneyline_home": "moneyline_home",
                        "moneyline_away": "moneyline_away",
                    },
                },
                {
                    "name": "secondary",
                    "license": "unknown",
                    "priority": 20,
                    "source_kind": "mapped_csv",
                    "path": "source_b.csv",
                    "line_phase": "close",
                    "column_map": {
                        "game_date": "game_date",
                        "home_team": "home_team",
                        "away_team": "away_team",
                        "spread_home": "spread_home",
                        "total_points": "total_points",
                        "moneyline_home": "moneyline_home",
                        "moneyline_away": "moneyline_away",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    source_rows = import_historical_odds_sources(manifest_path=manifest)
    canonical_df, conflicts_df = reconcile_historical_odds(source_rows)

    spread_row = canonical_df[canonical_df["market"] == "spread"].iloc[0]
    moneyline_row = canonical_df[canonical_df["market"] == "moneyline"].iloc[0]

    assert spread_row["source_name"] == "primary"
    assert spread_row["spread_home"] == -3.5
    assert moneyline_row["source_name"] == "primary"
    assert moneyline_row["implied_prob_home_vig_free"] > 0.5
    assert len(conflicts_df) == 2


def test_positive_favorite_spread_is_converted_to_signed_home_line(tmp_path):
    source = tmp_path / "source.csv"
    manifest = tmp_path / "source_manifest.json"

    pd.DataFrame(
        [
            {
                "game_date": "2026-01-11",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "spread": 4.5,
                "whos_favored": "away",
            }
        ]
    ).to_csv(source, index=False)
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "favored_source",
                    "priority": 10,
                    "source_kind": "mapped_csv",
                    "path": "source.csv",
                    "line_phase": "close",
                    "spread_format": "positive_favorite",
                    "column_map": {
                        "game_date": "game_date",
                        "home_team": "home_team",
                        "away_team": "away_team",
                        "spread_value": "spread",
                        "favored_side": "whos_favored",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    source_rows = import_historical_odds_sources(manifest_path=manifest)
    canonical_df, _ = reconcile_historical_odds(source_rows)
    spread_row = canonical_df[canonical_df["market"] == "spread"].iloc[0]
    assert spread_row["spread_home"] == 4.5


def test_backfill_player_logs_and_synthetic_snapshots():
    canonical_df = pd.DataFrame(
        [
            {
                "game_date": "2026-01-10",
                "season": "2025",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "home_team_abbrev": "ATL",
                "away_team_abbrev": "BOS",
                "market_scope": "full_game",
                "market": "spread",
                "line_phase": "close",
                "sportsbook": "historical_import",
                "source_name": "primary",
                "source_license": "CC0-1.0",
                "source_priority": 10,
                "coverage_confidence": "high",
                "spread_home": -3.5,
                "total_points": None,
                "moneyline_home": None,
                "moneyline_away": None,
                "implied_prob_home_raw": None,
                "implied_prob_away_raw": None,
                "implied_prob_home_vig_free": None,
                "implied_prob_away_vig_free": None,
            },
            {
                "game_date": "2026-01-10",
                "season": "2025",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "home_team_abbrev": "ATL",
                "away_team_abbrev": "BOS",
                "market_scope": "full_game",
                "market": "total",
                "line_phase": "close",
                "sportsbook": "historical_import",
                "source_name": "primary",
                "source_license": "CC0-1.0",
                "source_priority": 10,
                "coverage_confidence": "high",
                "spread_home": None,
                "total_points": 221.5,
                "moneyline_home": None,
                "moneyline_away": None,
                "implied_prob_home_raw": None,
                "implied_prob_away_raw": None,
                "implied_prob_home_vig_free": None,
                "implied_prob_away_vig_free": None,
            },
            {
                "game_date": "2026-01-10",
                "season": "2025",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "home_team_abbrev": "ATL",
                "away_team_abbrev": "BOS",
                "market_scope": "full_game",
                "market": "moneyline",
                "line_phase": "close",
                "sportsbook": "historical_import",
                "source_name": "primary",
                "source_license": "CC0-1.0",
                "source_priority": 10,
                "coverage_confidence": "high",
                "spread_home": None,
                "total_points": None,
                "moneyline_home": -150,
                "moneyline_away": 130,
                "implied_prob_home_raw": 0.6,
                "implied_prob_away_raw": 0.4347826087,
                "implied_prob_home_vig_free": 0.5798319328,
                "implied_prob_away_vig_free": 0.4201680672,
            },
        ]
    )
    logs_df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-01-10",
                "season": 2025,
                "player_id": 1,
                "player_name": "Player A",
                "team_abbrev": "ATL",
                "opp_abbrev": "BOS",
                "is_home": 1,
                "minutes": 30,
                "pts": 20,
                "reb": 5,
                "ast": 6,
                "spread_close": None,
                "total_close": None,
                "ml_team": None,
                "ml_opp": None,
            },
            {
                "game_id": "g1",
                "game_date": "2026-01-10",
                "season": 2025,
                "player_id": 2,
                "player_name": "Player B",
                "team_abbrev": "BOS",
                "opp_abbrev": "ATL",
                "is_home": 0,
                "minutes": 32,
                "pts": 22,
                "reb": 8,
                "ast": 4,
                "spread_close": None,
                "total_close": None,
                "ml_team": None,
                "ml_opp": None,
            },
        ]
    )

    backfilled_logs, coverage = backfill_player_logs(logs_df, canonical_df)
    home_row = backfilled_logs[backfilled_logs["is_home"] == 1].iloc[0]
    away_row = backfilled_logs[backfilled_logs["is_home"] == 0].iloc[0]

    assert home_row["spread_close"] == -3.5
    assert away_row["spread_close"] == 3.5
    assert home_row["ml_team"] == -150
    assert away_row["ml_team"] == 130
    assert home_row["ml_team_true_prob"] > 0.5
    assert coverage["moneyline_coverage_rate"] == 1.0

    snapshot_df = build_historical_snapshot_frame(canonical_df)
    assert set(snapshot_df["market"]) == {"game_moneyline", "game_spread", "game_total"}
    assert len(snapshot_df) == 6
