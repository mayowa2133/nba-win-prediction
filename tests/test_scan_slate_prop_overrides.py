from __future__ import annotations

import pandas as pd

from src.inference.scan_slate_with_model import (
    build_current_prop_overrides,
    build_feature_map,
    find_latest_feature_row,
    player_initial_last_key,
)


def test_current_market_context_overrides_historical_prop_features():
    feat_row = pd.Series(
        {
            "pts_roll5": 18.0,
            "pts_roll15": 20.0,
            "player_pts_season_mean": 21.5,
            "player_pts_career_mean": 22.5,
            "prop_pts_line": 12.5,
            "prop_over_odds_best": -110.0,
            "prop_under_odds_best": -110.0,
        }
    )
    market_row = pd.Series(
        {
            "prop_pts_line": 24.5,
            "over_odds_best": 105.0,
            "under_odds_best": -125.0,
        }
    )

    overrides = build_current_prop_overrides(feat_row, market_row)
    feature_cols = [
        "prop_pts_line",
        "prop_over_odds_best",
        "prop_under_odds_best",
        "has_prop_line",
        "prop_minus_pts_roll5",
        "prop_minus_pts_roll15",
        "prop_minus_season_mean",
        "prop_minus_career_mean",
    ]
    feat_map = build_feature_map(feature_cols, feat_row, overrides)

    assert feat_map["prop_pts_line"] == 24.5
    assert feat_map["prop_over_odds_best"] == 105.0
    assert feat_map["prop_under_odds_best"] == -125.0
    assert feat_map["has_prop_line"] == 1.0
    assert feat_map["prop_minus_pts_roll5"] == 6.5
    assert feat_map["prop_minus_pts_roll15"] == 4.5
    assert feat_map["prop_minus_season_mean"] == 3.0
    assert feat_map["prop_minus_career_mean"] == 2.0


def test_player_initial_last_key_handles_suffixes():
    assert player_initial_last_key("K. Oubre Jr.") == ("k", "oubre")
    assert player_initial_last_key("Kelly Oubre Jr.") == ("k", "oubre")
    assert player_initial_last_key("VJ Edgecombe") == ("v", "edgecombe")


def test_find_latest_feature_row_resolves_abbreviation_by_matchup_team_context():
    features = pd.DataFrame(
        [
            {
                "player_name": "Amen Thompson",
                "player_name_norm": "amen thompson",
                "player_initial_last_key": ("a", "thompson"),
                "team_abbrev": "HOU",
                "team_abbrev_norm": "HOU",
                "game_date_ts": pd.Timestamp("2026-04-01"),
            },
            {
                "player_name": "Ausar Thompson",
                "player_name_norm": "ausar thompson",
                "player_initial_last_key": ("a", "thompson"),
                "team_abbrev": "DET",
                "team_abbrev_norm": "DET",
                "game_date_ts": pd.Timestamp("2026-04-02"),
            },
        ]
    )
    market_row = pd.Series(
        {
            "player": "A. Thompson",
            "home_team": "Philadelphia 76ers",
            "away_team": "Detroit Pistons",
        }
    )

    row = find_latest_feature_row(
        features,
        "a thompson",
        pd.Timestamp("2026-04-04"),
        market_row=market_row,
    )

    assert row is not None
    assert row["player_name"] == "Ausar Thompson"


def test_find_latest_feature_row_returns_none_for_ambiguous_same_team_abbreviation():
    features = pd.DataFrame(
        [
            {
                "player_name": "Jalen Williams",
                "player_name_norm": "jalen williams",
                "player_initial_last_key": ("j", "williams"),
                "team_abbrev": "OKC",
                "team_abbrev_norm": "OKC",
                "game_date_ts": pd.Timestamp("2026-04-01"),
            },
            {
                "player_name": "Jaylin Williams",
                "player_name_norm": "jaylin williams",
                "player_initial_last_key": ("j", "williams"),
                "team_abbrev": "OKC",
                "team_abbrev_norm": "OKC",
                "game_date_ts": pd.Timestamp("2026-04-02"),
            },
        ]
    )
    market_row = pd.Series(
        {
            "player": "J. Williams",
            "home_team": "Oklahoma City Thunder",
            "away_team": "Houston Rockets",
        }
    )

    row = find_latest_feature_row(
        features,
        "j williams",
        pd.Timestamp("2026-04-04"),
        market_row=market_row,
    )

    assert row is None
