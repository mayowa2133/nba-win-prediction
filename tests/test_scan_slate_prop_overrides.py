from __future__ import annotations

import pandas as pd

from src.inference.scan_slate_with_model import build_current_prop_overrides, build_feature_map


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

