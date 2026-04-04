from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.inference import scan_slate_with_model


def test_frozen_slate_scoring_uses_current_prop_line(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()

    feature_cols = [
        "pts_roll5",
        "is_home",
        "days_since_last_game",
        "prop_pts_line",
        "has_prop_line",
        "prop_minus_pts_roll5",
    ]

    X_train = [
        [10.0, 1.0, 2.0, 12.0, 1.0, 2.0],
        [15.0, 0.0, 1.0, 18.0, 1.0, 3.0],
        [18.0, 1.0, 2.0, 22.0, 1.0, 4.0],
        [20.0, 1.0, 3.0, 25.0, 1.0, 5.0],
    ]
    y_train = [13.0, 19.5, 23.0, 27.0]
    model = LinearRegression().fit(X_train, y_train)

    model_bundle_path = models_dir / "points_regression.pkl"
    with open(model_bundle_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "sigma": 1.0,
                "feature_cols": feature_cols,
                "target": "target_pts",
            },
            f,
        )

    features_csv = data_dir / "player_points_features.csv"
    pd.DataFrame(
        [
            {
                "game_id": "old_game_1",
                "season": 2025,
                "game_date": "2026-01-08",
                "player_id": 1,
                "player_name": "Test Player",
                "team_abbrev": "ATL",
                "opp_abbrev": "BOS",
                "is_home": 0,
                "days_since_last_game": 1.0,
                "pts_roll5": 15.0,
                "target_pts": 17.0,
            },
            {
                "game_id": "old_game_1",
                "season": 2025,
                "game_date": "2026-01-08",
                "player_id": 2,
                "player_name": "Opponent Player",
                "team_abbrev": "BOS",
                "opp_abbrev": "ATL",
                "is_home": 1,
                "days_since_last_game": 2.0,
                "pts_roll5": 14.0,
                "target_pts": 16.0,
            },
        ]
    ).to_csv(features_csv, index=False)

    market_lines_csv = data_dir / "market_lines.csv"
    pd.DataFrame(
        [
            {
                "player": "Test Player",
                "prop_pts_line": 20.0,
                "over_odds_best": -110.0,
                "under_odds_best": -110.0,
                "game_date": "2026-01-10",
                "home_team": "Atlanta Hawks",
                "away_team": "Boston Celtics",
                "market_key": "player_points",
            }
        ]
    ).to_csv(market_lines_csv, index=False)

    output_csv = data_dir / "edges.csv"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan_slate_with_model.py",
            "--model-path",
            str(model_bundle_path),
            "--features-csv",
            str(features_csv),
            "--market-lines",
            str(market_lines_csv),
            "--output",
            str(output_csv),
            "--min-edge",
            "0.0",
        ],
    )

    scan_slate_with_model.main()

    scored = pd.read_csv(output_csv)
    assert len(scored) == 1

    scored_row = scored.iloc[0]
    assert float(scored_row["prop_pts_line"]) == 20.0

    expected_vector = [[15.0, 1.0, 2.0, 20.0, 1.0, 5.0]]
    expected_mu = float(model.predict(expected_vector)[0])
    historical_mu = float(model.predict([[15.0, 1.0, 2.0, 12.0, 1.0, -3.0]])[0])

    assert float(scored_row["model_mean_pts"]) == expected_mu
    assert float(scored_row["model_mean_pts"]) != historical_mu
