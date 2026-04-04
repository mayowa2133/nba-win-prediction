from __future__ import annotations

import pickle

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.inference.score_game_markets import build_game_market_recommendations
from src.utils.game_market_features import GAME_MARKET_FEATURE_COLUMNS


def _write_bundle(path, bundle):
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def test_game_market_scoring_emits_recommendations_for_all_markets(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    X_train = pd.DataFrame([[0.0] * len(GAME_MARKET_FEATURE_COLUMNS)] * 4, columns=GAME_MARKET_FEATURE_COLUMNS)

    moneyline_model = DummyClassifier(strategy="prior")
    moneyline_model.fit(X_train, [1, 1, 1, 0])
    spread_model = DummyRegressor(strategy="constant", constant=8.0)
    spread_model.fit(X_train, [8.0, 8.0, 8.0, 8.0])
    total_model = DummyRegressor(strategy="constant", constant=225.0)
    total_model.fit(X_train, [225.0, 225.0, 225.0, 225.0])

    _write_bundle(models_dir / "game_moneyline_model.pkl", {"model": moneyline_model, "feature_cols": GAME_MARKET_FEATURE_COLUMNS, "metadata": {"artifact_created_at": "2026-03-31T12:00:00+00:00"}})
    _write_bundle(models_dir / "game_spread_model.pkl", {"model": spread_model, "sigma": 8.0, "feature_cols": GAME_MARKET_FEATURE_COLUMNS, "metadata": {"artifact_created_at": "2026-03-31T12:00:00+00:00"}})
    _write_bundle(models_dir / "game_total_model.pkl", {"model": total_model, "sigma": 10.0, "feature_cols": GAME_MARKET_FEATURE_COLUMNS, "metadata": {"artifact_created_at": "2026-03-31T12:00:00+00:00"}})

    logs = pd.DataFrame(
        [
            {"game_id": "1", "season": "2025-26", "game_date": "2026-03-20", "player_id": 1, "player_name": "ATL A", "team_abbrev": "ATL", "opp_abbrev": "BOS", "is_home": 1, "minutes": 30, "pts": 26, "reb": 5, "ast": 7, "stl": 1, "blk": 0, "tov": 2, "fg3m": 3, "fg3a": 8, "fga": 18, "fgm": 9, "fta": 6, "ftm": 5, "pf": 2, "oreb": 1, "team_score": 112, "opp_score": 104, "spread_close": 5.5, "total_close": 220.5, "ml_team": -140, "ml_opp": 120},
            {"game_id": "1", "season": "2025-26", "game_date": "2026-03-20", "player_id": 2, "player_name": "BOS A", "team_abbrev": "BOS", "opp_abbrev": "ATL", "is_home": 0, "minutes": 31, "pts": 24, "reb": 6, "ast": 6, "stl": 1, "blk": 1, "tov": 3, "fg3m": 2, "fg3a": 7, "fga": 17, "fgm": 8, "fta": 4, "ftm": 4, "pf": 2, "oreb": 1, "team_score": 104, "opp_score": 112, "spread_close": -5.5, "total_close": 220.5, "ml_team": 120, "ml_opp": -140},
            {"game_id": "2", "season": "2025-26", "game_date": "2026-03-24", "player_id": 1, "player_name": "ATL A", "team_abbrev": "ATL", "opp_abbrev": "NYK", "is_home": 0, "minutes": 34, "pts": 29, "reb": 4, "ast": 8, "stl": 1, "blk": 0, "tov": 2, "fg3m": 4, "fg3a": 9, "fga": 19, "fgm": 10, "fta": 5, "ftm": 5, "pf": 2, "oreb": 1, "team_score": 118, "opp_score": 109, "spread_close": -3.5, "total_close": 221.5, "ml_team": -130, "ml_opp": 110},
            {"game_id": "2", "season": "2025-26", "game_date": "2026-03-24", "player_id": 3, "player_name": "NYK A", "team_abbrev": "NYK", "opp_abbrev": "ATL", "is_home": 1, "minutes": 32, "pts": 22, "reb": 8, "ast": 5, "stl": 2, "blk": 1, "tov": 2, "fg3m": 2, "fg3a": 6, "fga": 16, "fgm": 8, "fta": 4, "ftm": 4, "pf": 2, "oreb": 2, "team_score": 109, "opp_score": 118, "spread_close": 3.5, "total_close": 221.5, "ml_team": 110, "ml_opp": -130},
            {"game_id": "3", "season": "2025-26", "game_date": "2026-03-25", "player_id": 2, "player_name": "BOS A", "team_abbrev": "BOS", "opp_abbrev": "MIA", "is_home": 1, "minutes": 33, "pts": 27, "reb": 7, "ast": 5, "stl": 1, "blk": 1, "tov": 2, "fg3m": 3, "fg3a": 8, "fga": 19, "fgm": 10, "fta": 5, "ftm": 4, "pf": 2, "oreb": 1, "team_score": 115, "opp_score": 103, "spread_close": 6.5, "total_close": 217.5, "ml_team": -150, "ml_opp": 130},
            {"game_id": "3", "season": "2025-26", "game_date": "2026-03-25", "player_id": 4, "player_name": "MIA A", "team_abbrev": "MIA", "opp_abbrev": "BOS", "is_home": 0, "minutes": 31, "pts": 21, "reb": 5, "ast": 7, "stl": 1, "blk": 0, "tov": 3, "fg3m": 2, "fg3a": 7, "fga": 17, "fgm": 8, "fta": 5, "ftm": 3, "pf": 3, "oreb": 1, "team_score": 103, "opp_score": 115, "spread_close": -6.5, "total_close": 217.5, "ml_team": 130, "ml_opp": -150},
        ]
    )
    odds = pd.DataFrame(
        [
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_moneyline", "side": "home", "sportsbook": "pinnacle", "line_value": 0.0, "price": 120.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_moneyline", "side": "away", "sportsbook": "pinnacle", "line_value": 0.0, "price": -140.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_spread", "side": "home", "sportsbook": "pinnacle", "line_value": 5.5, "price": -110.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_spread", "side": "away", "sportsbook": "pinnacle", "line_value": 5.5, "price": -110.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_total", "side": "over", "sportsbook": "pinnacle", "line_value": 220.5, "price": 110.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_total", "side": "under", "sportsbook": "pinnacle", "line_value": 220.5, "price": -130.0, "captured_at": "2026-03-31T18:00:00Z"},
        ]
    )

    scored = build_game_market_recommendations(
        logs_df=logs,
        odds_snapshots_df=odds,
        models_dir=models_dir,
        sportsbook="pinnacle",
        target_date="2026-03-31",
        min_edge=0.0,
    )

    assert set(scored["market"]) == {"game_moneyline", "game_spread", "game_total"}
    assert set(scored["market_readiness_status"]) == {"experimental"}
    assert scored["injury_context_json"].notna().all()


def test_game_market_scoring_skips_missing_market_bundle(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    X_train = pd.DataFrame([[0.0] * len(GAME_MARKET_FEATURE_COLUMNS)] * 4, columns=GAME_MARKET_FEATURE_COLUMNS)

    spread_model = DummyRegressor(strategy="constant", constant=8.0)
    spread_model.fit(X_train, [8.0, 8.0, 8.0, 8.0])
    total_model = DummyRegressor(strategy="constant", constant=225.0)
    total_model.fit(X_train, [225.0, 225.0, 225.0, 225.0])

    _write_bundle(models_dir / "game_spread_model.pkl", {"model": spread_model, "sigma": 8.0, "feature_cols": GAME_MARKET_FEATURE_COLUMNS, "metadata": {"artifact_created_at": "2026-03-31T12:00:00+00:00"}})
    _write_bundle(models_dir / "game_total_model.pkl", {"model": total_model, "sigma": 10.0, "feature_cols": GAME_MARKET_FEATURE_COLUMNS, "metadata": {"artifact_created_at": "2026-03-31T12:00:00+00:00"}})

    logs = pd.DataFrame(
        [
            {"game_id": "1", "season": "2025-26", "game_date": "2026-03-20", "player_id": 1, "player_name": "ATL A", "team_abbrev": "ATL", "opp_abbrev": "BOS", "is_home": 1, "minutes": 30, "pts": 26, "reb": 5, "ast": 7, "stl": 1, "blk": 0, "tov": 2, "fg3m": 3, "fg3a": 8, "fga": 18, "fgm": 9, "fta": 6, "ftm": 5, "pf": 2, "oreb": 1, "team_score": 112, "opp_score": 104, "spread_close": 5.5, "total_close": 220.5, "ml_team": -140, "ml_opp": 120},
            {"game_id": "1", "season": "2025-26", "game_date": "2026-03-20", "player_id": 2, "player_name": "BOS A", "team_abbrev": "BOS", "opp_abbrev": "ATL", "is_home": 0, "minutes": 31, "pts": 24, "reb": 6, "ast": 6, "stl": 1, "blk": 1, "tov": 3, "fg3m": 2, "fg3a": 7, "fga": 17, "fgm": 8, "fta": 4, "ftm": 4, "pf": 2, "oreb": 1, "team_score": 104, "opp_score": 112, "spread_close": -5.5, "total_close": 220.5, "ml_team": 120, "ml_opp": -140},
        ]
    )
    odds = pd.DataFrame(
        [
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_moneyline", "side": "home", "sportsbook": "pinnacle", "line_value": 0.0, "price": 120.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_moneyline", "side": "away", "sportsbook": "pinnacle", "line_value": 0.0, "price": -140.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_spread", "side": "home", "sportsbook": "pinnacle", "line_value": 5.5, "price": -110.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_spread", "side": "away", "sportsbook": "pinnacle", "line_value": 5.5, "price": -110.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_total", "side": "over", "sportsbook": "pinnacle", "line_value": 220.5, "price": 110.0, "captured_at": "2026-03-31T18:00:00Z"},
            {"fixture_id": "fx1", "game_id": "game_future", "game_date": "2026-03-31", "commence_time": "2026-03-31T23:00:00Z", "home_team": "Atlanta Hawks", "away_team": "Boston Celtics", "market": "game_total", "side": "under", "sportsbook": "pinnacle", "line_value": 220.5, "price": -130.0, "captured_at": "2026-03-31T18:00:00Z"},
        ]
    )

    scored = build_game_market_recommendations(
        logs_df=logs,
        odds_snapshots_df=odds,
        models_dir=models_dir,
        sportsbook="pinnacle",
        target_date="2026-03-31",
        min_edge=0.0,
    )

    assert set(scored["market"]) == {"game_spread", "game_total"}
