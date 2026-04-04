from __future__ import annotations

from src.data.build_starter_history import normalize_boxscore_game_id


def test_normalize_boxscore_game_id_pads_logs_style_ids():
    assert normalize_boxscore_game_id("22501128") == "0022501128"
    assert normalize_boxscore_game_id("21500101") == "0021500101"
    assert normalize_boxscore_game_id("0022501128") == "0022501128"
