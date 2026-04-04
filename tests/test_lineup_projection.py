from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.build_lineup_projections import build_lineup_projection_frame


def test_lineup_projection_replaces_ruled_out_starter_from_recent_context():
    starter_history = pd.DataFrame(
        [
            {"game_id": "g_prev", "game_date": "2026-03-28", "team_abbrev": "ATL", "opponent_abbrev": "BOS", "player_id": 1, "player_name": "Starter A", "start_position": "G"},
            {"game_id": "g_prev", "game_date": "2026-03-28", "team_abbrev": "ATL", "opponent_abbrev": "BOS", "player_id": 2, "player_name": "Starter B", "start_position": "G"},
            {"game_id": "g_prev", "game_date": "2026-03-28", "team_abbrev": "ATL", "opponent_abbrev": "BOS", "player_id": 3, "player_name": "Starter C", "start_position": "F"},
            {"game_id": "g_prev", "game_date": "2026-03-28", "team_abbrev": "ATL", "opponent_abbrev": "BOS", "player_id": 4, "player_name": "Starter D", "start_position": "F"},
            {"game_id": "g_prev", "game_date": "2026-03-28", "team_abbrev": "ATL", "opponent_abbrev": "BOS", "player_id": 5, "player_name": "Starter E", "start_position": "C"},
            {"game_id": "g_old", "game_date": "2026-03-20", "team_abbrev": "ATL", "opponent_abbrev": "MIA", "player_id": 6, "player_name": "Bench F", "start_position": "G"},
            {"game_id": "g_old2", "game_date": "2026-03-22", "team_abbrev": "ATL", "opponent_abbrev": "NYK", "player_id": 6, "player_name": "Bench F", "start_position": "G"},
        ]
    )
    logs = pd.DataFrame(
        [
            {"game_date": "2026-03-20", "player_id": 6, "player_name": "Bench F", "team_abbrev": "ATL", "minutes": 28},
            {"game_date": "2026-03-22", "player_id": 6, "player_name": "Bench F", "team_abbrev": "ATL", "minutes": 30},
            {"game_date": "2026-03-24", "player_id": 7, "player_name": "Bench G", "team_abbrev": "ATL", "minutes": 18},
        ]
    )
    injuries = pd.DataFrame(
        [
            {"game_id": "game_1", "game_date": "2026-03-31", "report_date": "2026-03-31", "matchup": "ATL@BOS", "row_kind": "team_status", "player_name": "", "team_abbrev": "ATL", "normalized_status": "inactive_other", "reported_at": "2026-03-31T10:00:00Z"},
            {"game_id": "game_1", "game_date": "2026-03-31", "report_date": "2026-03-31", "matchup": "ATL@BOS", "row_kind": "player_status", "player_name": "Starter A", "team_abbrev": "ATL", "normalized_status": "out", "reported_at": "2026-03-31T10:00:00Z"},
            {"game_id": "game_1", "game_date": "2026-03-31", "report_date": "2026-03-31", "matchup": "ATL@BOS", "row_kind": "player_status", "player_name": "Starter B", "team_abbrev": "ATL", "normalized_status": "probable", "reported_at": "2026-03-31T10:00:00Z"},
        ]
    )
    positions = pd.DataFrame(
        [
            {"player_name": "Starter A", "position": "G"},
            {"player_name": "Starter B", "position": "G"},
            {"player_name": "Starter C", "position": "F"},
            {"player_name": "Starter D", "position": "F"},
            {"player_name": "Starter E", "position": "C"},
            {"player_name": "Bench F", "position": "G"},
            {"player_name": "Bench G", "position": "G"},
        ]
    )

    df = build_lineup_projection_frame(
        target_date=date(2026, 3, 31),
        starter_history_df=starter_history,
        logs_df=logs,
        injuries_df=injuries,
        player_positions_df=positions,
    )

    assert len(df) == 5
    assert "Bench F" in set(df["projected_starter"])
    probable = df[df["projected_starter"] == "Starter B"].iloc[0]
    assert probable["starter_probability"] == 0.80
    replacement = df[df["projected_starter"] == "Bench F"].iloc[0]
    assert replacement["projection_reason"] == "replacement_from_recent_starts_minutes"


def test_lineup_projection_handles_empty_starter_history_frame():
    logs = pd.DataFrame(
        [
            {"game_date": "2026-03-20", "player_id": 6, "player_name": "Bench F", "team_abbrev": "ATL", "minutes": 28},
            {"game_date": "2026-03-22", "player_id": 7, "player_name": "Bench G", "team_abbrev": "ATL", "minutes": 30},
            {"game_date": "2026-03-24", "player_id": 8, "player_name": "Bench H", "team_abbrev": "ATL", "minutes": 27},
            {"game_date": "2026-03-25", "player_id": 9, "player_name": "Bench I", "team_abbrev": "ATL", "minutes": 26},
            {"game_date": "2026-03-26", "player_id": 10, "player_name": "Bench J", "team_abbrev": "ATL", "minutes": 25},
        ]
    )
    injuries = pd.DataFrame(
        [
            {"game_id": "game_1", "game_date": "2026-03-31", "report_date": "2026-03-31", "matchup": "ATL@BOS", "row_kind": "team_status", "player_name": "", "team_abbrev": "ATL", "normalized_status": "inactive_other", "reported_at": "2026-03-31T10:00:00Z"},
        ]
    )
    positions = pd.DataFrame(
        [
            {"player_name": "Bench F", "position": "G"},
            {"player_name": "Bench G", "position": "G"},
            {"player_name": "Bench H", "position": "F"},
            {"player_name": "Bench I", "position": "F"},
            {"player_name": "Bench J", "position": "C"},
        ]
    )

    df = build_lineup_projection_frame(
        target_date=date(2026, 3, 31),
        starter_history_df=pd.DataFrame(),
        logs_df=logs,
        injuries_df=injuries,
        player_positions_df=positions,
    )

    assert len(df) == 5
    assert set(df["projected_starter"]) == {"Bench F", "Bench G", "Bench H", "Bench I", "Bench J"}


def test_lineup_projection_dedupes_repeated_team_status_rows():
    logs = pd.DataFrame(
        [
            {"game_date": "2026-03-20", "player_id": 6, "player_name": "Bench F", "team_abbrev": "ATL", "minutes": 28},
            {"game_date": "2026-03-22", "player_id": 7, "player_name": "Bench G", "team_abbrev": "ATL", "minutes": 30},
            {"game_date": "2026-03-24", "player_id": 8, "player_name": "Bench H", "team_abbrev": "ATL", "minutes": 27},
            {"game_date": "2026-03-25", "player_id": 9, "player_name": "Bench I", "team_abbrev": "ATL", "minutes": 26},
            {"game_date": "2026-03-26", "player_id": 10, "player_name": "Bench J", "team_abbrev": "ATL", "minutes": 25},
        ]
    )
    injuries = pd.DataFrame(
        [
            {"game_id": "game_1", "game_date": "2026-03-31", "report_date": "2026-03-31", "matchup": "ATL@BOS", "row_kind": "team_status", "player_name": "", "team_abbrev": "ATL", "normalized_status": "inactive_other", "reported_at": "2026-03-31T09:00:00Z"},
            {"game_id": "game_1", "game_date": "2026-03-31", "report_date": "2026-03-31", "matchup": "ATL@BOS", "row_kind": "team_status", "player_name": "", "team_abbrev": "ATL", "normalized_status": "inactive_other", "reported_at": "2026-03-31T10:00:00Z"},
        ]
    )

    df = build_lineup_projection_frame(
        target_date=date(2026, 3, 31),
        starter_history_df=pd.DataFrame(),
        logs_df=logs,
        injuries_df=injuries,
        player_positions_df=pd.DataFrame(),
    )

    assert len(df) == 5
    assert not df.duplicated(subset=["projection_id", "team_abbrev", "projected_starter"]).any()
