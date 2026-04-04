"""Build historical NBA starter records from official box scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import delete

from src.warehouse.db import init_database, session_scope
from src.warehouse.models import StarterHistoryRecord


@dataclass(frozen=True)
class StarterRow:
    game_id: str
    game_date: str
    team_abbrev: str
    opponent_abbrev: str
    player_id: int
    player_name: str
    start_position: str


def fetch_game_starters_from_api(game_id: str) -> List[dict]:
    from nba_api.stats.endpoints import boxscoretraditionalv2

    endpoint = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    df = endpoint.get_data_frames()[0]
    if "START_POSITION" not in df.columns:
        return []
    starters = df[df["START_POSITION"].fillna("").astype(str).str.strip() != ""].copy()
    return starters.to_dict(orient="records")


def build_starter_history_frame(
    logs_df: pd.DataFrame,
    *,
    existing_game_ids: Optional[Iterable[str]] = None,
    max_games: Optional[int] = None,
) -> pd.DataFrame:
    existing = {str(game_id) for game_id in (existing_game_ids or [])}
    if logs_df.empty:
        return pd.DataFrame()

    logs = logs_df.copy()
    logs["game_id"] = logs["game_id"].astype(str)
    logs["game_date"] = logs["game_date"].astype(str)

    game_context: Dict[str, dict] = {}
    for _, row in logs.drop_duplicates(subset=["game_id", "team_abbrev"]).iterrows():
        game_context.setdefault(
            str(row["game_id"]),
            {},
        )
        game_context[str(row["game_id"])][str(row["team_abbrev"])] = {
            "game_date": str(row["game_date"]),
            "opp_abbrev": str(row.get("opp_abbrev") or ""),
        }

    rows: List[StarterRow] = []
    unique_game_ids = [game_id for game_id in logs["game_id"].drop_duplicates().tolist() if game_id not in existing]
    if max_games is not None:
        unique_game_ids = unique_game_ids[:max_games]

    for game_id in unique_game_ids:
        for record in fetch_game_starters_from_api(game_id):
            team_abbrev = str(record.get("TEAM_ABBREVIATION") or "")
            context = game_context.get(game_id, {}).get(team_abbrev, {})
            rows.append(
                StarterRow(
                    game_id=game_id,
                    game_date=str(context.get("game_date") or ""),
                    team_abbrev=team_abbrev,
                    opponent_abbrev=str(context.get("opp_abbrev") or ""),
                    player_id=int(record.get("PLAYER_ID")),
                    player_name=str(record.get("PLAYER_NAME") or ""),
                    start_position=str(record.get("START_POSITION") or ""),
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def persist_starter_history(
    df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
    database_url: Optional[str] = None,
) -> int:
    if df.empty:
        return 0

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["game_id", "team_abbrev", "player_id"], keep="last")
        else:
            combined = df.copy()
        combined.to_csv(output_path, index=False)

    init_database(database_url)
    with session_scope(database_url) as session:
        game_ids = sorted({str(value) for value in df["game_id"].dropna().unique().tolist()})
        if game_ids:
            session.execute(delete(StarterHistoryRecord).where(StarterHistoryRecord.game_id.in_(game_ids)))

        count = 0
        for row in df.to_dict(orient="records"):
            session.add(
                StarterHistoryRecord(
                    game_id=str(row["game_id"]),
                    game_date=str(row["game_date"]),
                    team_abbrev=str(row["team_abbrev"]),
                    opponent_abbrev=str(row.get("opponent_abbrev") or ""),
                    player_id=int(row["player_id"]),
                    player_name=str(row["player_name"]),
                    start_position=str(row.get("start_position") or ""),
                )
            )
            count += 1
    return count
