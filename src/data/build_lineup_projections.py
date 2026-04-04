"""Generate projected NBA starting lineups from starter history and official injuries."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import delete

from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import InjuryReportRecord, LineupProjectionRecord


UNAVAILABLE_STATUSES = {"out", "doubtful", "inactive_other"}


def _latest_team_games(starter_history: pd.DataFrame, *, before_date: str) -> pd.DataFrame:
    history = starter_history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"])
    history = history[history["game_date"] < pd.to_datetime(before_date)]
    if history.empty:
        return history
    latest_dates = history.groupby("team_abbrev")["game_date"].transform("max")
    return history[history["game_date"] == latest_dates].copy()


def _recent_start_counts(starter_history: pd.DataFrame, *, before_date: str, window: int = 10) -> pd.DataFrame:
    history = starter_history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"])
    history = history[history["game_date"] < pd.to_datetime(before_date)]
    if history.empty:
        return pd.DataFrame(columns=["team_abbrev", "player_name", "recent_start_count"])

    recent_game_dates = (
        history[["team_abbrev", "game_date"]]
        .drop_duplicates()
        .sort_values(["team_abbrev", "game_date"])
        .groupby("team_abbrev")
        .tail(window)
    )
    history = history.merge(recent_game_dates, on=["team_abbrev", "game_date"], how="inner")
    counts = (
        history.groupby(["team_abbrev", "player_name"], as_index=False)
        .size()
        .rename(columns={"size": "recent_start_count"})
    )
    return counts


def _recent_minutes(logs_df: pd.DataFrame, *, before_date: str, window: int = 15) -> pd.DataFrame:
    logs = logs_df.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs[logs["game_date"] < pd.to_datetime(before_date)]
    if logs.empty:
        return pd.DataFrame(columns=["team_abbrev", "player_name", "minutes_roll15"])

    logs = logs.sort_values(["player_id", "game_date"])
    logs["minutes_roll15"] = logs.groupby("player_id")["minutes"].transform(
        lambda series: series.shift(0).rolling(window=window, min_periods=1).mean()
    )
    latest = logs.groupby(["team_abbrev", "player_name"], as_index=False).tail(1)
    return latest[["team_abbrev", "player_name", "minutes_roll15"]]


def _position_lookup(player_positions_df: pd.DataFrame, starter_history: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not player_positions_df.empty:
        for _, row in player_positions_df.iterrows():
            mapping[str(row["player_name"])] = str(row.get("position") or "")
    if not starter_history.empty:
        recent_positions = (
            starter_history.groupby("player_name")["start_position"]
            .agg(lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else "")
            .to_dict()
        )
        for player_name, position in recent_positions.items():
            mapping.setdefault(str(player_name), str(position))
    return mapping


def build_lineup_projection_frame(
    *,
    target_date: date,
    starter_history_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
    player_positions_df: Optional[pd.DataFrame] = None,
    consensus_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    target_date_str = target_date.isoformat()
    starter_history = starter_history_df.copy()
    logs = logs_df.copy()
    injuries = injuries_df.copy()
    player_positions = player_positions_df.copy() if player_positions_df is not None else pd.DataFrame()

    if injuries.empty:
        return pd.DataFrame()

    latest_injuries = injuries[injuries["report_date"].astype(str) == target_date_str].copy()
    if latest_injuries.empty:
        latest_injuries = injuries[injuries["game_date"].astype(str) == target_date_str].copy()
    if latest_injuries.empty:
        return pd.DataFrame()

    latest_injuries["team_abbrev"] = latest_injuries["team_abbrev"].map(lambda value: canonical_team_abbrev(value) or value)
    team_status_rows = latest_injuries[latest_injuries["row_kind"].astype(str) == "team_status"].copy()
    if team_status_rows.empty:
        team_status_rows = latest_injuries.drop_duplicates(subset=["team_abbrev", "game_id"]).copy()

    baseline_starters = _latest_team_games(starter_history, before_date=target_date_str)
    recent_start_counts = _recent_start_counts(starter_history, before_date=target_date_str)
    recent_minutes = _recent_minutes(logs, before_date=target_date_str)
    position_lookup = _position_lookup(player_positions, starter_history)

    injuries_lookup = (
        latest_injuries[latest_injuries["row_kind"].astype(str) == "player_status"]
        .sort_values(["team_abbrev", "player_name", "reported_at"])
        .drop_duplicates(subset=["team_abbrev", "player_name"], keep="last")
        .set_index(["team_abbrev", "player_name"])
    )

    candidate_pool = (
        logs.groupby(["team_abbrev", "player_name"], as_index=False)
        .agg(player_id=("player_id", "last"))
        .merge(recent_start_counts, on=["team_abbrev", "player_name"], how="left")
        .merge(recent_minutes, on=["team_abbrev", "player_name"], how="left")
        .fillna({"recent_start_count": 0, "minutes_roll15": 0.0})
    )

    consensus_lookup = {}
    if consensus_df is not None and not consensus_df.empty:
        for _, row in consensus_df.iterrows():
            key = (str(row.get("game_date", target_date_str)), str(row.get("team_abbrev")), str(row.get("projected_starter")))
            consensus_lookup[key] = True

    projection_rows: List[dict] = []
    generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    for _, team_row in team_status_rows.iterrows():
        team_abbrev = str(team_row.get("team_abbrev") or "")
        if not team_abbrev:
            continue

        matchup = str(team_row.get("matchup") or "")
        away_team, _, home_team = matchup.partition("@")
        opponent_abbrev = home_team if team_abbrev == away_team else away_team
        projection_id = stable_id(target_date_str, matchup or team_abbrev, prefix="lineup")

        prior_starters = baseline_starters[baseline_starters["team_abbrev"] == team_abbrev].copy()
        starter_rows = []
        unavailable = set()
        for _, starter in prior_starters.iterrows():
            player_name = str(starter["player_name"])
            injury_key = (team_abbrev, player_name)
            injury_status = ""
            if injury_key in injuries_lookup.index:
                injury_status = str(injuries_lookup.loc[injury_key, "normalized_status"] or "")
            if injury_status in UNAVAILABLE_STATUSES:
                unavailable.add(player_name)
                continue

            probability = 0.95
            if injury_status == "probable":
                probability = 0.80
            elif injury_status == "questionable":
                probability = 0.60

            starter_rows.append(
                {
                    "projection_id": projection_id,
                    "game_id": str(team_row.get("game_id") or stable_id(target_date_str, matchup, prefix="game")),
                    "game_date": target_date_str,
                    "team_abbrev": team_abbrev,
                    "opponent_abbrev": opponent_abbrev,
                    "projected_starter": player_name,
                    "projected_position": str(starter.get("start_position") or position_lookup.get(player_name, "")),
                    "starter_probability": probability,
                    "projection_reason": "carry_forward_confirmed_starter",
                    "injury_status": injury_status,
                }
            )

        selected_players = {row["projected_starter"] for row in starter_rows}
        needed = max(0, 5 - len(starter_rows))
        if needed > 0:
            candidates = candidate_pool[candidate_pool["team_abbrev"] == team_abbrev].copy()
            candidates = candidates[~candidates["player_name"].isin(selected_players | unavailable)]
            candidates["position"] = candidates["player_name"].map(lambda name: position_lookup.get(str(name), ""))
            candidates = candidates.sort_values(
                ["recent_start_count", "minutes_roll15", "player_name"],
                ascending=[False, False, True],
            )
            for _, candidate in candidates.head(needed).iterrows():
                player_name = str(candidate["player_name"])
                starter_rows.append(
                    {
                        "projection_id": projection_id,
                        "game_id": str(team_row.get("game_id") or stable_id(target_date_str, matchup, prefix="game")),
                        "game_date": target_date_str,
                        "team_abbrev": team_abbrev,
                        "opponent_abbrev": opponent_abbrev,
                        "projected_starter": player_name,
                        "projected_position": str(candidate.get("position") or ""),
                        "starter_probability": 0.70,
                        "projection_reason": "replacement_from_recent_starts_minutes",
                        "injury_status": "",
                    }
                )
                selected_players.add(player_name)

        for row in starter_rows[:5]:
            row["projection_generated_at"] = generated_at
            key = (target_date_str, row["team_abbrev"], row["projected_starter"])
            row["consensus_disagreement"] = 0 if not consensus_lookup else int(key not in consensus_lookup)
            projection_rows.append(row)

    return pd.DataFrame(projection_rows)


def persist_lineup_projections(
    df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
    database_url: Optional[str] = None,
) -> int:
    if df.empty:
        return 0

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    init_database(database_url)
    with session_scope(database_url) as session:
        projection_ids = sorted({str(value) for value in df["projection_id"].dropna().unique().tolist()})
        if projection_ids:
            session.execute(delete(LineupProjectionRecord).where(LineupProjectionRecord.projection_id.in_(projection_ids)))

        count = 0
        for row in df.to_dict(orient="records"):
            session.add(
                LineupProjectionRecord(
                    projection_id=str(row["projection_id"]),
                    game_id=str(row["game_id"]),
                    game_date=str(row["game_date"]),
                    team_abbrev=str(row["team_abbrev"]),
                    opponent_abbrev=str(row.get("opponent_abbrev") or ""),
                    projected_starter=str(row["projected_starter"]),
                    projected_position=str(row.get("projected_position") or ""),
                    starter_probability=float(row.get("starter_probability") or 0.0),
                    projection_reason=str(row["projection_reason"]),
                    injury_status=str(row.get("injury_status") or ""),
                    consensus_disagreement=int(row.get("consensus_disagreement") or 0),
                    projection_generated_at=str(row["projection_generated_at"]),
                )
            )
            count += 1
    return count
