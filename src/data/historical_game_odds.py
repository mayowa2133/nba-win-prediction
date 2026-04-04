"""Canonical historical NBA odds ingestion and local backfill helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from sqlalchemy import delete

from src.utils.artifact_metadata import stable_id
from src.utils.betting import american_to_prob, remove_vig_two_way
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import HistoricalOddsConflictRecord, HistoricalOddsRecord


DEFAULT_HISTORICAL_ODDS_DIR = Path("data/historical_odds")
DEFAULT_HISTORICAL_ODDS_MANIFEST = DEFAULT_HISTORICAL_ODDS_DIR / "source_manifest.json"
DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV = DEFAULT_HISTORICAL_ODDS_DIR / "canonical_historical_odds.csv"
DEFAULT_HISTORICAL_ODDS_CONFLICTS_CSV = DEFAULT_HISTORICAL_ODDS_DIR / "historical_odds_conflicts.csv"


def _to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _season_for_game_date(game_date: str, season_override: object = None) -> str:
    if season_override is not None and not pd.isna(season_override):
        text = str(season_override).strip()
        if text:
            return text
    dt = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(dt):
        return ""
    year = dt.year if dt.month >= 10 else dt.year - 1
    return str(year)


def _normalize_abbrev(value: object) -> str:
    mapped = canonical_team_abbrev(None if value is None else str(value))
    return mapped or str(value or "").strip().upper()


def _normalize_full_name(value: object) -> str:
    abbr = _normalize_abbrev(value)
    return canonical_team_name_from_abbrev(abbr) or abbr


def _normalize_side_token(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in {"home", "h", "favorite_home"}:
        return "home"
    if token in {"away", "a", "favorite_away"}:
        return "away"
    return ""


def _resolve_spread_home(
    row: pd.Series,
    *,
    spread_value_column: Optional[str],
    spread_home_column: Optional[str],
    spread_format: str,
    favored_side_column: Optional[str],
    favored_team_column: Optional[str],
    home_team_column: str,
    away_team_column: str,
) -> float:
    if spread_home_column:
        return _to_float(row.get(spread_home_column))

    value = _to_float(row.get(spread_value_column)) if spread_value_column else float("nan")
    if pd.isna(value):
        return float("nan")
    if spread_format == "signed_home":
        return value

    favored_side = ""
    if favored_side_column:
        favored_side = _normalize_side_token(row.get(favored_side_column))
    elif favored_team_column:
        favored_team = _normalize_abbrev(row.get(favored_team_column))
        home_team = _normalize_abbrev(row.get(home_team_column))
        away_team = _normalize_abbrev(row.get(away_team_column))
        if favored_team == home_team:
            favored_side = "home"
        elif favored_team == away_team:
            favored_side = "away"

    if favored_side == "home":
        return -abs(value)
    if favored_side == "away":
        return abs(value)
    return float("nan")


def _fallback_manifest() -> list[dict]:
    local_file = Path("data/historical_vegas_lines.csv")
    if not local_file.exists():
        return []
    return [
        {
            "name": "repo_historical_vegas_lines",
            "license": "unknown",
            "priority": 100,
            "source_kind": "repo_historical_vegas_lines",
            "path": str(local_file),
            "line_phase": "single_snapshot",
            "coverage_confidence": "medium",
        }
    ]


def load_source_manifest(path: Path) -> list[dict]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return list(payload.get("sources") or [])
        return list(payload)
    return _fallback_manifest()


def _resolve_source_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _normalize_source_rows(source: dict, *, root_dir: Path) -> pd.DataFrame:
    path = _resolve_source_path(root_dir, str(source.get("path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Historical odds source not found: {path}")

    source_kind = str(source.get("source_kind") or "mapped_csv")
    line_phase = str(source.get("line_phase") or "single_snapshot")
    confidence = str(source.get("coverage_confidence") or "medium")
    priority = int(source.get("priority") or 999)
    license_name = str(source.get("license") or "")
    source_name = str(source.get("name") or path.stem)

    if source_kind == "repo_historical_vegas_lines":
        raw = pd.read_csv(path)
        frame = pd.DataFrame(
            {
                "game_date": raw["game_date"].astype(str),
                "season": raw["game_date"].map(_season_for_game_date),
                "home_team_abbrev": raw["home_team"].map(_normalize_abbrev),
                "away_team_abbrev": raw["away_team"].map(_normalize_abbrev),
                "home_team": raw["home_team"].map(_normalize_full_name),
                "away_team": raw["away_team"].map(_normalize_full_name),
                "line_phase": line_phase,
                "sportsbook": source.get("sportsbook"),
                "source_name": source_name,
                "source_license": license_name,
                "source_priority": priority,
                "coverage_confidence": confidence,
                "spread_home": pd.to_numeric(raw["vegas_spread"], errors="coerce"),
                "total_points": pd.to_numeric(raw["vegas_game_total"], errors="coerce"),
                "moneyline_home": float("nan"),
                "moneyline_away": float("nan"),
                "raw_values_json": raw.apply(lambda row: json.dumps(row.to_dict()), axis=1),
                "source_path": str(path),
            }
        )
        return frame

    raw = pd.read_csv(path)
    column_map = dict(source.get("column_map") or {})
    required = {
        "game_date": column_map.get("game_date", "game_date"),
        "home_team": column_map.get("home_team", "home_team"),
        "away_team": column_map.get("away_team", "away_team"),
    }
    for required_name, column_name in required.items():
        if column_name not in raw.columns:
            raise RuntimeError(f"Historical source '{source_name}' is missing required column '{column_name}' for {required_name}")

    spread_home_column = column_map.get("spread_home")
    spread_value_column = column_map.get("spread_value")
    total_column = column_map.get("total_points")
    ml_home_column = column_map.get("moneyline_home")
    ml_away_column = column_map.get("moneyline_away")

    rows = []
    for _, row in raw.iterrows():
        game_date = str(row.get(required["game_date"]) or "").strip()
        home_value = row.get(required["home_team"])
        away_value = row.get(required["away_team"])
        home_abbrev = _normalize_abbrev(home_value)
        away_abbrev = _normalize_abbrev(away_value)
        spread_home = _resolve_spread_home(
            row,
            spread_value_column=spread_value_column,
            spread_home_column=spread_home_column,
            spread_format=str(source.get("spread_format") or "signed_home"),
            favored_side_column=column_map.get("favored_side"),
            favored_team_column=column_map.get("favored_team"),
            home_team_column=required["home_team"],
            away_team_column=required["away_team"],
        )
        total_points = _to_float(row.get(total_column)) if total_column else float("nan")
        moneyline_home = _to_float(row.get(ml_home_column)) if ml_home_column else float("nan")
        moneyline_away = _to_float(row.get(ml_away_column)) if ml_away_column else float("nan")
        if pd.isna(spread_home) and pd.isna(total_points) and pd.isna(moneyline_home) and pd.isna(moneyline_away):
            continue
        rows.append(
            {
                "game_date": game_date,
                "season": _season_for_game_date(game_date, row.get(column_map.get("season")) if column_map.get("season") else None),
                "home_team_abbrev": home_abbrev,
                "away_team_abbrev": away_abbrev,
                "home_team": canonical_team_name_from_abbrev(home_abbrev) or str(home_value),
                "away_team": canonical_team_name_from_abbrev(away_abbrev) or str(away_value),
                "line_phase": line_phase,
                "sportsbook": source.get("sportsbook"),
                "source_name": source_name,
                "source_license": license_name,
                "source_priority": priority,
                "coverage_confidence": confidence,
                "spread_home": spread_home,
                "total_points": total_points,
                "moneyline_home": moneyline_home,
                "moneyline_away": moneyline_away,
                "raw_values_json": json.dumps({key: row.get(key) for key in raw.columns}),
                "source_path": str(path),
            }
        )

    return pd.DataFrame(rows)


def import_historical_odds_sources(
    *,
    manifest_path: Path = DEFAULT_HISTORICAL_ODDS_MANIFEST,
) -> pd.DataFrame:
    manifest = load_source_manifest(manifest_path)
    if not manifest:
        return pd.DataFrame()

    root_dir = manifest_path.parent
    frames = []
    for source in manifest:
        frame = _normalize_source_rows(source, root_dir=root_dir)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["spread_home"] = pd.to_numeric(combined["spread_home"], errors="coerce")
    combined["total_points"] = pd.to_numeric(combined["total_points"], errors="coerce")
    combined["moneyline_home"] = pd.to_numeric(combined["moneyline_home"], errors="coerce")
    combined["moneyline_away"] = pd.to_numeric(combined["moneyline_away"], errors="coerce")
    return combined.dropna(subset=["game_date", "home_team_abbrev", "away_team_abbrev"], how="any").reset_index(drop=True)


def reconcile_historical_odds(source_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    canonical_rows = []
    conflict_rows = []
    group_keys = ["game_date", "season", "home_team_abbrev", "away_team_abbrev", "home_team", "away_team", "line_phase"]

    for keys, frame in source_rows.groupby(group_keys, dropna=False):
        game_date, season, home_abbrev, away_abbrev, home_team, away_team, line_phase = keys
        sorted_frame = frame.sort_values(["source_priority", "source_name", "source_path"])

        market_specs = {
            "spread": ("spread_home",),
            "total": ("total_points",),
            "moneyline": ("moneyline_home", "moneyline_away"),
        }
        for market, columns in market_specs.items():
            candidates = sorted_frame.dropna(subset=list(columns), how="any").copy()
            if candidates.empty:
                continue
            chosen = candidates.iloc[0]
            candidate_values = []
            distinct = set()
            for _, candidate in candidates.iterrows():
                value_tuple = tuple(None if pd.isna(candidate[column]) else float(candidate[column]) for column in columns)
                distinct.add(value_tuple)
                candidate_values.append(
                    {
                        "source_name": str(candidate["source_name"]),
                        "source_priority": int(candidate["source_priority"]),
                        "values": {column: candidate[column] for column in columns},
                    }
                )
            if len(distinct) > 1:
                conflict_rows.append(
                    {
                        "game_date": game_date,
                        "home_team_abbrev": home_abbrev,
                        "away_team_abbrev": away_abbrev,
                        "market": market,
                        "line_phase": line_phase,
                        "conflict_reason": f"Multiple candidate values found for {market}",
                        "candidate_values_json": json.dumps(candidate_values),
                        "resolved_source_name": str(chosen["source_name"]),
                    }
                )

            home_prob_raw = away_prob_raw = home_prob_vig_free = away_prob_vig_free = float("nan")
            if market == "moneyline":
                home_prob_raw = american_to_prob(chosen["moneyline_home"])
                away_prob_raw = american_to_prob(chosen["moneyline_away"])
                home_prob_vig_free, away_prob_vig_free = remove_vig_two_way(
                    chosen["moneyline_home"],
                    chosen["moneyline_away"],
                )

            canonical_rows.append(
                {
                    "game_date": game_date,
                    "season": season,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_team_abbrev": home_abbrev,
                    "away_team_abbrev": away_abbrev,
                    "market_scope": "full_game",
                    "market": market,
                    "line_phase": line_phase,
                    "sportsbook": chosen.get("sportsbook"),
                    "source_name": str(chosen["source_name"]),
                    "source_license": str(chosen.get("source_license") or ""),
                    "source_priority": int(chosen["source_priority"]),
                    "coverage_confidence": str(chosen.get("coverage_confidence") or "medium"),
                    "spread_home": chosen.get("spread_home"),
                    "total_points": chosen.get("total_points"),
                    "moneyline_home": chosen.get("moneyline_home"),
                    "moneyline_away": chosen.get("moneyline_away"),
                    "implied_prob_home_raw": home_prob_raw,
                    "implied_prob_away_raw": away_prob_raw,
                    "implied_prob_home_vig_free": home_prob_vig_free,
                    "implied_prob_away_vig_free": away_prob_vig_free,
                    "raw_values_json": chosen.get("raw_values_json"),
                    "source_path": str(chosen.get("source_path") or ""),
                }
            )

    canonical_df = pd.DataFrame(canonical_rows)
    conflicts_df = pd.DataFrame(conflict_rows)
    return canonical_df, conflicts_df


def persist_historical_odds(
    canonical_df: pd.DataFrame,
    conflicts_df: pd.DataFrame,
    *,
    database_url: Optional[str],
) -> tuple[int, int]:
    init_database(database_url)
    with session_scope(database_url) as session:
        session.execute(delete(HistoricalOddsRecord))
        session.execute(delete(HistoricalOddsConflictRecord))

        odds_count = 0
        for row in canonical_df.fillna("").to_dict(orient="records"):
            raw_values = row.get("raw_values_json")
            if isinstance(raw_values, str) and raw_values.strip():
                raw_values = json.loads(raw_values)
            session.add(
                HistoricalOddsRecord(
                    game_date=str(row["game_date"]),
                    season=str(row.get("season") or ""),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    home_team_abbrev=str(row["home_team_abbrev"]),
                    away_team_abbrev=str(row["away_team_abbrev"]),
                    market_scope=str(row.get("market_scope") or "full_game"),
                    market=str(row["market"]),
                    line_phase=str(row["line_phase"]),
                    sportsbook=str(row.get("sportsbook") or "") or None,
                    source_name=str(row["source_name"]),
                    source_license=str(row.get("source_license") or "") or None,
                    source_priority=int(row.get("source_priority") or 999),
                    coverage_confidence=str(row.get("coverage_confidence") or "medium"),
                    spread_home=float(row["spread_home"]) if row.get("spread_home") not in {"", None} else None,
                    total_points=float(row["total_points"]) if row.get("total_points") not in {"", None} else None,
                    moneyline_home=float(row["moneyline_home"]) if row.get("moneyline_home") not in {"", None} else None,
                    moneyline_away=float(row["moneyline_away"]) if row.get("moneyline_away") not in {"", None} else None,
                    implied_prob_home_raw=float(row["implied_prob_home_raw"]) if row.get("implied_prob_home_raw") not in {"", None} else None,
                    implied_prob_away_raw=float(row["implied_prob_away_raw"]) if row.get("implied_prob_away_raw") not in {"", None} else None,
                    implied_prob_home_vig_free=float(row["implied_prob_home_vig_free"]) if row.get("implied_prob_home_vig_free") not in {"", None} else None,
                    implied_prob_away_vig_free=float(row["implied_prob_away_vig_free"]) if row.get("implied_prob_away_vig_free") not in {"", None} else None,
                    raw_values_json=raw_values if isinstance(raw_values, dict) else None,
                    source_path=str(row.get("source_path") or "") or None,
                )
            )
            odds_count += 1

        conflict_count = 0
        for row in conflicts_df.fillna("").to_dict(orient="records"):
            values = row.get("candidate_values_json")
            if isinstance(values, str) and values.strip():
                values = json.loads(values)
            session.add(
                HistoricalOddsConflictRecord(
                    game_date=str(row["game_date"]),
                    home_team_abbrev=str(row["home_team_abbrev"]),
                    away_team_abbrev=str(row["away_team_abbrev"]),
                    market=str(row["market"]),
                    line_phase=str(row["line_phase"]),
                    conflict_reason=str(row["conflict_reason"]),
                    candidate_values_json={"candidates": values} if isinstance(values, list) else None,
                    resolved_source_name=str(row.get("resolved_source_name") or "") or None,
                )
            )
            conflict_count += 1
    return odds_count, conflict_count


def write_historical_odds_artifacts(
    canonical_df: pd.DataFrame,
    conflicts_df: pd.DataFrame,
    *,
    canonical_output_path: Path = DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV,
    conflicts_output_path: Path = DEFAULT_HISTORICAL_ODDS_CONFLICTS_CSV,
) -> None:
    canonical_output_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts_output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_csv(canonical_output_path, index=False)
    conflicts_df.to_csv(conflicts_output_path, index=False)


def _best_phase_rows(canonical_df: pd.DataFrame) -> pd.DataFrame:
    if canonical_df.empty:
        return canonical_df
    phase_rank = {"close": 0, "single_snapshot": 1, "open": 2}
    frame = canonical_df.copy()
    frame["phase_rank"] = frame["line_phase"].map(lambda value: phase_rank.get(str(value), 99))
    return frame.sort_values(["game_date", "home_team_abbrev", "away_team_abbrev", "market", "phase_rank", "source_priority"])


def build_backfill_market_frame(canonical_df: pd.DataFrame) -> pd.DataFrame:
    if canonical_df.empty:
        return pd.DataFrame()

    frame = _best_phase_rows(canonical_df)
    selected = frame.groupby(["game_date", "home_team_abbrev", "away_team_abbrev", "market"], as_index=False).head(1)
    pivot = (
        selected.pivot_table(
            index=["game_date", "home_team_abbrev", "away_team_abbrev"],
            columns="market",
            values=[
                "spread_home",
                "total_points",
                "moneyline_home",
                "moneyline_away",
                "implied_prob_home_vig_free",
                "implied_prob_away_vig_free",
                "source_name",
                "line_phase",
            ],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    return pivot.rename(
        columns={
            "spread_home_spread": "spread_home",
            "total_points_total": "total_points",
            "moneyline_home_moneyline": "moneyline_home",
            "moneyline_away_moneyline": "moneyline_away",
            "implied_prob_home_vig_free_moneyline": "moneyline_home_vig_free",
            "implied_prob_away_vig_free_moneyline": "moneyline_away_vig_free",
            "source_name_spread": "spread_source",
            "source_name_total": "total_source",
            "source_name_moneyline": "moneyline_source",
            "line_phase_spread": "spread_line_phase",
            "line_phase_total": "total_line_phase",
            "line_phase_moneyline": "moneyline_line_phase",
        }
    )


def backfill_player_logs(logs_df: pd.DataFrame, canonical_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if logs_df.empty or canonical_df.empty:
        return logs_df.copy(), {
            "spread_coverage_rate": 0.0,
            "total_coverage_rate": 0.0,
            "moneyline_coverage_rate": 0.0,
        }

    logs = logs_df.copy()
    home_games = (
        logs[logs["is_home"] == 1][["game_id", "game_date", "team_abbrev", "opp_abbrev"]]
        .drop_duplicates(subset=["game_id"])
        .rename(columns={"team_abbrev": "home_team_abbrev", "opp_abbrev": "away_team_abbrev"})
    )
    game_market = build_backfill_market_frame(canonical_df)
    enriched_games = home_games.merge(
        game_market,
        on=["game_date", "home_team_abbrev", "away_team_abbrev"],
        how="left",
    )
    logs = logs.merge(enriched_games, on=["game_id", "game_date"], how="left")

    existing_spread = pd.to_numeric(logs.get("spread_close"), errors="coerce")
    existing_total = pd.to_numeric(logs.get("total_close"), errors="coerce")
    existing_ml_team = pd.to_numeric(logs.get("ml_team"), errors="coerce")
    existing_ml_opp = pd.to_numeric(logs.get("ml_opp"), errors="coerce")

    backfill_spread = pd.to_numeric(logs.get("spread_home"), errors="coerce")
    backfill_total = pd.to_numeric(logs.get("total_points"), errors="coerce")
    backfill_ml_home = pd.to_numeric(logs.get("moneyline_home"), errors="coerce")
    backfill_ml_away = pd.to_numeric(logs.get("moneyline_away"), errors="coerce")
    backfill_prob_home = pd.to_numeric(logs.get("moneyline_home_vig_free"), errors="coerce")
    backfill_prob_away = pd.to_numeric(logs.get("moneyline_away_vig_free"), errors="coerce")

    logs["spread_close"] = existing_spread.where(existing_spread.notna(), backfill_spread.where(logs["is_home"] == 1, -backfill_spread))
    logs["total_close"] = existing_total.where(existing_total.notna(), backfill_total)
    logs["ml_team"] = existing_ml_team.where(existing_ml_team.notna(), backfill_ml_home.where(logs["is_home"] == 1, backfill_ml_away))
    logs["ml_opp"] = existing_ml_opp.where(existing_ml_opp.notna(), backfill_ml_away.where(logs["is_home"] == 1, backfill_ml_home))

    logs["ml_team_true_prob"] = backfill_prob_home.where(logs["is_home"] == 1, backfill_prob_away)
    logs["ml_opp_true_prob"] = backfill_prob_away.where(logs["is_home"] == 1, backfill_prob_home)

    def _series(name: str) -> pd.Series:
        if name in logs.columns:
            return logs[name]
        return pd.Series(pd.NA, index=logs.index)

    logs["spread_source"] = _series("spread_source")
    logs["total_source"] = _series("total_source")
    logs["moneyline_source"] = _series("moneyline_source")
    logs["market_line_phase"] = _series("moneyline_line_phase").fillna(_series("spread_line_phase")).fillna(_series("total_line_phase"))

    logs = logs.drop(
        columns=[
            "home_team_abbrev",
            "away_team_abbrev",
            "spread_home",
            "total_points",
            "moneyline_home",
            "moneyline_away",
            "moneyline_home_vig_free",
            "moneyline_away_vig_free",
            "spread_line_phase",
            "total_line_phase",
            "moneyline_line_phase",
        ],
        errors="ignore",
    )

    coverage = {
        "spread_coverage_rate": float(pd.to_numeric(logs["spread_close"], errors="coerce").notna().mean()),
        "total_coverage_rate": float(pd.to_numeric(logs["total_close"], errors="coerce").notna().mean()),
        "moneyline_coverage_rate": float(
            (
                pd.to_numeric(logs["ml_team"], errors="coerce").notna()
                & pd.to_numeric(logs["ml_opp"], errors="coerce").notna()
            ).mean()
        ),
    }
    return logs, coverage


def build_historical_snapshot_frame(canonical_df: pd.DataFrame) -> pd.DataFrame:
    if canonical_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in _best_phase_rows(canonical_df).iterrows():
        if str(row["line_phase"]) not in {"close", "single_snapshot"}:
            continue
        fixture_id = stable_id(row["game_date"], row["home_team_abbrev"], row["away_team_abbrev"], prefix="histfx")
        commence_time = f"{row['game_date']}T23:59:00Z"
        captured_at = f"{row['game_date']}T23:55:00Z"
        common = {
            "fixture_id": fixture_id,
            "game_id": stable_id(row["game_date"], row["home_team_abbrev"], row["away_team_abbrev"], prefix="game"),
            "game_date": row["game_date"],
            "commence_time": commence_time,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "sportsbook": str(row.get("sportsbook") or row.get("source_name") or "historical_import"),
            "bookmaker_id": None,
            "market_id": None,
            "market_name": str(row["market"]),
            "is_historical": 1,
            "source_url": str(row.get("source_path") or ""),
            "snapshot_type": f"historical_{row['line_phase']}",
            "captured_at": captured_at,
        }

        if row["market"] == "spread" and not pd.isna(row.get("spread_home")):
            line = abs(float(row["spread_home"]))
            rows.append({**common, "market": "game_spread", "side": "home", "line_value": line, "price": None})
            rows.append({**common, "market": "game_spread", "side": "away", "line_value": line, "price": None})
        elif row["market"] == "total" and not pd.isna(row.get("total_points")):
            total = float(row["total_points"])
            rows.append({**common, "market": "game_total", "side": "over", "line_value": total, "price": None})
            rows.append({**common, "market": "game_total", "side": "under", "line_value": total, "price": None})
        elif row["market"] == "moneyline":
            if not pd.isna(row.get("moneyline_home")):
                rows.append({**common, "market": "game_moneyline", "side": "home", "line_value": 0.0, "price": float(row["moneyline_home"])})
            if not pd.isna(row.get("moneyline_away")):
                rows.append({**common, "market": "game_moneyline", "side": "away", "line_value": 0.0, "price": float(row["moneyline_away"])})

    return pd.DataFrame(rows).drop_duplicates(
        subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
        keep="last",
    )


def export_game_lines_history(canonical_df: pd.DataFrame, *, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _best_phase_rows(canonical_df)
    if frame.empty:
        return 0

    records = []
    for keys, group in frame.groupby(["game_date", "home_team", "away_team", "home_team_abbrev", "away_team_abbrev"], dropna=False):
        game_date, home_team, away_team, home_abbrev, away_abbrev = keys
        spread_row = group[group["market"] == "spread"].head(1)
        total_row = group[group["market"] == "total"].head(1)
        if spread_row.empty and total_row.empty:
            continue
        spread_home = float(spread_row.iloc[0]["spread_home"]) if not spread_row.empty and not pd.isna(spread_row.iloc[0]["spread_home"]) else float("nan")
        total_points = float(total_row.iloc[0]["total_points"]) if not total_row.empty and not pd.isna(total_row.iloc[0]["total_points"]) else float("nan")
        records.append(
            {
                "event_id": stable_id(game_date, home_abbrev, away_abbrev, prefix="histgame"),
                "game_date": game_date,
                "commence_time": f"{game_date}T23:59:00Z",
                "home_team": home_team,
                "away_team": away_team,
                "vegas_game_total": total_points,
                "vegas_home_spread": spread_home,
                "vegas_away_spread": -spread_home if not pd.isna(spread_home) else float("nan"),
                "vegas_abs_spread": abs(spread_home) if not pd.isna(spread_home) else float("nan"),
            }
        )

    if not records:
        return 0

    lines_df = pd.DataFrame(records)
    for game_date, group in lines_df.groupby("game_date"):
        (output_dir / f"game_lines_{game_date}.csv").write_text(group.to_csv(index=False), encoding="utf-8")
    return int(len(lines_df))
