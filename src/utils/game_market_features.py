"""Shared feature engineering for NBA game-level markets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.utils.betting import american_to_prob
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev


ROLL_WINDOWS = (5, 15)
TEAM_FORM_BASE_COLS = [
    "points_for",
    "points_against",
    "margin",
    "win",
    "total_points",
    "minutes_sum",
]
INJURY_CONTEXT_COLS = [
    "out_count",
    "doubtful_count",
    "questionable_count",
    "probable_count",
    "unavailable_minutes_sum",
]
LINEUP_CONTEXT_COLS = [
    "projected_returning_starters",
    "projected_replacements",
    "projected_avg_starter_prob",
    "projected_low_confidence_count",
]


def _first_non_null(series: pd.Series):
    cleaned = series.dropna()
    return cleaned.iloc[0] if not cleaned.empty else None


def _season_start_year(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = df.copy()
    frame[column] = pd.to_datetime(frame[column])
    return frame


def build_team_games(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse player logs into one row per team per game."""
    if logs_df.empty:
        return pd.DataFrame()

    logs = logs_df.copy()
    logs["game_id"] = logs["game_id"].astype(str)
    logs["team_abbrev"] = logs["team_abbrev"].map(lambda value: canonical_team_abbrev(value) or str(value))
    logs["opp_abbrev"] = logs["opp_abbrev"].map(lambda value: canonical_team_abbrev(value) or str(value))
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs["season_start"] = logs["season"].map(_season_start_year)

    grouped = (
        logs.sort_values(["game_id", "team_abbrev", "player_id"])
        .groupby(["game_id", "game_date", "season", "season_start", "team_abbrev", "opp_abbrev", "is_home"], as_index=False)
        .agg(
            minutes_sum=("minutes", "sum"),
            player_count=("player_id", "nunique"),
            team_score=("team_score", _first_non_null),
            opp_score=("opp_score", _first_non_null),
            spread_close=("spread_close", _first_non_null),
            total_close=("total_close", _first_non_null),
            ml_team=("ml_team", _first_non_null),
            ml_opp=("ml_opp", _first_non_null),
            ml_team_true_prob=("ml_team_true_prob", _first_non_null) if "ml_team_true_prob" in logs.columns else ("ml_team", lambda _: None),
            ml_opp_true_prob=("ml_opp_true_prob", _first_non_null) if "ml_opp_true_prob" in logs.columns else ("ml_opp", lambda _: None),
        )
    )
    grouped["points_for"] = pd.to_numeric(grouped["team_score"], errors="coerce").fillna(0.0)
    grouped["points_against"] = pd.to_numeric(grouped["opp_score"], errors="coerce").fillna(0.0)
    grouped["margin"] = grouped["points_for"] - grouped["points_against"]
    grouped["total_points"] = grouped["points_for"] + grouped["points_against"]
    grouped["win"] = (grouped["margin"] > 0).astype(float)
    return grouped


def add_team_form_features(team_games_df: pd.DataFrame) -> pd.DataFrame:
    if team_games_df.empty:
        return pd.DataFrame()

    team_games = team_games_df.copy().sort_values(["team_abbrev", "game_date", "game_id"])
    for base_col in TEAM_FORM_BASE_COLS:
        for window in ROLL_WINDOWS:
            team_games[f"{base_col}_roll{window}"] = team_games.groupby("team_abbrev")[base_col].transform(
                lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
            )

    team_games["rest_days"] = team_games.groupby("team_abbrev")["game_date"].diff().dt.days.fillna(7).clip(lower=0)
    team_games["is_b2b"] = (team_games["rest_days"] <= 1).astype(float)
    team_games["is_long_rest"] = (team_games["rest_days"] >= 3).astype(float)
    return team_games


def build_latest_player_minutes(logs_df: pd.DataFrame, *, before_date: Optional[str] = None) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame(columns=["team_abbrev", "player_name", "minutes_roll15"])

    logs = logs_df.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    if before_date:
        logs = logs[logs["game_date"] < pd.to_datetime(before_date)]
    if logs.empty:
        return pd.DataFrame(columns=["team_abbrev", "player_name", "minutes_roll15"])

    logs = logs.sort_values(["player_id", "game_date"])
    logs["minutes_roll15"] = logs.groupby("player_id")["minutes"].transform(
        lambda series: series.shift(1).rolling(window=15, min_periods=1).mean()
    )
    latest = logs.groupby(["team_abbrev", "player_name"], as_index=False).tail(1).copy()
    return latest[["team_abbrev", "player_name", "minutes_roll15"]]


def build_team_injury_aggregates(
    injuries_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    *,
    target_date: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate official injury rows into team-level context features."""
    if injuries_df.empty:
        return pd.DataFrame(columns=["game_date", "team_abbrev", *INJURY_CONTEXT_COLS, "summary"])

    injuries = injuries_df.copy()
    injuries["row_kind"] = injuries["row_kind"].astype(str)
    injuries = injuries[injuries["row_kind"] == "player_status"].copy()
    if injuries.empty:
        return pd.DataFrame(columns=["game_date", "team_abbrev", *INJURY_CONTEXT_COLS, "summary"])

    injuries["report_date"] = injuries["report_date"].fillna(injuries["game_date"]).astype(str)
    injuries["game_date"] = injuries["game_date"].fillna(injuries["report_date"]).astype(str)
    injuries["team_abbrev"] = injuries["team_abbrev"].map(lambda value: canonical_team_abbrev(value) or str(value))
    if target_date:
        injuries = injuries[(injuries["report_date"] == target_date) | (injuries["game_date"] == target_date)].copy()
    if injuries.empty:
        return pd.DataFrame(columns=["game_date", "team_abbrev", *INJURY_CONTEXT_COLS, "summary"])

    sort_cols = ["team_abbrev", "player_name", "report_date", "reported_at"]
    for col in sort_cols:
        if col not in injuries.columns:
            injuries[col] = ""
    injuries = injuries.sort_values(sort_cols).drop_duplicates(["team_abbrev", "player_name"], keep="last")
    minutes_df = build_latest_player_minutes(logs_df, before_date=target_date)
    injuries = injuries.merge(minutes_df, on=["team_abbrev", "player_name"], how="left")
    injuries["minutes_roll15"] = pd.to_numeric(injuries["minutes_roll15"], errors="coerce").fillna(0.0)

    def _count(frame: pd.DataFrame, status: str) -> int:
        return int((frame["normalized_status"].astype(str) == status).sum())

    rows: List[dict] = []
    for (game_date, team_abbrev), frame in injuries.groupby(["game_date", "team_abbrev"], dropna=False):
        unavailable = frame[frame["normalized_status"].astype(str).isin({"out", "doubtful", "inactive_other"})].copy()
        summary_parts = []
        if not unavailable.empty:
            names = ", ".join(unavailable["player_name"].astype(str).tolist()[:4])
            summary_parts.append(f"Unavailable: {names}")
        questionable = frame[frame["normalized_status"].astype(str) == "questionable"]
        if not questionable.empty:
            names = ", ".join(questionable["player_name"].astype(str).tolist()[:4])
            summary_parts.append(f"Questionable: {names}")

        rows.append(
            {
                "game_date": str(game_date),
                "team_abbrev": str(team_abbrev),
                "out_count": _count(frame, "out") + _count(frame, "inactive_other"),
                "doubtful_count": _count(frame, "doubtful"),
                "questionable_count": _count(frame, "questionable"),
                "probable_count": _count(frame, "probable"),
                "unavailable_minutes_sum": float(unavailable["minutes_roll15"].sum()),
                "summary": "; ".join(summary_parts),
            }
        )
    return pd.DataFrame(rows)


def build_lineup_projection_aggregates(
    lineup_df: pd.DataFrame,
    starter_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate projected lineups into team-level continuity signals."""
    if lineup_df.empty:
        return pd.DataFrame(columns=["game_date", "team_abbrev", *LINEUP_CONTEXT_COLS])

    history = starter_history_df.copy()
    history["game_date"] = pd.to_datetime(history["game_date"])
    latest_by_team_date = (
        history.sort_values(["team_abbrev", "game_date"])
        .groupby(["team_abbrev", "game_date"])["player_name"]
        .agg(lambda values: set(values.astype(str)))
        .reset_index()
    )

    rows: List[dict] = []
    for (game_date, team_abbrev), frame in lineup_df.groupby(["game_date", "team_abbrev"], dropna=False):
        projected = set(frame["projected_starter"].astype(str))
        before = latest_by_team_date[
            (latest_by_team_date["team_abbrev"] == team_abbrev)
            & (latest_by_team_date["game_date"] < pd.to_datetime(game_date))
        ]
        prior_starters = set()
        if not before.empty:
            prior_starters = set(before.sort_values("game_date").iloc[-1]["player_name"])
        returning = len(projected & prior_starters)
        rows.append(
            {
                "game_date": str(game_date),
                "team_abbrev": str(team_abbrev),
                "projected_returning_starters": returning,
                "projected_replacements": max(0, len(projected) - returning),
                "projected_avg_starter_prob": float(pd.to_numeric(frame["starter_probability"], errors="coerce").fillna(0.0).mean()),
                "projected_low_confidence_count": int((pd.to_numeric(frame["starter_probability"], errors="coerce").fillna(0.0) < 0.75).sum()),
            }
        )
    return pd.DataFrame(rows)


def _prefixed_context(frame: pd.DataFrame, prefix: str, columns: Iterable[str]) -> pd.DataFrame:
    renamed = frame.copy()
    renamed = renamed.rename(columns={column: f"{prefix}_{column}" for column in columns})
    renamed = renamed.rename(columns={"team_abbrev": f"{prefix}_team_abbrev"})
    return renamed


def _base_game_frame(team_games_df: pd.DataFrame) -> pd.DataFrame:
    team_games = add_team_form_features(team_games_df)
    if team_games.empty:
        return pd.DataFrame()

    home = team_games[team_games["is_home"] == 1].copy()
    away = team_games[team_games["is_home"] == 0].copy()

    home_cols = [
        "game_id",
        "game_date",
        "season",
        "season_start",
        "team_abbrev",
        "opp_abbrev",
        "points_for",
        "points_against",
        "margin",
        "total_points",
        "spread_close",
        "total_close",
        "ml_team",
        "ml_opp",
        "rest_days",
        "is_b2b",
        "is_long_rest",
    ]
    away_cols = [
        "game_id",
        "game_date",
        "season",
        "season_start",
        "team_abbrev",
        "opp_abbrev",
        "points_for",
        "points_against",
        "margin",
        "total_points",
        "spread_close",
        "total_close",
        "ml_team",
        "ml_opp",
        "rest_days",
        "is_b2b",
        "is_long_rest",
    ]
    for base_col in TEAM_FORM_BASE_COLS:
        for window in ROLL_WINDOWS:
            home_cols.append(f"{base_col}_roll{window}")
            away_cols.append(f"{base_col}_roll{window}")

    home = home[home_cols].rename(columns={col: f"home_{col}" for col in home_cols if col not in {"game_id", "game_date", "season", "season_start"}})
    away = away[away_cols].rename(columns={col: f"away_{col}" for col in away_cols if col not in {"game_id", "game_date", "season", "season_start"}})
    games = home.merge(
        away,
        on=["game_id", "game_date", "season", "season_start"],
        how="inner",
        validate="one_to_one",
    )
    games["home_team"] = games["home_team_abbrev"].map(lambda abbr: canonical_team_name_from_abbrev(abbr) or abbr)
    games["away_team"] = games["away_team_abbrev"].map(lambda abbr: canonical_team_name_from_abbrev(abbr) or abbr)
    games["home_win"] = (games["home_margin"] > 0).astype(int)
    games["home_margin_target"] = pd.to_numeric(games["home_margin"], errors="coerce").fillna(0.0)
    games["game_total_target"] = pd.to_numeric(games["home_total_points"], errors="coerce").fillna(0.0)
    games["market_total_line"] = pd.to_numeric(games["home_total_close"], errors="coerce")
    games["market_home_spread_line"] = pd.to_numeric(games["home_spread_close"], errors="coerce").abs()
    home_true = pd.to_numeric(games.get("home_ml_team_true_prob"), errors="coerce")
    away_true = pd.to_numeric(games.get("home_ml_opp_true_prob"), errors="coerce")
    games["market_home_ml_implied"] = home_true.where(home_true.notna(), games["home_ml_team"].map(american_to_prob))
    games["market_away_ml_implied"] = away_true.where(away_true.notna(), games["home_ml_opp"].map(american_to_prob))
    return games


def _apply_context_merges(
    games: pd.DataFrame,
    *,
    injuries_df: Optional[pd.DataFrame],
    logs_df: pd.DataFrame,
    lineup_df: Optional[pd.DataFrame],
    starter_history_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    merged = games.copy()
    if injuries_df is not None and not injuries_df.empty:
        injury_aggs = build_team_injury_aggregates(injuries_df, logs_df)
        if not injury_aggs.empty:
            home_inj = _prefixed_context(injury_aggs, "home", ["game_date", *INJURY_CONTEXT_COLS, "summary"]).rename(
                columns={"home_game_date": "game_date"}
            )
            away_inj = _prefixed_context(injury_aggs, "away", ["game_date", *INJURY_CONTEXT_COLS, "summary"]).rename(
                columns={"away_game_date": "game_date"}
            )
            merged = merged.merge(
                home_inj,
                left_on=["game_date", "home_team_abbrev"],
                right_on=["game_date", "home_team_abbrev"],
                how="left",
            )
            merged = merged.merge(
                away_inj,
                left_on=["game_date", "away_team_abbrev"],
                right_on=["game_date", "away_team_abbrev"],
                how="left",
            )

    if lineup_df is not None and starter_history_df is not None and not lineup_df.empty and not starter_history_df.empty:
        lineup_aggs = build_lineup_projection_aggregates(lineup_df, starter_history_df)
        if not lineup_aggs.empty:
            home_lineup = _prefixed_context(lineup_aggs, "home", ["game_date", *LINEUP_CONTEXT_COLS]).rename(
                columns={"home_game_date": "game_date"}
            )
            away_lineup = _prefixed_context(lineup_aggs, "away", ["game_date", *LINEUP_CONTEXT_COLS]).rename(
                columns={"away_game_date": "game_date"}
            )
            merged = merged.merge(
                home_lineup,
                left_on=["game_date", "home_team_abbrev"],
                right_on=["game_date", "home_team_abbrev"],
                how="left",
            )
            merged = merged.merge(
                away_lineup,
                left_on=["game_date", "away_team_abbrev"],
                right_on=["game_date", "away_team_abbrev"],
                how="left",
            )

    for side in ("home", "away"):
        for column in INJURY_CONTEXT_COLS + LINEUP_CONTEXT_COLS:
            field = f"{side}_{column}"
            if field not in merged.columns:
                merged[field] = 0.0
            merged[field] = pd.to_numeric(merged[field], errors="coerce").fillna(0.0)
        summary_field = f"{side}_summary"
        if summary_field not in merged.columns:
            merged[summary_field] = ""
        merged[summary_field] = merged[summary_field].fillna("").astype(str)
    return merged


def _add_diff_features(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.copy()

    def _series(name: str) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=frame.index, dtype=float)

    for metric in ("points_for", "points_against", "margin", "win", "total_points", "minutes_sum"):
        for window in ROLL_WINDOWS:
            home_col = f"home_{metric}_roll{window}"
            away_col = f"away_{metric}_roll{window}"
            diff_col = f"{metric}_diff_roll{window}"
            if home_col in frame.columns and away_col in frame.columns:
                frame[diff_col] = _series(home_col) - _series(away_col)

    frame["rest_days_diff"] = _series("home_rest_days") - _series("away_rest_days")
    frame["unavailable_minutes_diff"] = _series("home_unavailable_minutes_sum") - _series("away_unavailable_minutes_sum")
    frame["projected_returning_diff"] = _series("home_projected_returning_starters") - _series("away_projected_returning_starters")
    return frame


def build_historical_game_market_frame(
    logs_df: pd.DataFrame,
    *,
    injuries_df: Optional[pd.DataFrame] = None,
    lineup_df: Optional[pd.DataFrame] = None,
    starter_history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    team_games = build_team_games(logs_df)
    games = _base_game_frame(team_games)
    if games.empty:
        return games
    games = _apply_context_merges(
        games,
        injuries_df=injuries_df,
        logs_df=logs_df,
        lineup_df=lineup_df,
        starter_history_df=starter_history_df,
    )
    return _add_diff_features(games)


@dataclass(frozen=True)
class UpcomingGame:
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_team_abbrev: str
    away_team_abbrev: str
    fixture_id: Optional[str] = None
    market_snapshot_at: Optional[str] = None
    market_total_line: Optional[float] = None
    market_home_spread_line: Optional[float] = None
    market_home_ml_implied: Optional[float] = None
    market_away_ml_implied: Optional[float] = None


def build_upcoming_game_market_frame(
    logs_df: pd.DataFrame,
    upcoming_games_df: pd.DataFrame,
    *,
    injuries_df: Optional[pd.DataFrame] = None,
    lineup_df: Optional[pd.DataFrame] = None,
    starter_history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if upcoming_games_df.empty:
        return pd.DataFrame()

    team_games = add_team_form_features(build_team_games(logs_df))
    if team_games.empty:
        return pd.DataFrame()

    lineup_aggs = (
        build_lineup_projection_aggregates(lineup_df, starter_history_df)
        if lineup_df is not None and starter_history_df is not None and not lineup_df.empty and not starter_history_df.empty
        else pd.DataFrame()
    )
    injury_aggs = (
        build_team_injury_aggregates(injuries_df, logs_df)
        if injuries_df is not None and not injuries_df.empty
        else pd.DataFrame()
    )

    rows: List[dict] = []
    for record in upcoming_games_df.to_dict(orient="records"):
        game_date = str(record["game_date"])
        game_dt = pd.to_datetime(game_date)
        home_abbrev = canonical_team_abbrev(record.get("home_team_abbrev") or record.get("home_team")) or str(record.get("home_team_abbrev") or record.get("home_team"))
        away_abbrev = canonical_team_abbrev(record.get("away_team_abbrev") or record.get("away_team")) or str(record.get("away_team_abbrev") or record.get("away_team"))

        home_hist = team_games[(team_games["team_abbrev"] == home_abbrev) & (team_games["game_date"] < game_dt)].sort_values("game_date")
        away_hist = team_games[(team_games["team_abbrev"] == away_abbrev) & (team_games["game_date"] < game_dt)].sort_values("game_date")
        if home_hist.empty or away_hist.empty:
            continue
        home_row = home_hist.iloc[-1]
        away_row = away_hist.iloc[-1]
        season_candidates = [
            candidate
            for candidate in (
                _season_start_year(home_row.get("season")),
                _season_start_year(away_row.get("season")),
            )
            if candidate is not None
        ]
        row = {
            "game_id": str(record["game_id"]),
            "fixture_id": record.get("fixture_id"),
            "game_date": game_date,
            "season_start": max(season_candidates) if season_candidates else None,
            "home_team": canonical_team_name_from_abbrev(home_abbrev) or str(record.get("home_team") or home_abbrev),
            "away_team": canonical_team_name_from_abbrev(away_abbrev) or str(record.get("away_team") or away_abbrev),
            "home_team_abbrev": home_abbrev,
            "away_team_abbrev": away_abbrev,
            "market_snapshot_at": record.get("market_snapshot_at"),
            "market_total_line": record.get("market_total_line"),
            "market_home_spread_line": record.get("market_home_spread_line"),
            "market_home_ml_implied": record.get("market_home_ml_implied"),
            "market_away_ml_implied": record.get("market_away_ml_implied"),
        }
        for prefix, team_row in (("home", home_row), ("away", away_row)):
            row[f"{prefix}_rest_days"] = float(team_row.get("rest_days") or 0.0)
            row[f"{prefix}_is_b2b"] = float(team_row.get("is_b2b") or 0.0)
            row[f"{prefix}_is_long_rest"] = float(team_row.get("is_long_rest") or 0.0)
            for base_col in TEAM_FORM_BASE_COLS:
                for window in ROLL_WINDOWS:
                    row[f"{prefix}_{base_col}_roll{window}"] = float(team_row.get(f"{base_col}_roll{window}") or 0.0)

        if not injury_aggs.empty:
            for prefix, team_abbrev in (("home", home_abbrev), ("away", away_abbrev)):
                match = injury_aggs[(injury_aggs["game_date"] == game_date) & (injury_aggs["team_abbrev"] == team_abbrev)]
                if match.empty:
                    for column in INJURY_CONTEXT_COLS:
                        row[f"{prefix}_{column}"] = 0.0
                    row[f"{prefix}_summary"] = ""
                else:
                    latest = match.iloc[-1]
                    for column in INJURY_CONTEXT_COLS:
                        row[f"{prefix}_{column}"] = float(latest.get(column) or 0.0)
                    row[f"{prefix}_summary"] = str(latest.get("summary") or "")

        if not lineup_aggs.empty:
            for prefix, team_abbrev in (("home", home_abbrev), ("away", away_abbrev)):
                match = lineup_aggs[(lineup_aggs["game_date"] == game_date) & (lineup_aggs["team_abbrev"] == team_abbrev)]
                if match.empty:
                    for column in LINEUP_CONTEXT_COLS:
                        row[f"{prefix}_{column}"] = 0.0
                else:
                    latest = match.iloc[-1]
                    for column in LINEUP_CONTEXT_COLS:
                        row[f"{prefix}_{column}"] = float(latest.get(column) or 0.0)

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return _add_diff_features(pd.DataFrame(rows))


GAME_MARKET_FEATURE_COLUMNS = [
    *(f"home_{metric}_roll{window}" for metric in TEAM_FORM_BASE_COLS for window in ROLL_WINDOWS),
    *(f"away_{metric}_roll{window}" for metric in TEAM_FORM_BASE_COLS for window in ROLL_WINDOWS),
    "home_rest_days",
    "away_rest_days",
    "home_is_b2b",
    "away_is_b2b",
    "home_is_long_rest",
    "away_is_long_rest",
    "market_total_line",
    "market_home_spread_line",
    "market_home_ml_implied",
    "market_away_ml_implied",
    *(f"home_{column}" for column in INJURY_CONTEXT_COLS),
    *(f"away_{column}" for column in INJURY_CONTEXT_COLS),
    *(f"home_{column}" for column in LINEUP_CONTEXT_COLS),
    *(f"away_{column}" for column in LINEUP_CONTEXT_COLS),
    *(f"{metric}_diff_roll{window}" for metric in TEAM_FORM_BASE_COLS for window in ROLL_WINDOWS),
    "rest_days_diff",
    "unavailable_minutes_diff",
    "projected_returning_diff",
]
