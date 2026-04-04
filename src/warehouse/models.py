"""SQLAlchemy models for the local analytics warehouse."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Float, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.contracts.versions import (
    GAME_ODDS_SCHEMA_VERSION,
    HISTORICAL_ODDS_SCHEMA_VERSION,
    INJURY_REPORT_SCHEMA_VERSION,
    LINEUP_PROJECTION_SCHEMA_VERSION,
    RECOMMENDATION_API_SCHEMA_VERSION,
    SETTLEMENT_SCHEMA_VERSION,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class PlayerLogRecord(Base):
    __tablename__ = "player_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False)
    team_abbrev: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    opponent_abbrev: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    season: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rebounds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    assists: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="nba_api")
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        Index("ix_player_logs_player_date", "player_name", "game_date"),
    )


class TeamGameContextRecord(Base):
    __tablename__ = "team_game_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    home_team: Mapped[str] = mapped_column(String(64), nullable=False)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False)
    home_rest_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_rest_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    projected_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    projected_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="pipeline")
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        UniqueConstraint("game_id", name="uq_team_game_context_game_id"),
    )


class InjuryReportRecord(Base):
    __tablename__ = "injury_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    report_date: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    report_time_et: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    matchup: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    row_kind: Mapped[str] = mapped_column(String(32), default="player_status")
    player_name: Mapped[str] = mapped_column(String(128), nullable=False)
    team_abbrev: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    report_status: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    raw_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normalized_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    projected_availability: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), default=INJURY_REPORT_SCHEMA_VERSION)
    source: Mapped[str] = mapped_column(String(64), default="manual")
    reported_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    pulled_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        Index("ix_injury_reports_game_player", "game_date", "player_name"),
        Index("ix_injury_reports_report_date_team", "report_date", "team_abbrev"),
    )


class GameOddsSnapshotRecord(Base):
    __tablename__ = "game_odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    commence_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    home_team: Mapped[str] = mapped_column(String(64), nullable=False)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    sportsbook: Mapped[str] = mapped_column(String(64), nullable=False)
    bookmaker_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    line_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    market_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_historical: Mapped[int] = mapped_column(Integer, default=0)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_provider: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_page_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_book: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_consensus_quote: Mapped[int] = mapped_column(Integer, default=0)
    page_snapshot_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    snapshot_type: Mapped[str] = mapped_column(String(32), default="intraday")
    schema_version: Mapped[str] = mapped_column(String(16), default=GAME_ODDS_SCHEMA_VERSION)
    captured_at: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        Index("ix_game_odds_lookup", "game_date", "market", "sportsbook"),
        Index("ix_game_odds_fixture_snapshot", "fixture_id", "captured_at"),
    )


class ClosingLineRecord(Base):
    __tablename__ = "closing_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    commence_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    home_team: Mapped[str] = mapped_column(String(64), nullable=False)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    sportsbook: Mapped[str] = mapped_column(String(64), nullable=False)
    line_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_provider: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_page_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_book: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_consensus_quote: Mapped[int] = mapped_column(Integer, default=0)
    page_snapshot_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    captured_at: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_type: Mapped[str] = mapped_column(String(32), default="closing")
    schema_version: Mapped[str] = mapped_column(String(16), default=GAME_ODDS_SCHEMA_VERSION)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        UniqueConstraint("fixture_id", "market", "side", "sportsbook", name="uq_closing_lines_fixture_market_side_book"),
    )


class HistoricalOddsRecord(Base):
    __tablename__ = "historical_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    season: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    home_team: Mapped[str] = mapped_column(String(64), nullable=False)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False)
    home_team_abbrev: Mapped[str] = mapped_column(String(16), nullable=False)
    away_team_abbrev: Mapped[str] = mapped_column(String(16), nullable=False)
    market_scope: Mapped[str] = mapped_column(String(32), default="full_game")
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    line_phase: Mapped[str] = mapped_column(String(32), nullable=False)
    sportsbook: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    source_license: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_priority: Mapped[int] = mapped_column(Integer, nullable=False, default=999)
    coverage_confidence: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")
    spread_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    moneyline_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    moneyline_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    implied_prob_home_raw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    implied_prob_away_raw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    implied_prob_home_vig_free: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    implied_prob_away_vig_free: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_values_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), default=HISTORICAL_ODDS_SCHEMA_VERSION)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        UniqueConstraint(
            "game_date",
            "home_team_abbrev",
            "away_team_abbrev",
            "market",
            "line_phase",
            "source_name",
            "sportsbook",
            name="uq_historical_odds_game_market_phase_source",
        ),
        Index("ix_historical_odds_lookup", "game_date", "home_team_abbrev", "away_team_abbrev", "market"),
    )


class HistoricalOddsConflictRecord(Base):
    __tablename__ = "historical_odds_conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    home_team_abbrev: Mapped[str] = mapped_column(String(16), nullable=False)
    away_team_abbrev: Mapped[str] = mapped_column(String(16), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    line_phase: Mapped[str] = mapped_column(String(32), nullable=False)
    conflict_reason: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_values_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    resolved_source_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), default=HISTORICAL_ODDS_SCHEMA_VERSION)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        Index("ix_historical_odds_conflicts_lookup", "game_date", "home_team_abbrev", "away_team_abbrev", "market"),
    )


class StarterHistoryRecord(Base):
    __tablename__ = "starter_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    team_abbrev: Mapped[str] = mapped_column(String(16), nullable=False)
    opponent_abbrev: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False)
    start_position: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="nba_api")
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        UniqueConstraint("game_id", "team_abbrev", "player_id", name="uq_starter_history_game_team_player"),
        Index("ix_starter_history_lookup", "game_date", "team_abbrev"),
    )


class LineupProjectionRecord(Base):
    __tablename__ = "lineup_projections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    projection_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    team_abbrev: Mapped[str] = mapped_column(String(16), nullable=False)
    opponent_abbrev: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    projected_starter: Mapped[str] = mapped_column(String(128), nullable=False)
    projected_position: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    starter_probability: Mapped[float] = mapped_column(Float, nullable=False)
    projection_reason: Mapped[str] = mapped_column(Text, nullable=False)
    injury_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    consensus_disagreement: Mapped[int] = mapped_column(Integer, default=0)
    schema_version: Mapped[str] = mapped_column(String(16), default=LINEUP_PROJECTION_SCHEMA_VERSION)
    projection_generated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        UniqueConstraint("projection_id", "team_abbrev", "projected_starter", name="uq_lineup_projection_projection_team_player"),
        Index("ix_lineup_projection_lookup", "game_date", "team_abbrev"),
    )


class RecommendationRecord(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    game_id: Mapped[str] = mapped_column(String(64), nullable=False)
    player: Mapped[str] = mapped_column(String(128), nullable=False)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    home_team: Mapped[str] = mapped_column(String(64), nullable=False)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(64), nullable=False)
    selection: Mapped[str] = mapped_column(String(16), nullable=False)
    sportsbook_line: Mapped[float] = mapped_column(Float, nullable=False)
    sportsbook_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fair_line: Mapped[float] = mapped_column(Float, nullable=False)
    fair_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    selected_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_implied_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    data_timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    recommendation_origin: Mapped[str] = mapped_column(String(32), default="live_daily")
    published_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    published_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    published_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    quote_source_provider: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    quote_source_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    quote_source_book: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    clv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lineup_context_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    injury_context_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    reasons_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    api_schema_version: Mapped[str] = mapped_column(String(16), default=RECOMMENDATION_API_SCHEMA_VERSION)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        Index("ix_recommendations_date_market", "game_date", "market"),
        Index("ix_recommendations_game_id", "game_id"),
        Index("ix_recommendations_origin_date", "recommendation_origin", "game_date"),
    )


class SettledBetOutcomeRecord(Base):
    __tablename__ = "settled_bet_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_id: Mapped[str] = mapped_column(String(64), nullable=False)
    game_date: Mapped[str] = mapped_column(String(16), nullable=False)
    market: Mapped[str] = mapped_column(String(64), nullable=False)
    selection: Mapped[str] = mapped_column(String(16), nullable=False)
    recommendation_origin: Mapped[str] = mapped_column(String(32), default="live_daily")
    published_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    published_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    clv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), default=SETTLEMENT_SCHEMA_VERSION)
    settled_at: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(default=utc_now)

    __table_args__ = (
        UniqueConstraint("recommendation_id", name="uq_settled_bet_outcomes_recommendation"),
        Index("ix_settled_bet_outcomes_market_date", "market", "game_date"),
    )


class MarketReadinessSnapshotRecord(Base):
    __tablename__ = "market_readiness_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    as_of_timestamp: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("market", name="uq_market_readiness_market"),
    )
