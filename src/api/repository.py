"""Repository layer for precomputed recommendation artifacts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.api.mapping import ensure_edge_identifiers, row_to_recommendation_payload
from src.api.models import (
    GameRecommendations,
    MarketReadinessEntry,
    Recommendation,
)
from src.contracts.market_readiness import DEFAULT_MARKET_READINESS
from src.warehouse.db import DEFAULT_DATABASE_URL, default_sqlite_database_path, session_scope
from src.warehouse.models import MarketReadinessSnapshotRecord, RecommendationRecord


DEFAULT_EDGES_PATH = Path("data/edges_with_market.csv")
DEFAULT_RECOMMENDATION_ORIGINS = ("live_daily",)


class RecommendationRepository:
    """Load precomputed scored outputs and map them into API contracts."""

    def __init__(self, edges_path: Optional[Path] = None, database_url: Optional[str] = None):
        env_path = os.getenv("NBA_BETTING_EDGES_PATH")
        self.edges_path = Path(env_path) if env_path else (edges_path or DEFAULT_EDGES_PATH)
        self.database_url = database_url or os.getenv("NBA_BETTING_DATABASE_URL")
        self.default_database_path = default_sqlite_database_path()

    def _data_timestamp(self) -> str:
        try:
            dt_obj = datetime.fromtimestamp(self.edges_path.stat().st_mtime, tz=timezone.utc)
            return dt_obj.replace(microsecond=0).isoformat()
        except FileNotFoundError:
            return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _load_edges(self) -> pd.DataFrame:
        if not self.edges_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.edges_path)
        return ensure_edge_identifiers(df, data_timestamp=self._data_timestamp())

    def _database_enabled(self) -> bool:
        return bool(self.database_url) or self.default_database_path.exists()

    def _resolved_database_url(self) -> str:
        return self.database_url or DEFAULT_DATABASE_URL

    def _record_to_recommendation(self, record: RecommendationRecord) -> Recommendation:
        payload = {
            "id": record.id,
            "game_id": record.game_id,
            "player": record.player or None,
            "game_date": record.game_date,
            "home_team": record.home_team,
            "away_team": record.away_team,
            "market": record.market,
            "selection": record.selection,
            "sportsbook_line": record.sportsbook_line,
            "sportsbook_odds": record.sportsbook_odds,
            "fair_line": record.fair_line,
            "fair_odds": record.fair_odds,
            "edge": record.edge,
            "selected_probability": record.selected_probability,
            "market_implied_probability": record.market_implied_probability,
            "confidence": record.confidence,
            "status": record.status,
            "model_version": record.model_version,
            "data_timestamp": record.data_timestamp,
            "published_line": record.published_line,
            "published_odds": record.published_odds,
            "published_at": record.published_at,
            "likely_range_low": record.likely_range_low,
            "likely_range_high": record.likely_range_high,
            "likely_range_confidence": record.likely_range_confidence,
            "most_likely_milestone": record.most_likely_milestone,
            "most_likely_milestone_probability": record.most_likely_milestone_probability,
            "milestone_probabilities": record.milestone_probabilities_json or [],
            "closing_line": record.closing_line,
            "closing_odds": record.closing_odds,
            "actual_value": record.actual_value,
            "result": record.result,
            "clv": record.clv,
            "roi": record.roi,
            "lineup_context_json": record.lineup_context_json,
            "injury_context_json": record.injury_context_json,
            "reasons": record.reasons_json or [],
            "api_schema_version": record.api_schema_version,
        }
        return Recommendation(**payload)

    def _origin_filter(self, origins: Optional[Iterable[str]]) -> List[str]:
        if origins is None:
            return list(DEFAULT_RECOMMENDATION_ORIGINS)
        return [str(item) for item in origins if str(item)]

    def _list_recommendations_from_database(
        self,
        *,
        date: Optional[str] = None,
        market: Optional[str] = None,
        status: Optional[str] = None,
        origins: Optional[Iterable[str]] = None,
    ) -> List[Recommendation]:
        if not self._database_enabled():
            return []

        origin_values = self._origin_filter(origins)
        try:
            with session_scope(self._resolved_database_url()) as session:
                query = select(RecommendationRecord)
                if origin_values:
                    query = query.where(RecommendationRecord.recommendation_origin.in_(origin_values))
                if date:
                    query = query.where(RecommendationRecord.game_date == date)
                if market:
                    query = query.where(RecommendationRecord.market == market)
                if status:
                    query = query.where(RecommendationRecord.status == status)
                query = query.order_by(RecommendationRecord.game_date, RecommendationRecord.edge.desc())
                rows = session.execute(query).scalars().all()
            return [self._record_to_recommendation(row) for row in rows]
        except SQLAlchemyError:
            return []

    def _list_recommendations_from_csv(
        self,
        *,
        date: Optional[str] = None,
        market: Optional[str] = None,
        status: Optional[str] = None,
        origins: Optional[Iterable[str]] = None,
    ) -> List[Recommendation]:
        df = self._load_edges()
        if df.empty:
            return []

        origin_values = self._origin_filter(origins)
        if origin_values and "recommendation_origin" in df.columns:
            df = df[df["recommendation_origin"].astype(str).isin(origin_values)]
        if date:
            df = df[df["game_date"].astype(str) == date]
        if market:
            df = df[df["market_key"].astype(str) == market]
        if status:
            df = df[df["market_readiness_status"].astype(str) == status]

        return [
            Recommendation(**row_to_recommendation_payload(row, data_timestamp=self._data_timestamp()))
            for _, row in df.iterrows()
        ]

    def list_recommendations(
        self,
        *,
        date: Optional[str] = None,
        market: Optional[str] = None,
        status: Optional[str] = None,
        origins: Optional[Iterable[str]] = None,
    ) -> List[Recommendation]:
        database_items = self._list_recommendations_from_database(date=date, market=market, status=status, origins=origins)
        if database_items:
            return database_items
        return self._list_recommendations_from_csv(date=date, market=market, status=status, origins=origins)

    def get_recommendation(self, recommendation_id: str, *, origins: Optional[Iterable[str]] = None) -> Optional[Recommendation]:
        origin_values = self._origin_filter(origins)
        if self._database_enabled():
            try:
                with session_scope(self._resolved_database_url()) as session:
                    query = select(RecommendationRecord).where(RecommendationRecord.id == recommendation_id)
                    if origin_values:
                        query = query.where(RecommendationRecord.recommendation_origin.in_(origin_values))
                    record = session.execute(query).scalar_one_or_none()
                if record is not None:
                    return self._record_to_recommendation(record)
            except SQLAlchemyError:
                pass

        df = self._load_edges()
        if df.empty:
            return None
        if origin_values and "recommendation_origin" in df.columns:
            df = df[df["recommendation_origin"].astype(str).isin(origin_values)]
        matches = df[df["recommendation_id"].astype(str) == recommendation_id]
        if matches.empty:
            return None
        return Recommendation(**row_to_recommendation_payload(matches.iloc[0], data_timestamp=self._data_timestamp()))

    def get_game(self, game_id: str) -> Optional[GameRecommendations]:
        recommendations = self.list_recommendations()
        matches = [item for item in recommendations if item.game_id == game_id]
        if not matches:
            return None

        first = matches[0]
        return GameRecommendations(
            id=first.game_id,
            game_date=first.game_date,
            home_team=first.home_team,
            away_team=first.away_team,
            recommendations=matches,
        )

    def get_slate(self, date: str) -> List[GameRecommendations]:
        recommendations = self.list_recommendations(date=date)
        games_by_id: Dict[str, List[Recommendation]] = {}
        for recommendation in recommendations:
            games_by_id.setdefault(recommendation.game_id, []).append(recommendation)

        games: List[GameRecommendations] = []
        for game_id, items in games_by_id.items():
            first = items[0]
            games.append(
                GameRecommendations(
                    id=game_id,
                    game_date=first.game_date,
                    home_team=first.home_team,
                    away_team=first.away_team,
                    recommendations=items,
                )
            )
        return games

    def get_market_readiness(self) -> List[MarketReadinessEntry]:
        if self._database_enabled():
            try:
                with session_scope(self._resolved_database_url()) as session:
                    query = select(MarketReadinessSnapshotRecord).order_by(MarketReadinessSnapshotRecord.market)
                    rows = session.execute(query).scalars().all()
                if rows:
                    return [
                        MarketReadinessEntry(
                            market=row.market,
                            status=row.status,
                            tier=row.tier,
                            label=row.label,
                            summary=row.summary,
                        )
                        for row in rows
                    ]
            except SQLAlchemyError:
                pass

        return [MarketReadinessEntry(market=market, **payload) for market, payload in DEFAULT_MARKET_READINESS.items()]
