"""Repository layer for precomputed recommendation artifacts."""

from __future__ import annotations

import os
from collections import defaultdict
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
    MobileGameDetailResponse,
    MobileGameSummary,
    MobileHomeResponse,
    MobileInjuryEntry,
    MobileLineupStarter,
    MobileParlaySuggestion,
    MobileTeamLineupSummary,
    MobileTrendPoint,
    MobileTrendsResponse,
    Recommendation,
)
from src.contracts.market_readiness import DEFAULT_MARKET_READINESS
from src.utils.nba_teams import canonical_team_abbrev
from src.warehouse.db import DEFAULT_DATABASE_URL, default_sqlite_database_path, session_scope
from src.warehouse.models import (
    ClosingLineRecord,
    GameOddsSnapshotRecord,
    InjuryReportRecord,
    LineupProjectionRecord,
    MarketReadinessSnapshotRecord,
    RecommendationRecord,
)


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

    @staticmethod
    def _recommendation_score(item: Recommendation) -> float:
        return float(item.edge * 100.0) + float(item.selected_probability or 0.0)

    def _sorted_recommendations(self, items: Iterable[Recommendation]) -> List[Recommendation]:
        return sorted(
            items,
            key=lambda item: (
                self._recommendation_score(item),
                float(item.edge),
                float(item.selected_probability or 0.0),
                str(item.published_at or item.data_timestamp),
            ),
            reverse=True,
        )

    @staticmethod
    def _decimal_odds(american: Optional[float]) -> float:
        if american in (None, 0):
            return 1.0
        value = float(american)
        return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))

    @staticmethod
    def _american_odds(decimal: float) -> int:
        if decimal <= 1:
            return 0
        if decimal >= 2:
            return int(round((decimal - 1) * 100))
        return int(round(-100 / (decimal - 1)))

    def _combined_parlay_odds(self, picks: Iterable[Recommendation]) -> int:
        decimal = 1.0
        count = 0
        for recommendation in picks:
            decimal *= self._decimal_odds(recommendation.sportsbook_odds)
            count += 1
        if count == 0:
            return 0
        return self._american_odds(decimal)

    @staticmethod
    def _combined_probability(picks: Iterable[Recommendation]) -> float:
        probability = 1.0
        count = 0
        for recommendation in picks:
            probability *= float(recommendation.selected_probability or 0.5)
            count += 1
        return probability if count else 0.0

    def _available_dates(self, items: Iterable[Recommendation]) -> List[str]:
        return sorted({item.game_date for item in items if item.game_date}, reverse=True)

    def _select_date(self, date: Optional[str], available_dates: List[str]) -> str:
        if date and date in available_dates:
            return date
        if available_dates:
            return available_dates[0]
        return date or ""

    def _load_commence_time_map(self, *, date: Optional[str] = None) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if self._database_enabled():
            try:
                with session_scope(self._resolved_database_url()) as session:
                    snapshot_query = select(
                        GameOddsSnapshotRecord.game_id,
                        GameOddsSnapshotRecord.commence_time,
                        GameOddsSnapshotRecord.captured_at,
                    )
                    if date:
                        snapshot_query = snapshot_query.where(GameOddsSnapshotRecord.game_date == date)
                    snapshot_rows = session.execute(
                        snapshot_query.order_by(GameOddsSnapshotRecord.captured_at.desc())
                    ).all()

                    closing_query = select(
                        ClosingLineRecord.game_id,
                        ClosingLineRecord.commence_time,
                        ClosingLineRecord.captured_at,
                    )
                    if date:
                        closing_query = closing_query.where(ClosingLineRecord.game_date == date)
                    closing_rows = session.execute(
                        closing_query.order_by(ClosingLineRecord.captured_at.desc())
                    ).all()

                for game_id, commence_time, _captured_at in [*snapshot_rows, *closing_rows]:
                    if not game_id or not commence_time:
                        continue
                    game_key = str(game_id)
                    if game_key not in mapping:
                        mapping[game_key] = str(commence_time)
            except SQLAlchemyError:
                mapping = {}

        if mapping:
            return mapping

        df = self._load_edges()
        if df.empty or "commence_time" not in df.columns:
            return {}

        normalized = ensure_edge_identifiers(df, data_timestamp=self._data_timestamp())
        if date:
            normalized = normalized[normalized["game_date"].astype(str) == date]
        for _, row in normalized.iterrows():
            game_id = row.get("game_id")
            commence_time = row.get("commence_time")
            if not game_id or not commence_time or pd.isna(commence_time):
                continue
            mapping.setdefault(str(game_id), str(commence_time))
        return mapping

    def _load_injuries_for_game(
        self,
        *,
        game_id: str,
        game_date: str,
        home_team: str,
        away_team: str,
    ) -> List[MobileInjuryEntry]:
        if not self._database_enabled():
            return []

        team_abbrevs = {
            canonical_team_abbrev(home_team) or home_team,
            canonical_team_abbrev(away_team) or away_team,
        }
        try:
            with session_scope(self._resolved_database_url()) as session:
                query = (
                    select(InjuryReportRecord)
                    .where(InjuryReportRecord.game_date == game_date)
                    .order_by(InjuryReportRecord.reported_at.desc(), InjuryReportRecord.player_name)
                )
                rows = session.execute(query).scalars().all()
        except SQLAlchemyError:
            return []

        items: List[MobileInjuryEntry] = []
        seen: set[tuple[str, str]] = set()
        for record in rows:
            team_abbrev = str(record.team_abbrev or "")
            if team_abbrev and team_abbrev not in team_abbrevs:
                continue
            if record.game_id and str(record.game_id) != game_id and team_abbrev not in team_abbrevs:
                continue
            if record.row_kind != "player_status" or not str(record.player_name or "").strip():
                continue
            key = (team_abbrev, str(record.player_name))
            if key in seen:
                continue
            seen.add(key)
            items.append(
                MobileInjuryEntry(
                    player_name=str(record.player_name),
                    team_abbrev=team_abbrev or None,
                    report_status=str(record.report_status),
                    normalized_status=record.normalized_status,
                    projected_availability=record.projected_availability,
                    raw_reason=record.raw_reason,
                    reported_at=record.reported_at,
                )
            )
        return items

    def _lineup_context_counts(
        self,
        *,
        recommendations: Iterable[Recommendation],
        home_team: str,
        away_team: str,
    ) -> Dict[str, Dict[str, Optional[int]]]:
        home_abbrev = canonical_team_abbrev(home_team) or home_team
        away_abbrev = canonical_team_abbrev(away_team) or away_team
        for recommendation in recommendations:
            context = recommendation.lineup_context_json or {}
            if not context:
                continue
            home_returning = context.get("home_projected_returning_starters")
            away_returning = context.get("away_projected_returning_starters")
            counts: Dict[str, Dict[str, Optional[int]]] = {}
            if home_returning is not None:
                counts[home_abbrev] = {
                    "projected_returning_starters": int(round(float(home_returning))),
                    "projected_replacements": None,
                }
            if away_returning is not None:
                counts[away_abbrev] = {
                    "projected_returning_starters": int(round(float(away_returning))),
                    "projected_replacements": None,
                }
            if counts:
                return counts
        return {}

    def _load_lineup_summary_for_game(
        self,
        *,
        game_id: str,
        game_date: str,
        home_team: str,
        away_team: str,
        recommendations: Iterable[Recommendation],
    ) -> List[MobileTeamLineupSummary]:
        team_abbrevs = [
            canonical_team_abbrev(away_team) or away_team,
            canonical_team_abbrev(home_team) or home_team,
        ]
        grouped: Dict[str, List[LineupProjectionRecord]] = defaultdict(list)
        if self._database_enabled():
            try:
                with session_scope(self._resolved_database_url()) as session:
                    query = (
                        select(LineupProjectionRecord)
                        .where(LineupProjectionRecord.game_date == game_date)
                        .where(LineupProjectionRecord.team_abbrev.in_(team_abbrevs))
                        .order_by(
                            LineupProjectionRecord.team_abbrev,
                            LineupProjectionRecord.starter_probability.desc(),
                            LineupProjectionRecord.projected_position,
                            LineupProjectionRecord.projected_starter,
                        )
                    )
                    rows = session.execute(query).scalars().all()
            except SQLAlchemyError:
                rows = []

            for row in rows:
                if row.game_id and str(row.game_id) != game_id:
                    continue
                grouped[str(row.team_abbrev)].append(row)

        context_counts = self._lineup_context_counts(
            recommendations=recommendations,
            home_team=home_team,
            away_team=away_team,
        )

        summary: List[MobileTeamLineupSummary] = []
        for team_abbrev in team_abbrevs:
            starters = [
                MobileLineupStarter(
                    player_name=str(row.projected_starter),
                    projected_position=row.projected_position,
                    starter_probability=float(row.starter_probability),
                    injury_status=row.injury_status,
                    projection_reason=str(row.projection_reason),
                )
                for row in grouped.get(team_abbrev, [])[:5]
            ]
            context = context_counts.get(team_abbrev, {})
            projected_returning = context.get("projected_returning_starters")
            projected_replacements = context.get("projected_replacements")
            if projected_replacements is None and projected_returning is not None and starters:
                projected_replacements = max(0, len(starters) - int(projected_returning))
            if starters or context:
                summary.append(
                    MobileTeamLineupSummary(
                        team_abbrev=team_abbrev,
                        projected_returning_starters=projected_returning,
                        projected_replacements=projected_replacements,
                        starters=starters,
                    )
                )
        return summary

    def _build_mobile_game_summary(
        self,
        *,
        items: List[Recommendation],
        commence_time: Optional[str],
    ) -> MobileGameSummary:
        ranked = self._sorted_recommendations(items)
        first = ranked[0]
        return MobileGameSummary(
            id=first.game_id,
            game_date=first.game_date,
            commence_time=commence_time,
            home_team=first.home_team,
            away_team=first.away_team,
            recommendation_count=len(ranked),
            top_recommendation=first,
        )

    def _build_trending_parlays(self, recommendations: List[Recommendation]) -> List[MobileParlaySuggestion]:
        ranked = self._sorted_recommendations(recommendations)
        if len(ranked) < 2:
            return []

        distinct_two: List[Recommendation] = []
        seen_games: set[str] = set()
        for item in ranked:
            if item.game_id in seen_games:
                continue
            distinct_two.append(item)
            seen_games.add(item.game_id)
            if len(distinct_two) == 2:
                break
        if len(distinct_two) < 2:
            distinct_two = ranked[:2]

        three_leg = ranked[: min(3, len(ranked))]
        suggestions: List[MobileParlaySuggestion] = [
            MobileParlaySuggestion(
                id="value-two-leg",
                title="2-Leg Value",
                combined_odds=self._combined_parlay_odds(distinct_two),
                combined_probability=self._combined_probability(distinct_two),
                recommendations=distinct_two,
            )
        ]
        if len(three_leg) >= 3:
            suggestions.append(
                MobileParlaySuggestion(
                    id="aggressive-three-leg",
                    title="3-Leg Moonshot",
                    combined_odds=self._combined_parlay_odds(three_leg),
                    combined_probability=self._combined_probability(three_leg),
                    recommendations=three_leg,
                )
            )
        return suggestions

    def get_mobile_home(
        self,
        *,
        date: Optional[str] = None,
        market: Optional[str] = None,
        confidence: Optional[str] = None,
        origins: Optional[Iterable[str]] = None,
    ) -> MobileHomeResponse:
        market_value = None if not market or market == "all" else market
        recommendations = self.list_recommendations(date=None, market=market_value, status=None, origins=origins)
        if confidence and confidence not in {"", "all"}:
            recommendations = [
                item for item in recommendations if item.confidence.lower() == confidence.lower()
            ]

        available_dates = self._available_dates(recommendations)
        selected_date = self._select_date(date, available_dates)
        current = [item for item in recommendations if not selected_date or item.game_date == selected_date]
        featured = self._sorted_recommendations(current)[:6]

        commence_time_map = self._load_commence_time_map(date=selected_date or None)
        games_by_id: Dict[str, List[Recommendation]] = defaultdict(list)
        for recommendation in current:
            games_by_id[recommendation.game_id].append(recommendation)

        games = [
            self._build_mobile_game_summary(
                items=items,
                commence_time=commence_time_map.get(game_id),
            )
            for game_id, items in games_by_id.items()
        ]
        games = sorted(games, key=lambda item: self._recommendation_score(item.top_recommendation), reverse=True)

        return MobileHomeResponse(
            selected_date=selected_date,
            available_dates=available_dates,
            featured_recommendations=featured,
            games=games,
            trending_parlays=self._build_trending_parlays(current),
        )

    def get_mobile_game_detail(self, game_id: str, *, origins: Optional[Iterable[str]] = None) -> Optional[MobileGameDetailResponse]:
        game = self.get_game(game_id)
        if game is None:
            return None

        commence_time = self._load_commence_time_map(date=game.game_date).get(game_id)
        injuries = self._load_injuries_for_game(
            game_id=game.id,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
        )
        lineup_summary = self._load_lineup_summary_for_game(
            game_id=game.id,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
            recommendations=game.recommendations,
        )
        return MobileGameDetailResponse(
            id=game.id,
            game_date=game.game_date,
            commence_time=commence_time,
            home_team=game.home_team,
            away_team=game.away_team,
            recommendations=self._sorted_recommendations(game.recommendations),
            injuries=injuries,
            lineup_summary=lineup_summary,
        )

    def get_mobile_trends(self, *, origins: Optional[Iterable[str]] = None) -> MobileTrendsResponse:
        recommendations = self.list_recommendations(origins=origins)
        settled = [
            item for item in recommendations if item.result is not None or item.roi is not None or item.clv is not None
        ]
        settled = sorted(
            settled,
            key=lambda item: (
                str(item.game_date),
                str(item.published_at or item.data_timestamp),
                item.id,
            ),
            reverse=True,
        )

        wins = sum(1 for item in settled if (item.result or "").lower() == "win")
        losses = sum(1 for item in settled if (item.result or "").lower() == "loss")
        pushes = sum(1 for item in settled if (item.result or "").lower() == "push")
        roi_values = [float(item.roi) for item in settled if item.roi is not None]
        clv_values = [float(item.clv) for item in settled if item.clv is not None]
        hit_rate = float(wins / (wins + losses)) if (wins + losses) else 0.0

        roi_by_date: Dict[str, float] = defaultdict(float)
        for item in sorted(settled, key=lambda recommendation: (recommendation.game_date, recommendation.id)):
            if item.roi is None:
                continue
            roi_by_date[item.game_date] += float(item.roi)

        chart_points: List[MobileTrendPoint] = []
        running_roi = 0.0
        for game_date in sorted(roi_by_date):
            running_roi += roi_by_date[game_date]
            chart_points.append(
                MobileTrendPoint(
                    game_date=game_date,
                    label=game_date[5:] if len(game_date) >= 10 else game_date,
                    cumulative_roi=running_roi,
                )
            )

        return MobileTrendsResponse(
            roi=sum(roi_values) / len(roi_values) if roi_values else 0.0,
            clv=sum(clv_values) / len(clv_values) if clv_values else 0.0,
            hit_rate=hit_rate,
            wins=wins,
            losses=losses,
            pushes=pushes,
            recent_settlements=settled[:5],
            chart_points=chart_points,
        )

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
