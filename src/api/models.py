"""Pydantic models for the beta recommendation API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.contracts.versions import RECOMMENDATION_API_SCHEMA_VERSION


class Reason(BaseModel):
    label: str
    detail: str


class MilestoneProbability(BaseModel):
    threshold: float
    probability: float
    fair_odds: Optional[int] = None
    line_equivalent: Optional[float] = None


class Recommendation(BaseModel):
    id: str
    game_id: str
    player: Optional[str] = None
    game_date: str
    home_team: str
    away_team: str
    market: str
    selection: str
    sportsbook_line: float
    sportsbook_odds: Optional[float] = None
    fair_line: float
    fair_odds: Optional[int] = None
    edge: float
    selected_probability: Optional[float] = None
    market_implied_probability: Optional[float] = None
    confidence: str
    status: str
    model_version: str
    data_timestamp: str
    published_line: Optional[float] = None
    published_odds: Optional[float] = None
    published_at: Optional[str] = None
    likely_range_low: Optional[float] = None
    likely_range_high: Optional[float] = None
    likely_range_confidence: Optional[float] = None
    most_likely_milestone: Optional[float] = None
    most_likely_milestone_probability: Optional[float] = None
    milestone_probabilities: List[MilestoneProbability] = Field(default_factory=list)
    closing_line: Optional[float] = None
    closing_odds: Optional[float] = None
    actual_value: Optional[float] = None
    result: Optional[str] = None
    clv: Optional[float] = None
    roi: Optional[float] = None
    lineup_context_json: Optional[Dict[str, Any]] = None
    injury_context_json: Optional[Dict[str, Any]] = None
    reasons: List[Reason] = Field(default_factory=list)
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION


class RecommendationListResponse(BaseModel):
    items: List[Recommendation]
    count: int
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION


class GameRecommendations(BaseModel):
    id: str
    game_date: str
    home_team: str
    away_team: str
    recommendations: List[Recommendation]
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION


class SlateResponse(BaseModel):
    date: str
    games: List[GameRecommendations]
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION


class MarketReadinessEntry(BaseModel):
    market: str
    status: str
    tier: str
    label: str
    summary: str


class MarketReadinessResponse(BaseModel):
    items: List[MarketReadinessEntry]
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION
