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


class MobileGameSummary(BaseModel):
    id: str
    game_date: str
    commence_time: Optional[str] = None
    home_team: str
    away_team: str
    recommendation_count: int
    top_recommendation: Recommendation


class MobileParlaySuggestion(BaseModel):
    id: str
    title: str
    combined_odds: int
    combined_probability: float
    recommendations: List[Recommendation] = Field(default_factory=list)


class MobileInjuryEntry(BaseModel):
    player_name: str
    team_abbrev: Optional[str] = None
    report_status: str
    normalized_status: Optional[str] = None
    projected_availability: Optional[str] = None
    raw_reason: Optional[str] = None
    reported_at: Optional[str] = None


class MobileLineupStarter(BaseModel):
    player_name: str
    projected_position: Optional[str] = None
    starter_probability: float
    injury_status: Optional[str] = None
    projection_reason: str


class MobileTeamLineupSummary(BaseModel):
    team_abbrev: str
    projected_returning_starters: Optional[int] = None
    projected_replacements: Optional[int] = None
    starters: List[MobileLineupStarter] = Field(default_factory=list)


class MobileTrendPoint(BaseModel):
    game_date: str
    label: str
    cumulative_roi: float


class MobileHomeResponse(BaseModel):
    selected_date: str = ""
    available_dates: List[str] = Field(default_factory=list)
    featured_recommendations: List[Recommendation] = Field(default_factory=list)
    games: List[MobileGameSummary] = Field(default_factory=list)
    trending_parlays: List[MobileParlaySuggestion] = Field(default_factory=list)
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION


class MobileGameDetailResponse(BaseModel):
    id: str
    game_date: str
    commence_time: Optional[str] = None
    home_team: str
    away_team: str
    recommendations: List[Recommendation]
    injuries: List[MobileInjuryEntry] = Field(default_factory=list)
    lineup_summary: List[MobileTeamLineupSummary] = Field(default_factory=list)
    game_status: Optional[Dict[str, Any]] = None
    api_schema_version: str = RECOMMENDATION_API_SCHEMA_VERSION


class MobileTrendsResponse(BaseModel):
    roi: float = 0.0
    clv: float = 0.0
    hit_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    recent_settlements: List[Recommendation] = Field(default_factory=list)
    chart_points: List[MobileTrendPoint] = Field(default_factory=list)
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
