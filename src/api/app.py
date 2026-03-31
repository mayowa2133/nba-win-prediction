"""FastAPI app for serving precomputed recommendation artifacts."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.api.models import (
    MarketReadinessResponse,
    RecommendationListResponse,
    Recommendation,
    SlateResponse,
)
from src.api.repository import RecommendationRepository


def create_app() -> FastAPI:
    app = FastAPI(
        title="NBA Betting Beta API",
        version="1.0.0",
        description="Serve precomputed NBA betting recommendations for the private beta app.",
    )
    repository = RecommendationRepository()

    @app.get("/v1/recommendations", response_model=RecommendationListResponse)
    def list_recommendations(
        date: str | None = Query(default=None),
        market: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ) -> RecommendationListResponse:
        items = repository.list_recommendations(date=date, market=market, status=status)
        return RecommendationListResponse(items=items, count=len(items))

    @app.get("/v1/recommendations/{recommendation_id}", response_model=Recommendation)
    def get_recommendation(recommendation_id: str) -> Recommendation:
        item = repository.get_recommendation(recommendation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        return item

    @app.get("/v1/games/{game_id}")
    def get_game(game_id: str):
        game = repository.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        return game

    @app.get("/v1/slates/{date}", response_model=SlateResponse)
    def get_slate(date: str) -> SlateResponse:
        games = repository.get_slate(date)
        return SlateResponse(date=date, games=games)

    @app.get("/v1/markets/readiness", response_model=MarketReadinessResponse)
    def get_market_readiness() -> MarketReadinessResponse:
        items = repository.get_market_readiness()
        return MarketReadinessResponse(items=items)

    return app


app = create_app()
