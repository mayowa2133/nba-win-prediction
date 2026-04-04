"""Persist scored recommendation artifacts into the local warehouse."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import delete

from src.api.mapping import ensure_edge_identifiers, row_to_recommendation_payload
from src.contracts.market_readiness import DEFAULT_MARKET_READINESS
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import MarketReadinessSnapshotRecord, RecommendationRecord


def timestamp_for_path(path: Path) -> str:
    try:
        dt_obj = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return dt_obj.replace(microsecond=0).isoformat()
    except FileNotFoundError:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_market_readiness_rows(snapshot_path: Optional[Path] = None) -> List[dict]:
    if snapshot_path is not None and snapshot_path.exists():
        df = pd.read_csv(snapshot_path).fillna("")
        return df.to_dict(orient="records")

    return [
        {"market": market, **payload}
        for market, payload in DEFAULT_MARKET_READINESS.items()
    ]


def _persist_market_readiness(rows: Iterable[dict], *, as_of_timestamp: str, database_url: Optional[str]) -> int:
    with session_scope(database_url) as session:
        session.execute(delete(MarketReadinessSnapshotRecord))
        count = 0
        for row in rows:
            session.add(
                MarketReadinessSnapshotRecord(
                    market=str(row["market"]),
                    status=str(row["status"]),
                    tier=str(row["tier"]),
                    label=str(row["label"]),
                    summary=str(row["summary"]),
                    metrics_json=row.get("metrics_json"),
                    as_of_timestamp=as_of_timestamp,
                )
            )
            count += 1
    return count


def materialize_edges(
    edges_path: Path,
    *,
    database_url: Optional[str] = None,
    readiness_path: Optional[Path] = None,
    recommendation_origin: str = "live_daily",
    persist_readiness: bool = True,
) -> tuple[int, int]:
    edges_path = Path(edges_path)
    init_database(database_url)

    readiness_rows = load_market_readiness_rows(readiness_path)
    as_of_timestamp = timestamp_for_path(readiness_path or edges_path)

    if not edges_path.exists():
        readiness_count = 0
        if persist_readiness:
            readiness_count = _persist_market_readiness(
                readiness_rows,
                as_of_timestamp=as_of_timestamp,
                database_url=database_url,
            )
        return 0, readiness_count

    df = pd.read_csv(edges_path)
    if df.empty:
        readiness_count = 0
        if persist_readiness:
            readiness_count = _persist_market_readiness(
                readiness_rows,
                as_of_timestamp=as_of_timestamp,
                database_url=database_url,
            )
        return 0, readiness_count

    has_origin_column = "recommendation_origin" in df.columns
    normalized = ensure_edge_identifiers(df, data_timestamp=timestamp_for_path(edges_path))
    if has_origin_column:
        normalized["recommendation_origin"] = (
            normalized.get("recommendation_origin", pd.Series(dtype=str))
            .replace("", pd.NA)
            .fillna(recommendation_origin)
        )
    else:
        normalized["recommendation_origin"] = recommendation_origin
    game_dates = sorted({str(value) for value in normalized["game_date"].dropna().astype(str).tolist()})
    markets = sorted({str(value) for value in normalized["market_key"].dropna().astype(str).tolist()})
    origins = sorted({str(value) for value in normalized["recommendation_origin"].dropna().astype(str).tolist()})

    with session_scope(database_url) as session:
        if game_dates and markets:
            delete_query = delete(RecommendationRecord).where(
                RecommendationRecord.game_date.in_(game_dates),
                RecommendationRecord.market.in_(markets),
            )
            if origins:
                delete_query = delete_query.where(RecommendationRecord.recommendation_origin.in_(origins))
            session.execute(delete_query)

        for _, row in normalized.iterrows():
            payload = row_to_recommendation_payload(row, data_timestamp=timestamp_for_path(edges_path))
            session.add(
                RecommendationRecord(
                    id=str(payload["id"]),
                    game_id=str(payload["game_id"]),
                    player=str(payload.get("player") or ""),
                    game_date=str(payload["game_date"]),
                    home_team=str(payload["home_team"]),
                    away_team=str(payload["away_team"]),
                    market=str(payload["market"]),
                    selection=str(payload["selection"]),
                    sportsbook_line=float(payload["sportsbook_line"]),
                    sportsbook_odds=payload["sportsbook_odds"],
                    fair_line=float(payload["fair_line"]),
                    fair_odds=payload["fair_odds"],
                    edge=float(payload["edge"]),
                    selected_probability=payload.get("selected_probability"),
                    market_implied_probability=payload.get("market_implied_probability"),
                    confidence=str(payload["confidence"]),
                    status=str(payload["status"]),
                    model_version=str(payload["model_version"]),
                    data_timestamp=str(payload["data_timestamp"]),
                    recommendation_origin=str(payload.get("recommendation_origin") or recommendation_origin),
                    published_line=payload.get("published_line"),
                    published_odds=payload.get("published_odds"),
                    published_at=payload.get("published_at"),
                    likely_range_low=payload.get("likely_range_low"),
                    likely_range_high=payload.get("likely_range_high"),
                    likely_range_confidence=payload.get("likely_range_confidence"),
                    most_likely_milestone=payload.get("most_likely_milestone"),
                    most_likely_milestone_probability=payload.get("most_likely_milestone_probability"),
                    milestone_probabilities_json=payload.get("milestone_probabilities"),
                    quote_source_provider=payload.get("quote_source_provider"),
                    quote_source_mode=payload.get("quote_source_mode"),
                    quote_source_book=payload.get("quote_source_book"),
                    closing_line=payload.get("closing_line"),
                    closing_odds=payload.get("closing_odds"),
                    actual_value=payload.get("actual_value"),
                    result=payload.get("result"),
                    clv=payload.get("clv"),
                    roi=payload.get("roi"),
                    lineup_context_json=payload.get("lineup_context_json"),
                    injury_context_json=payload.get("injury_context_json"),
                    reasons_json=list(payload["reasons"]),
                    api_schema_version=str(payload["api_schema_version"]),
                )
            )

    readiness_count = 0
    if persist_readiness:
        readiness_count = _persist_market_readiness(
            readiness_rows,
            as_of_timestamp=as_of_timestamp,
            database_url=database_url,
        )
    return len(normalized), readiness_count
