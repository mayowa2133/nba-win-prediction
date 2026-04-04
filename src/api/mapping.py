"""Shared mapping helpers for recommendation artifacts."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import pandas as pd

from src.contracts.market_readiness import get_market_readiness
from src.contracts.versions import RECOMMENDATION_API_SCHEMA_VERSION
from src.utils.artifact_metadata import stable_id
from src.utils.betting import american_to_prob


def prob_to_american(probability: float) -> Optional[int]:
    if probability <= 0 or probability >= 1 or math.isnan(probability):
        return None
    if probability >= 0.5:
        return int(round(-100 * probability / (1 - probability)))
    return int(round(100 * (1 - probability) / probability))


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_optional_json(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def market_key_for_row(row: pd.Series) -> str:
    return str(row.get("market_key") or row.get("market") or "")


def selection_for_row(row: pd.Series) -> str:
    return str(row.get("best_side") or row.get("selection") or "")


def selected_probability(row: pd.Series) -> float:
    if "selected_probability" in row.index and not pd.isna(row.get("selected_probability")):
        return _to_float(row.get("selected_probability"))
    if selection_for_row(row) == "under":
        return _to_float(row.get("model_p_under"))
    return _to_float(row.get("model_p_over"))


def selected_odds(row: pd.Series) -> Optional[float]:
    for field in ("published_odds", "sportsbook_odds"):
        value = row.get(field)
        if value is not None and not pd.isna(value):
            return float(value)
    if row.get("best_side") == "under":
        value = row.get("under_odds_best")
    else:
        value = row.get("over_odds_best")
    if value is None or pd.isna(value):
        return None
    return float(value)


def sportsbook_line_value(row: pd.Series) -> float:
    for field in ("published_line", "sportsbook_line", "prop_pts_line"):
        value = row.get(field)
        if value is not None and not pd.isna(value):
            return float(value)
    return 0.0


def fair_line_value(row: pd.Series) -> float:
    for field in ("fair_line", "model_mean_pts"):
        value = row.get(field)
        if value is not None and not pd.isna(value):
            return float(value)
    return 0.0


def confidence_label(probability: float, edge: float, status: str) -> str:
    if status != "production":
        return "experimental"
    if probability >= 0.75 and edge >= 0.08:
        return "high"
    if probability >= 0.65 and edge >= 0.05:
        return "medium"
    return "low"


def _generic_reason_payloads(row: pd.Series, readiness_summary: str) -> List[Dict[str, str]]:
    market = market_key_for_row(row)
    selection = selection_for_row(row)
    selected_prob = selected_probability(row)
    line = sportsbook_line_value(row)
    fair_line = fair_line_value(row)
    edge = _to_float(row.get("edge", row.get("best_edge")))
    implied = _to_float(row.get("market_implied_probability"), default=float("nan"))
    reasons = []

    if market == "game_moneyline":
        reasons.append(
            {
                "label": "Win probability",
                "detail": (
                    f"Model gives {selection} a {selected_prob:.3f} win probability versus "
                    f"market implied {implied:.3f}."
                ),
            }
        )
    else:
        reasons.append(
            {
                "label": "Model vs line",
                "detail": (
                    f"Model fair line is {fair_line:.2f} against a sportsbook line of "
                    f"{line:.2f} for the {selection} side."
                ),
            }
        )
        reasons.append(
            {
                "label": "Cover probability",
                "detail": (
                    f"Selected side '{selection}' carries model probability {selected_prob:.3f} "
                    f"and edge {edge:+.3f}."
                ),
            }
        )

    lineup_context = _parse_optional_json(row.get("lineup_context_json"))
    if isinstance(lineup_context, dict) and lineup_context:
        returning = lineup_context.get("projected_returning_starters")
        replacements = lineup_context.get("projected_replacements")
        if returning is not None or replacements is not None:
            reasons.append(
                {
                    "label": "Lineup context",
                    "detail": (
                        f"Projected returning starters: {int(returning or 0)}; "
                        f"projected replacements: {int(replacements or 0)}."
                    ),
                }
            )

    injury_context = _parse_optional_json(row.get("injury_context_json"))
    if isinstance(injury_context, dict) and injury_context:
        if injury_context.get("summary"):
            reasons.append({"label": "Injury context", "detail": str(injury_context["summary"])})

    captured_at = row.get("market_snapshot_at")
    if captured_at is not None and not pd.isna(captured_at):
        reasons.append(
            {
                "label": "Market snapshot",
                "detail": f"Recommendation uses the latest stored market snapshot from {captured_at}.",
            }
        )

    if readiness_summary:
        reasons.append({"label": "Market status", "detail": readiness_summary})
    return reasons


def build_reason_payloads(row: pd.Series, readiness_summary: str) -> List[Dict[str, str]]:
    existing = _parse_optional_json(row.get("reasons"))
    if isinstance(existing, list) and existing:
        return [
            {"label": str(item.get("label", "")), "detail": str(item.get("detail", ""))}
            for item in existing
        ]
    existing_json = _parse_optional_json(row.get("reasons_json"))
    if isinstance(existing_json, list) and existing_json:
        return [
            {"label": str(item.get("label", "")), "detail": str(item.get("detail", ""))}
            for item in existing_json
        ]

    market = market_key_for_row(row)
    if market.startswith("game_"):
        return _generic_reason_payloads(row, readiness_summary)

    fair_delta = fair_line_value(row) - sportsbook_line_value(row)
    selected_prob = selected_probability(row)
    reasons = [
        {
            "label": "Model vs line",
            "detail": (
                f"Model projects {fair_line_value(row):.2f} against a line of "
                f"{sportsbook_line_value(row):.2f} ({fair_delta:+.2f})."
            ),
        },
        {
            "label": "Win probability",
            "detail": (
                f"Selected side '{selection_for_row(row)}' carries model probability "
                f"{selected_prob:.3f} and edge {_to_float(row.get('best_edge', row.get('edge'))):.3f}."
            ),
        },
    ]

    if pd.notna(row.get("days_rest_used")):
        reasons.append(
            {
                "label": "Rest context",
                "detail": f"Rest input for this projection is {_to_float(row.get('days_rest_used')):.0f} day(s).",
            }
        )

    if pd.notna(row.get("is_home_used")):
        venue = "home" if _to_float(row.get("is_home_used")) >= 0.5 else "away"
        reasons.append({"label": "Venue", "detail": f"The upcoming matchup is modeled as a {venue} game."})

    if readiness_summary:
        reasons.append({"label": "Market status", "detail": readiness_summary})

    return reasons


def ensure_edge_identifiers(df: pd.DataFrame, *, data_timestamp: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    normalized = df.copy()
    if "market_key" not in normalized.columns and "market" in normalized.columns:
        normalized["market_key"] = normalized["market"]
    if "best_side" not in normalized.columns and "selection" in normalized.columns:
        normalized["best_side"] = normalized["selection"]
    if "player" not in normalized.columns:
        normalized["player"] = ""

    if "recommendation_id" not in normalized.columns:
        normalized["recommendation_id"] = normalized.apply(
            lambda row: stable_id(
                row.get("player"),
                row.get("market_key") or row.get("market"),
                row.get("game_date"),
                row.get("home_team"),
                row.get("away_team"),
                row.get("prop_pts_line", row.get("sportsbook_line")),
                row.get("best_side", row.get("selection")),
                prefix="rec",
            ),
            axis=1,
        )
    if "game_id" not in normalized.columns:
        normalized["game_id"] = normalized.apply(
            lambda row: stable_id(
                row.get("game_date"),
                row.get("home_team"),
                row.get("away_team"),
                prefix="game",
            ),
            axis=1,
        )
    if "generated_at_utc" not in normalized.columns:
        normalized["generated_at_utc"] = data_timestamp
    if "model_version" not in normalized.columns:
        normalized["model_version"] = "unknown"
    if "recommendation_origin" not in normalized.columns:
        normalized["recommendation_origin"] = "live_daily"
    if "market_readiness_status" not in normalized.columns:
        normalized["market_readiness_status"] = normalized["market_key"].map(
            lambda key: get_market_readiness(str(key)).get("status", "experimental")
        )

    return normalized


def row_to_recommendation_payload(row: pd.Series, *, data_timestamp: Optional[str] = None) -> Dict[str, object]:
    market = market_key_for_row(row)
    readiness = get_market_readiness(market)
    status = str(row.get("market_readiness_status", readiness["status"]))
    selected_prob = selected_probability(row)
    sportsbook_odds = selected_odds(row)
    implied_probability = row.get("market_implied_probability")
    if (implied_probability is None or pd.isna(implied_probability)) and sportsbook_odds is not None:
        implied_probability = american_to_prob(float(sportsbook_odds))
    confidence = str(row.get("confidence") or confidence_label(selected_prob, _to_float(row.get("best_edge", row.get("edge"))), status))
    lineup_context = _parse_optional_json(row.get("lineup_context_json"))
    injury_context = _parse_optional_json(row.get("injury_context_json"))
    published_line = row.get("published_line")
    if published_line is None or pd.isna(published_line):
        published_line = sportsbook_line_value(row)
    published_odds = row.get("published_odds")
    if published_odds is None or pd.isna(published_odds):
        published_odds = sportsbook_odds

    return {
        "id": str(row["recommendation_id"]),
        "game_id": str(row["game_id"]),
        "player": None if str(row.get("player", "")).strip() == "" else str(row.get("player")),
        "game_date": str(row.get("game_date")),
        "home_team": str(row.get("home_team")),
        "away_team": str(row.get("away_team")),
        "market": market,
        "selection": selection_for_row(row),
        "sportsbook_line": sportsbook_line_value(row),
        "sportsbook_odds": sportsbook_odds,
        "fair_line": fair_line_value(row),
        "fair_odds": row.get("fair_odds") if not pd.isna(row.get("fair_odds")) else prob_to_american(selected_prob),
        "edge": _to_float(row.get("best_edge", row.get("edge"))),
        "selected_probability": selected_prob,
        "market_implied_probability": None if pd.isna(implied_probability) else float(implied_probability),
        "confidence": confidence,
        "status": status,
        "model_version": str(row.get("model_version", "unknown")),
        "data_timestamp": str(row.get("generated_at_utc", data_timestamp or "")),
        "recommendation_origin": str(row.get("recommendation_origin", "live_daily") or "live_daily"),
        "reasons": build_reason_payloads(row, readiness["summary"]),
        "published_line": None if pd.isna(published_line) else float(published_line),
        "published_odds": None if pd.isna(published_odds) else float(published_odds),
        "published_at": str(row.get("published_at", row.get("generated_at_utc", data_timestamp or ""))),
        "quote_source_provider": None if pd.isna(row.get("quote_source_provider")) else str(row.get("quote_source_provider")),
        "quote_source_mode": None if pd.isna(row.get("quote_source_mode")) else str(row.get("quote_source_mode")),
        "quote_source_book": None if pd.isna(row.get("quote_source_book")) else str(row.get("quote_source_book")),
        "closing_line": None if pd.isna(row.get("closing_line")) else float(row.get("closing_line")),
        "closing_odds": None if pd.isna(row.get("closing_odds")) else float(row.get("closing_odds")),
        "actual_value": None if pd.isna(row.get("actual_value")) else float(row.get("actual_value")),
        "result": None if pd.isna(row.get("result")) else str(row.get("result")),
        "clv": None if pd.isna(row.get("clv")) else float(row.get("clv")),
        "roi": None if pd.isna(row.get("roi")) else float(row.get("roi")),
        "lineup_context_json": lineup_context,
        "injury_context_json": injury_context,
        "api_schema_version": RECOMMENDATION_API_SCHEMA_VERSION,
    }
