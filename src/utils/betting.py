"""Betting math helpers shared across ingest, scoring, and evaluation."""

from __future__ import annotations

import math
from typing import Optional


def american_to_prob(odds: Optional[float]) -> float:
    if odds is None or math.isnan(float(odds)):
        return float("nan")
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def remove_vig_two_way(
    home_odds: Optional[float],
    away_odds: Optional[float],
) -> tuple[float, float]:
    home_prob = american_to_prob(home_odds)
    away_prob = american_to_prob(away_odds)
    if math.isnan(home_prob) or math.isnan(away_prob):
        return float("nan"), float("nan")
    total = home_prob + away_prob
    if total <= 0:
        return float("nan"), float("nan")
    return home_prob / total, away_prob / total


def american_profit_for_unit(odds: Optional[float]) -> float:
    if odds is None or math.isnan(float(odds)):
        return 0.0
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def roi_for_result(result: str, odds: Optional[float]) -> float:
    result = str(result or "").lower()
    if result == "win":
        return american_profit_for_unit(odds)
    if result == "push":
        return 0.0
    return -1.0


def side_friendly_line(line_value: Optional[float], selection: str, market: str) -> float:
    line = float(line_value or 0.0)
    selection = str(selection or "").lower()
    market = str(market or "").lower()

    if market == "game_spread":
        return -line if selection == "home" else line
    if market in {"game_total", "player_points", "player_rebounds", "player_assists", "player_threes"}:
        return line if selection == "over" else -line
    return 0.0


def clv_price_component(published_odds: Optional[float], closing_odds: Optional[float]) -> float:
    published_prob = american_to_prob(published_odds)
    closing_prob = american_to_prob(closing_odds)
    if math.isnan(published_prob) or math.isnan(closing_prob):
        return 0.0
    return closing_prob - published_prob
