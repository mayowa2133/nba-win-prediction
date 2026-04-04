"""Current-day NBA game odds collection from The Odds API."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev


BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "basketball_nba"
DEFAULT_REGIONS = "us"
DEFAULT_BOOKMAKERS = ("draftkings", "fanduel")


def get_the_odds_api_key(cli_key: Optional[str]) -> str:
    api_key = cli_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required for live game odds collection")
    return api_key


def request_json(endpoint: str, *, params: dict) -> object:
    response = requests.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    return response.json()


def fetch_current_game_odds_snapshots(
    *,
    report_date: date,
    api_key: str,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
    sport_key: str = DEFAULT_SPORT_KEY,
    regions: str = DEFAULT_REGIONS,
) -> pd.DataFrame:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": ",".join(bookmakers),
    }
    payload = request_json(f"/sports/{sport_key}/odds", params=params)
    return build_game_odds_snapshot_frame(
        payload if isinstance(payload, list) else [],
        report_date=report_date,
        source_url=f"{BASE_URL}/sports/{sport_key}/odds",
    )


def build_game_odds_snapshot_frame(
    payloads: list[dict],
    *,
    report_date: date,
    source_url: str,
) -> pd.DataFrame:
    rows = []
    report_date_str = report_date.isoformat()
    for event in payloads:
        game_date = str(event.get("commence_time") or "")[:10]
        if game_date != report_date_str:
            continue

        fixture_id = str(event.get("id") or "")
        home_team = str(event.get("home_team") or "")
        away_team = str(event.get("away_team") or "")
        home_abbrev = canonical_team_abbrev(home_team) or home_team
        away_abbrev = canonical_team_abbrev(away_team) or away_team
        game_id = stable_id(game_date, home_abbrev, away_abbrev, prefix="game")
        commence_time = str(event.get("commence_time") or "")

        for bookmaker in event.get("bookmakers") or []:
            sportsbook = str(bookmaker.get("key") or "")
            default_captured_at = str(bookmaker.get("last_update") or commence_time)
            for market in bookmaker.get("markets") or []:
                market_key = str(market.get("key") or "")
                captured_at = str(market.get("last_update") or default_captured_at)
                if market_key not in {"h2h", "spreads", "totals"}:
                    continue
                for outcome in market.get("outcomes") or []:
                    price = outcome.get("price")
                    line_value = outcome.get("point")
                    name = str(outcome.get("name") or "").strip()
                    if market_key == "h2h":
                        side = "home" if name == home_team else "away" if name == away_team else ""
                        if not side:
                            continue
                        rows.append(
                            {
                                "fixture_id": fixture_id,
                                "game_id": game_id,
                                "game_date": game_date,
                                "commence_time": commence_time,
                                "home_team": canonical_team_name_from_abbrev(home_abbrev) or home_team,
                                "away_team": canonical_team_name_from_abbrev(away_abbrev) or away_team,
                                "market": "game_moneyline",
                                "side": side,
                                "sportsbook": sportsbook,
                                "bookmaker_id": None,
                                "line_value": 0.0,
                                "price": float(price) if price is not None else None,
                                "market_id": None,
                                "market_name": "h2h",
                                "is_historical": 0,
                                "source_url": source_url,
                                "snapshot_type": "intraday",
                                "captured_at": captured_at,
                            }
                        )
                    elif market_key == "spreads":
                        side = "home" if name == home_team else "away" if name == away_team else ""
                        if not side:
                            continue
                        rows.append(
                            {
                                "fixture_id": fixture_id,
                                "game_id": game_id,
                                "game_date": game_date,
                                "commence_time": commence_time,
                                "home_team": canonical_team_name_from_abbrev(home_abbrev) or home_team,
                                "away_team": canonical_team_name_from_abbrev(away_abbrev) or away_team,
                                "market": "game_spread",
                                "side": side,
                                "sportsbook": sportsbook,
                                "bookmaker_id": None,
                                "line_value": abs(float(line_value)) if line_value is not None else 0.0,
                                "price": float(price) if price is not None else None,
                                "market_id": None,
                                "market_name": "spreads",
                                "is_historical": 0,
                                "source_url": source_url,
                                "snapshot_type": "intraday",
                                "captured_at": captured_at,
                            }
                        )
                    elif market_key == "totals":
                        side = str(outcome.get("name") or "").strip().lower()
                        if side not in {"over", "under"}:
                            continue
                        rows.append(
                            {
                                "fixture_id": fixture_id,
                                "game_id": game_id,
                                "game_date": game_date,
                                "commence_time": commence_time,
                                "home_team": canonical_team_name_from_abbrev(home_abbrev) or home_team,
                                "away_team": canonical_team_name_from_abbrev(away_abbrev) or away_team,
                                "market": "game_total",
                                "side": side,
                                "sportsbook": sportsbook,
                                "bookmaker_id": None,
                                "line_value": float(line_value) if line_value is not None else 0.0,
                                "price": float(price) if price is not None else None,
                                "market_id": None,
                                "market_name": "totals",
                                "is_historical": 0,
                                "source_url": source_url,
                                "snapshot_type": "intraday",
                                "captured_at": captured_at,
                            }
                        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "game_id",
                "game_date",
                "commence_time",
                "home_team",
                "away_team",
                "market",
                "side",
                "sportsbook",
                "bookmaker_id",
                "line_value",
                "price",
                "market_id",
                "market_name",
                "is_historical",
                "source_url",
                "snapshot_type",
                "captured_at",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True, errors="coerce").astype(str)
    return frame.drop_duplicates(
        subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
        keep="last",
    )
