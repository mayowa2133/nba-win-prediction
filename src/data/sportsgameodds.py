"""Current-day NBA game odds and props collection from SportsGameOdds."""

from __future__ import annotations

from datetime import date, timedelta
import os
import statistics
from typing import Iterable, Optional

import pandas as pd
import requests

from src.data.public_page_game_odds import (
    GAME_LINES_COLUMNS,
    GAME_ODDS_COLUMNS,
    _canonical_book,
    _now_iso,
    _parse_american_odds,
    _parse_numeric_token,
)
from src.data.public_page_props import DEFAULT_SPORT_KEY, SUPPORTED_PROP_MARKETS
from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev


BASE_URL = "https://api.sportsgameodds.com/v2"
DEFAULT_LEAGUE_ID = "NBA"
DEFAULT_BOOKMAKERS = ("draftkings", "fanduel", "caesars", "betmgm")
SOURCE_PROVIDER = "sportsgameodds"
SOURCE_MODE = "api_snapshot"
SUPPORTED_PROP_STAT_IDS = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "threes_made": "player_threes",
}
TEAM_LEVEL_STAT_ENTITIES = {"all", "home", "away"}


def get_sportsgameodds_api_key(cli_key: Optional[str]) -> str:
    api_key = cli_key or os.getenv("SPORTSGAMEODDS_API_KEY")
    if not api_key:
        raise RuntimeError("SPORTSGAMEODDS_API_KEY is required for SportsGameOdds live collection")
    return api_key


def request_events(
    *,
    api_key: str,
    report_date: date,
    bookmaker_ids: Optional[Iterable[str]] = None,
    league_id: str = DEFAULT_LEAGUE_ID,
) -> list[dict]:
    starts_after = f"{report_date.isoformat()}T00:00:00Z"
    starts_before = f"{(report_date + timedelta(days=1)).isoformat()}T00:00:00Z"
    params = {
        "leagueID": league_id,
        "oddsAvailable": "true",
        "finalized": "false",
        "startsAfter": starts_after,
        "startsBefore": starts_before,
        "includeOpenCloseOdds": "true",
        "includeAltLines": "false",
        "limit": "50",
    }
    if bookmaker_ids:
        normalized_books = []
        for item in bookmaker_ids:
            normalized = _canonical_book(str(item).strip())
            if normalized:
                normalized_books.append(normalized)
        if normalized_books:
            params["bookmakerID"] = ",".join(normalized_books)

    response = requests.get(
        f"{BASE_URL}/events",
        params=params,
        headers={
            "x-api-key": api_key,
            "User-Agent": "Mozilla/5.0",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise RuntimeError(str(payload.get("error") or "SportsGameOdds request failed"))
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("SportsGameOdds response did not include a data list")
    return data


def _event_id(event: dict, game_date: str, home_abbrev: str, away_abbrev: str) -> str:
    return str(event.get("eventID") or event.get("id") or stable_id(game_date, home_abbrev, away_abbrev, prefix="sgo"))


def _team_name(team: object) -> str:
    if not isinstance(team, dict):
        return ""
    names = team.get("names")
    if isinstance(names, dict):
        for key in ("long", "medium", "short"):
            value = str(names.get(key) or "").strip()
            if value:
                return value
    return str(team.get("teamID") or "").strip()


def _event_teams(event: dict) -> tuple[str, str, str, str]:
    teams = event.get("teams") if isinstance(event.get("teams"), dict) else {}
    home_raw = _team_name(teams.get("home"))
    away_raw = _team_name(teams.get("away"))
    home_abbrev = canonical_team_abbrev(home_raw) or home_raw
    away_abbrev = canonical_team_abbrev(away_raw) or away_raw
    home_team = canonical_team_name_from_abbrev(home_abbrev) or home_raw
    away_team = canonical_team_name_from_abbrev(away_abbrev) or away_raw
    return home_team, away_team, home_abbrev, away_abbrev


def _commence_time(event: dict) -> str:
    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    return str(status.get("startsAt") or event.get("startsAt") or "")


def _format_player_name(player_id: str) -> str:
    raw = str(player_id or "").strip()
    if not raw:
        return ""
    parts = [part for part in raw.split("_") if part]
    if len(parts) >= 3 and parts[-1].isalpha() and parts[-2].isdigit():
        parts = parts[:-2]
    return " ".join(part.capitalize() for part in parts)


def _median_parsed(values: Iterable[object], parser) -> Optional[float]:
    parsed = []
    for value in values:
        parsed_value = parser(value)
        if parsed_value is not None:
            parsed.append(float(parsed_value))
    if not parsed:
        return None
    return float(statistics.median(parsed))


def _market_fields(odd: dict, bookmaker_data: dict) -> tuple[Optional[float], Optional[float]]:
    bet_type_id = str(odd.get("betTypeID") or "")
    if bet_type_id == "ml":
        return 0.0, _parse_american_odds(bookmaker_data.get("odds") or odd.get("bookOdds"))
    if bet_type_id == "sp":
        line = _parse_numeric_token(bookmaker_data.get("spread") or odd.get("bookSpread") or odd.get("fairSpread"))
        return abs(float(line)) if line is not None else None, _parse_american_odds(bookmaker_data.get("odds") or odd.get("bookOdds"))
    if bet_type_id == "ou":
        line = _parse_numeric_token(bookmaker_data.get("overUnder") or odd.get("bookOverUnder") or odd.get("fairOverUnder"))
        return float(line) if line is not None else None, _parse_american_odds(bookmaker_data.get("odds") or odd.get("bookOdds"))
    return None, None


def _game_market_for_odd(odd: dict) -> tuple[Optional[str], Optional[str]]:
    if str(odd.get("periodID") or "") != "game":
        return None, None
    if str(odd.get("statID") or "") != "points":
        return None, None
    bet_type_id = str(odd.get("betTypeID") or "")
    side_id = str(odd.get("sideID") or "")
    stat_entity_id = str(odd.get("statEntityID") or "")
    if bet_type_id == "ml" and side_id in {"home", "away"} and stat_entity_id in {"home", "away"}:
        return "game_moneyline", side_id
    if bet_type_id == "sp" and side_id in {"home", "away"} and stat_entity_id in {"home", "away"}:
        return "game_spread", side_id
    if bet_type_id == "ou" and side_id in {"over", "under"} and stat_entity_id == "all":
        return "game_total", side_id
    return None, None


def build_sportsgameodds_game_odds_snapshot_frame(
    events: list[dict],
    *,
    report_date: date,
    source_url: str = f"{BASE_URL}/events",
    page_snapshot_at: Optional[str] = None,
) -> pd.DataFrame:
    snapshot_at = page_snapshot_at or _now_iso()
    rows = []
    report_date_str = report_date.isoformat()

    for event in events:
        commence_time = _commence_time(event)
        game_date = commence_time[:10]
        if game_date != report_date_str:
            continue
        home_team, away_team, home_abbrev, away_abbrev = _event_teams(event)
        if not home_team or not away_team:
            continue
        fixture_id = _event_id(event, game_date, home_abbrev, away_abbrev)
        game_id = stable_id(game_date, home_abbrev, away_abbrev, prefix="game")
        for odd in (event.get("odds") or {}).values():
            if not isinstance(odd, dict):
                continue
            market, side = _game_market_for_odd(odd)
            if market is None or side is None:
                continue
            for bookmaker_id, bookmaker_data in (odd.get("byBookmaker") or {}).items():
                if not isinstance(bookmaker_data, dict):
                    continue
                if bookmaker_data.get("available") is False:
                    continue
                line_value, price = _market_fields(odd, bookmaker_data)
                if line_value is None or price is None:
                    continue
                captured_at = str(bookmaker_data.get("lastUpdatedAt") or commence_time or snapshot_at)
                sportsbook = _canonical_book(bookmaker_id)
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "game_id": game_id,
                        "game_date": game_date,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market": market,
                        "side": side,
                        "sportsbook": sportsbook,
                        "bookmaker_id": str(bookmaker_id),
                        "line_value": float(line_value),
                        "price": float(price),
                        "market_id": str(odd.get("oddID") or ""),
                        "market_name": str(odd.get("oddID") or ""),
                        "is_historical": 0,
                        "source_url": source_url,
                        "snapshot_type": "intraday",
                        "captured_at": captured_at,
                        "source_provider": SOURCE_PROVIDER,
                        "source_mode": SOURCE_MODE,
                        "source_page_url": source_url,
                        "source_book": sportsbook,
                        "is_consensus_quote": 0,
                        "page_snapshot_at": captured_at,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=GAME_ODDS_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True, errors="coerce").astype(str)
    return frame.drop_duplicates(
        subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
        keep="last",
    )


def _select_odd(event: dict, *, bet_type_id: str, side_id: str, stat_entity_id: str, stat_id: str = "points") -> dict:
    for odd in (event.get("odds") or {}).values():
        if not isinstance(odd, dict):
            continue
        if (
            str(odd.get("periodID") or "") == "game"
            and str(odd.get("statID") or "") == stat_id
            and str(odd.get("betTypeID") or "") == bet_type_id
            and str(odd.get("sideID") or "") == side_id
            and str(odd.get("statEntityID") or "") == stat_entity_id
        ):
            return odd
    return {}


def _median_book_field(odd: dict, field: str, parser) -> Optional[float]:
    by_bookmaker = odd.get("byBookmaker") or {}
    if not isinstance(by_bookmaker, dict):
        return None
    return _median_parsed(
        (
            bookmaker_data.get(field)
            for bookmaker_data in by_bookmaker.values()
            if isinstance(bookmaker_data, dict)
        ),
        parser,
    )


def build_sportsgameodds_game_lines_frame(
    events: list[dict],
    *,
    report_date: date,
    source_url: str = f"{BASE_URL}/events",
    page_snapshot_at: Optional[str] = None,
) -> pd.DataFrame:
    snapshot_at = page_snapshot_at or _now_iso()
    rows = []
    report_date_str = report_date.isoformat()

    for event in events:
        commence_time = _commence_time(event)
        game_date = commence_time[:10]
        if game_date != report_date_str:
            continue
        home_team, away_team, home_abbrev, away_abbrev = _event_teams(event)
        if not home_team or not away_team:
            continue

        home_ml = _select_odd(event, bet_type_id="ml", side_id="home", stat_entity_id="home")
        away_ml = _select_odd(event, bet_type_id="ml", side_id="away", stat_entity_id="away")
        home_spread = _select_odd(event, bet_type_id="sp", side_id="home", stat_entity_id="home")
        away_spread = _select_odd(event, bet_type_id="sp", side_id="away", stat_entity_id="away")
        total_over = _select_odd(event, bet_type_id="ou", side_id="over", stat_entity_id="all")
        total_under = _select_odd(event, bet_type_id="ou", side_id="under", stat_entity_id="all")

        current_home_spread = _parse_numeric_token(home_spread.get("bookSpread"))
        current_away_spread = _parse_numeric_token(away_spread.get("bookSpread"))
        if current_home_spread is None and current_away_spread is not None:
            current_home_spread = -float(current_away_spread)
        if current_away_spread is None and current_home_spread is not None:
            current_away_spread = -float(current_home_spread)
        open_home_spread = _median_book_field(home_spread, "openSpread", _parse_numeric_token)
        open_away_spread = _median_book_field(away_spread, "openSpread", _parse_numeric_token)
        if open_away_spread is None and open_home_spread is not None:
            open_away_spread = -float(open_home_spread)
        if open_home_spread is None and open_away_spread is not None:
            open_home_spread = -float(open_away_spread)

        vegas_game_total = _parse_numeric_token(total_over.get("bookOverUnder") or total_under.get("bookOverUnder"))
        open_game_total = _median_book_field(total_over, "openOverUnder", _parse_numeric_token)
        if open_game_total is None:
            open_game_total = _median_book_field(total_under, "openOverUnder", _parse_numeric_token)

        rows.append(
            {
                "event_id": _event_id(event, game_date, home_abbrev, away_abbrev),
                "game_date": game_date,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "vegas_game_total": vegas_game_total,
                "vegas_home_spread": current_home_spread,
                "vegas_away_spread": current_away_spread,
                "vegas_abs_spread": abs(float(current_home_spread)) if current_home_spread is not None else (abs(float(current_away_spread)) if current_away_spread is not None else None),
                "open_game_total": open_game_total,
                "open_game_total_odds": _median_book_field(total_over, "openOdds", _parse_american_odds),
                "open_home_spread": open_home_spread,
                "open_home_spread_odds": _median_book_field(home_spread, "openOdds", _parse_american_odds),
                "open_away_spread": open_away_spread,
                "open_away_spread_odds": _median_book_field(away_spread, "openOdds", _parse_american_odds),
                "current_total_odds_over": _parse_american_odds(total_over.get("bookOdds")),
                "current_total_odds_under": _parse_american_odds(total_under.get("bookOdds")),
                "current_home_moneyline": _parse_american_odds(home_ml.get("bookOdds")),
                "current_away_moneyline": _parse_american_odds(away_ml.get("bookOdds")),
                "matchup_url": source_url,
                "source_provider": SOURCE_PROVIDER,
                "source_mode": SOURCE_MODE,
                "source_page_url": source_url,
                "page_snapshot_at": snapshot_at,
            }
        )

    if not rows:
        return pd.DataFrame(columns=GAME_LINES_COLUMNS)
    return pd.DataFrame(rows).drop_duplicates(subset=["event_id"], keep="last")


def fetch_sportsgameodds_game_frames(
    *,
    report_date: date,
    api_key: Optional[str] = None,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_key = get_sportsgameodds_api_key(api_key)
    snapshot_at = _now_iso()
    events = request_events(api_key=resolved_key, report_date=report_date, bookmaker_ids=bookmakers)
    return (
        build_sportsgameodds_game_odds_snapshot_frame(events, report_date=report_date, page_snapshot_at=snapshot_at),
        build_sportsgameodds_game_lines_frame(events, report_date=report_date, page_snapshot_at=snapshot_at),
    )


def build_sportsgameodds_prop_rows(
    events: list[dict],
    *,
    report_date: date,
    allowed_markets: Iterable[str] = SUPPORTED_PROP_MARKETS,
    source_url: str = f"{BASE_URL}/events",
    page_snapshot_at: Optional[str] = None,
) -> list[dict]:
    snapshot_at = page_snapshot_at or _now_iso()
    wanted = {str(market) for market in allowed_markets}
    report_date_str = report_date.isoformat()
    rows: list[dict] = []

    for event in events:
        commence_time = _commence_time(event)
        game_date = commence_time[:10]
        if game_date != report_date_str:
            continue
        home_team, away_team, home_abbrev, away_abbrev = _event_teams(event)
        if not home_team or not away_team:
            continue
        fixture_id = _event_id(event, game_date, home_abbrev, away_abbrev)
        for odd in (event.get("odds") or {}).values():
            if not isinstance(odd, dict):
                continue
            if str(odd.get("periodID") or "") != "game":
                continue
            if str(odd.get("betTypeID") or "") != "ou":
                continue
            stat_entity = str(odd.get("statEntityID") or "")
            if stat_entity in TEAM_LEVEL_STAT_ENTITIES or not stat_entity:
                continue
            market_key = SUPPORTED_PROP_STAT_IDS.get(str(odd.get("statID") or ""))
            if market_key not in wanted:
                continue
            side = str(odd.get("sideID") or "")
            if side not in {"over", "under"}:
                continue
            player_name = _format_player_name(stat_entity)
            if not player_name:
                continue
            for bookmaker_id, bookmaker_data in (odd.get("byBookmaker") or {}).items():
                if not isinstance(bookmaker_data, dict):
                    continue
                if bookmaker_data.get("available") is False:
                    continue
                line = _parse_numeric_token(bookmaker_data.get("overUnder") or odd.get("bookOverUnder"))
                odds = _parse_american_odds(bookmaker_data.get("odds") or odd.get("bookOdds"))
                if line is None or odds is None:
                    continue
                book = _canonical_book(bookmaker_id)
                rows.append(
                    {
                        "player": player_name,
                        "line": float(line),
                        "side": side,
                        "odds": int(odds),
                        "book": book,
                        "sport_key": DEFAULT_SPORT_KEY,
                        "event_id": fixture_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_key": market_key,
                        "book_key": book,
                        "book_title": book,
                        "source_provider": SOURCE_PROVIDER,
                        "source_mode": SOURCE_MODE,
                        "source_page_url": source_url,
                        "source_book": book,
                        "page_snapshot_at": str(bookmaker_data.get("lastUpdatedAt") or snapshot_at),
                    }
                )
    return rows


def fetch_sportsgameodds_prop_rows(
    *,
    report_date: date,
    allowed_markets: Iterable[str] = SUPPORTED_PROP_MARKETS,
    api_key: Optional[str] = None,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
) -> list[dict]:
    resolved_key = get_sportsgameodds_api_key(api_key)
    snapshot_at = _now_iso()
    events = request_events(api_key=resolved_key, report_date=report_date, bookmaker_ids=bookmakers)
    return build_sportsgameodds_prop_rows(
        events,
        report_date=report_date,
        allowed_markets=allowed_markets,
        page_snapshot_at=snapshot_at,
    )
