"""Current-day NBA prop ingestion from public web pages."""

from __future__ import annotations

from datetime import date, datetime, timezone
import re
from typing import Iterable, Optional

from bs4 import BeautifulSoup
import requests

from src.data.fetch_props_from_the_odds_api import write_csv
from src.data.public_page_game_odds import (
    PUBLIC_PAGE_SOURCE_MODE,
    SCORESANDODDS_SCOREBOARD_URL,
    USER_AGENT,
    _absolute_url,
    _canonical_book,
    _clean_text,
    _get_html,
    _now_iso,
    _parse_american_odds,
    _parse_numeric_token,
    build_scoresandodds_games_frame,
)
from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev


COVERS_PROPS_URL = "https://www.covers.com/sport/basketball/nba/player-props"
HTTP_TIMEOUT_SECONDS = 30
DEFAULT_SPORT_KEY = "basketball_nba"
SUPPORTED_PROP_MARKETS = {
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
}


SCORESANDODDS_MARKET_MAP = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "3-pointers": "player_threes",
}


COVERS_MARKET_MAP = {
    "points scored": "player_points",
    "total rebounds": "player_rebounds",
    "total assists": "player_assists",
    "3-pointers made": "player_threes",
}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _parse_prop_cell(cell) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if cell is None:
        return None, None, None
    value_text = _clean_text(cell.select_one(".data-value").get_text(" ", strip=True) if cell.select_one(".data-value") is not None else cell.get_text(" ", strip=True))
    odds_text = _clean_text(cell.select_one(".data-odds").get_text(" ", strip=True) if cell.select_one(".data-odds") is not None else "")
    if not value_text:
        return None, None, None
    side = "over" if value_text.lower().startswith("o") else "under" if value_text.lower().startswith("u") else None
    line = _parse_numeric_token(value_text)
    odds = _parse_american_odds(odds_text)
    if side is None or line is None or odds is None:
        return None, None, None
    return float(line), float(odds), side


def _scoresandodds_market_key(tbody_key: str) -> Optional[str]:
    lowered = _clean_text(tbody_key).lower()
    for token, market_key in SCORESANDODDS_MARKET_MAP.items():
        if token in lowered:
            return market_key
    return None


def build_scoresandodds_prop_rows(
    html: str,
    *,
    game_row,
    source_url: str,
    allowed_markets: Iterable[str],
    page_snapshot_at: Optional[str] = None,
) -> list[dict]:
    snapshot_at = page_snapshot_at or _now_iso()
    wanted = {str(market) for market in allowed_markets}
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    for tbody in soup.select("tbody[data-key*='odds-table--'][data-key*='-p-']"):
        market_key = _scoresandodds_market_key(_clean_text(tbody.get("data-key")))
        if market_key not in wanted:
            continue

        table = tbody.find_parent("table")
        if table is None:
            continue
        books = [
            _canonical_book(img.get("alt"))
            for img in table.select("thead th.book-logo img[alt]")
        ]
        trs = tbody.find_all("tr")
        if len(trs) < 2:
            continue

        for index in range(0, len(trs), 2):
            over_row = trs[index]
            under_row = trs[index + 1] if index + 1 < len(trs) else None
            over_cells = over_row.find_all("td")
            if len(over_cells) < 2:
                continue
            player_name = _clean_text(over_cells[0].get_text(" ", strip=True))
            if not player_name or player_name.upper().startswith("PROJ"):
                continue

            for book, over_cell, under_cell in zip(
                books,
                over_cells[1:],
                under_row.find_all("td")[1:] if under_row is not None else [],
            ):
                over_line, over_odds, over_side = _parse_prop_cell(over_cell)
                under_line, under_odds, under_side = _parse_prop_cell(under_cell)

                for line_value, odds, side in (
                    (over_line, over_odds, over_side),
                    (under_line, under_odds, under_side),
                ):
                    if line_value is None or odds is None or side is None:
                        continue
                    rows.append(
                        {
                            "player": player_name,
                            "line": float(line_value),
                            "side": side,
                            "odds": int(odds),
                            "book": book,
                            "sport_key": DEFAULT_SPORT_KEY,
                            "event_id": str(game_row["fixture_id"]),
                            "commence_time": str(game_row["commence_time"]),
                            "home_team": str(game_row["home_team"]),
                            "away_team": str(game_row["away_team"]),
                            "market_key": market_key,
                            "book_key": book,
                            "book_title": book,
                            "source_provider": "scoresandodds",
                            "source_mode": PUBLIC_PAGE_SOURCE_MODE,
                            "source_page_url": source_url,
                            "source_book": book,
                            "page_snapshot_at": snapshot_at,
                        }
                    )

    return rows


def fetch_scoresandodds_prop_rows(
    *,
    report_date: date,
    allowed_markets: Iterable[str] = SUPPORTED_PROP_MARKETS,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    http = session or _session()
    page_snapshot_at = _now_iso()
    scoreboard_html = _get_html(SCORESANDODDS_SCOREBOARD_URL, session=http)
    games_df = build_scoresandodds_games_frame(
        scoreboard_html,
        report_date=report_date,
        source_url=SCORESANDODDS_SCOREBOARD_URL,
        page_snapshot_at=page_snapshot_at,
    )
    rows: list[dict] = []
    for _, game_row in games_df.iterrows():
        matchup_url = _clean_text(game_row.get("matchup_url"))
        if not matchup_url:
            continue
        matchup_html = _get_html(matchup_url, session=http)
        rows.extend(
            build_scoresandodds_prop_rows(
                matchup_html,
                game_row=game_row,
                source_url=matchup_url,
                allowed_markets=allowed_markets,
                page_snapshot_at=page_snapshot_at,
            )
        )
    return rows


def _covers_market_key(label: str) -> Optional[str]:
    lowered = _clean_text(label).lower()
    for token, market_key in COVERS_MARKET_MAP.items():
        if token == lowered:
            return market_key
    return None


def _parse_covers_prediction(text: str) -> tuple[Optional[str], Optional[float]]:
    cleaned = _clean_text(text).lower()
    match = re.search(r"\b([ou])\s*([0-9]+(?:\.[0-9]+)?)\b", cleaned)
    if match is None:
        return None, None
    side = "over" if match.group(1) == "o" else "under"
    return side, float(match.group(2))


def _parse_covers_book_column(column) -> tuple[str, Optional[float]]:
    image = column.select_one("img[alt]")
    book = _canonical_book(_clean_text(image.get("alt") if image is not None else ""))
    odds_link = column.select_one("a.book-odds")
    odds_text = _clean_text(odds_link.get_text(" ", strip=True) if odds_link is not None else "").lower()
    odds_text = odds_text.replace("−", "-")
    odds_matches = re.findall(r"(even|[+-]\d+)", odds_text)
    if odds_matches:
        odds = 100.0 if odds_matches[-1] == "even" else float(odds_matches[-1])
    else:
        odds = _parse_american_odds(odds_text)
    return book, odds


def _strip_player_position(text: str) -> str:
    return re.sub(r"\s*\([A-Z/]+\)\s*$", "", _clean_text(text))


def build_covers_prop_rows(
    html: str,
    *,
    report_date: date,
    source_url: str = COVERS_PROPS_URL,
    allowed_markets: Iterable[str] = SUPPORTED_PROP_MARKETS,
    page_snapshot_at: Optional[str] = None,
) -> list[dict]:
    snapshot_at = page_snapshot_at or _now_iso()
    wanted = {str(market) for market in allowed_markets}
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    for container in soup.select("tr.game-projections-container"):
        badge = container.select_one("span._badge")
        market_key = _covers_market_key(_clean_text(badge.get_text(" ", strip=True) if badge is not None else ""))
        if market_key not in wanted:
            continue

        game_link = container.select_one("a.projection-game-link")
        matchup = _clean_text(game_link.get_text(" ", strip=True) if game_link is not None else "")
        matchup_match = re.match(r"^([A-Z]{2,3})\s*@\s*([A-Z]{2,3})$", matchup)
        if matchup_match is None:
            continue
        away_abbrev, home_abbrev = matchup_match.groups()
        away_team = canonical_team_name_from_abbrev(away_abbrev) or away_abbrev
        home_team = canonical_team_name_from_abbrev(home_abbrev) or home_abbrev

        player_el = container.select_one(".category-title .category")
        prediction_el = container.select_one(".category-title .prediction")
        player_name = _strip_player_position(_clean_text(player_el.get_text(" ", strip=True) if player_el is not None else ""))
        side, line_value = _parse_covers_prediction(_clean_text(prediction_el.get_text(" ", strip=True) if prediction_el is not None else ""))
        if not player_name or side is None or line_value is None:
            continue

        columns = container.select(".compare-odds-column")
        if not columns:
            continue

        event_id = stable_id(report_date.isoformat(), home_abbrev, away_abbrev, prefix="covers")
        commence_time = f"{report_date.isoformat()}T00:00:00Z"
        for column in columns:
            book, odds = _parse_covers_book_column(column)
            if not book or odds is None:
                continue
            rows.append(
                {
                    "player": player_name,
                    "line": float(line_value),
                    "side": side,
                    "odds": int(odds),
                    "book": book,
                    "sport_key": DEFAULT_SPORT_KEY,
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_key": market_key,
                    "book_key": book,
                    "book_title": book,
                    "source_provider": "covers",
                    "source_mode": PUBLIC_PAGE_SOURCE_MODE,
                    "source_page_url": source_url,
                    "source_book": book,
                    "page_snapshot_at": snapshot_at,
                }
            )

    return rows


def fetch_covers_prop_rows(
    *,
    report_date: date,
    allowed_markets: Iterable[str] = SUPPORTED_PROP_MARKETS,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    http = session or _session()
    page_snapshot_at = _now_iso()
    html = _get_html(COVERS_PROPS_URL, session=http)
    return build_covers_prop_rows(
        html,
        report_date=report_date,
        source_url=COVERS_PROPS_URL,
        allowed_markets=allowed_markets,
        page_snapshot_at=page_snapshot_at,
    )


def write_prop_rows(rows: list[dict], output_path) -> None:
    write_csv(rows, output_path)
