"""Current-day NBA game odds ingestion from public web pages."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import re
from typing import Iterable, Optional

from bs4 import BeautifulSoup
import pandas as pd
import requests

from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev


SCORESANDODDS_SCOREBOARD_URL = "https://www.scoresandodds.com/nba"
ESPN_ODDS_URL = "https://www.espn.com/nba/odds"
PUBLIC_PAGE_SOURCE_MODE = "public_page_snapshot"
HTTP_TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0 Safari/537.36"


GAME_ODDS_COLUMNS = [
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
    "source_provider",
    "source_mode",
    "source_page_url",
    "source_book",
    "is_consensus_quote",
    "page_snapshot_at",
]


GAME_LINES_COLUMNS = [
    "event_id",
    "game_date",
    "commence_time",
    "home_team",
    "away_team",
    "vegas_game_total",
    "vegas_home_spread",
    "vegas_away_spread",
    "vegas_abs_spread",
    "open_game_total",
    "open_game_total_odds",
    "open_home_spread",
    "open_home_spread_odds",
    "open_away_spread",
    "open_away_spread_odds",
    "current_total_odds_over",
    "current_total_odds_under",
    "current_home_moneyline",
    "current_away_moneyline",
    "matchup_url",
    "source_provider",
    "source_mode",
    "source_page_url",
    "page_snapshot_at",
]


BOOK_ALIASES = {
    "betmgm": "betmgm",
    "bet365": "bet365",
    "caesars": "caesars",
    "draftkings": "draftkings",
    "fanduel": "fanduel",
    "fanatics": "fanatics",
    "hardrock": "hardrock",
    "prizepicks": "prizepicks",
    "riverscasino": "betrivers",
    "betrivers": "betrivers",
    "sleeper": "sleeper",
    "underdog": "underdog",
    "sports interaction": "sportsinteraction",
    "sportsinteraction": "sportsinteraction",
    "thescore bet": "thescorebet",
    "the score bet": "thescorebet",
    "bet99": "bet99",
    "betway": "betway",
    "betano": "betano",
    "tooniebet": "tooniebet",
    "pinnacle": "pinnacle",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _get_html(url: str, *, session: Optional[requests.Session] = None) -> str:
    http = session or _session()
    response = http.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonical_book(raw: object) -> str:
    text = _clean_text(raw).lower()
    if not text:
        return ""
    text = re.sub(r"\s+logo$", "", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return BOOK_ALIASES.get(text, text.replace(" ", ""))


def _parse_american_odds(text: object) -> Optional[float]:
    raw = _clean_text(text).lower()
    if not raw:
        return None
    if raw == "even":
        return 100.0
    raw = raw.replace("−", "-")
    match = re.search(r"([+-]?\d+)", raw)
    if not match:
        return None
    return float(match.group(1))


def _parse_numeric_token(text: object) -> Optional[float]:
    raw = _clean_text(text)
    if not raw:
        return None
    raw = raw.replace("−", "-")
    match = re.search(r"([+-]?\d+(?:\.\d+)?)", raw)
    if not match:
        return None
    return float(match.group(1))


def _absolute_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"https://www.scoresandodds.com{path}"


def _scoreandodds_row_team(row) -> Optional[str]:
    anchor = row.select_one("a[aria-label]")
    label = _clean_text(anchor.get("aria-label") if anchor is not None else "")
    if label:
        return label
    name = _clean_text(row.select_one(".team-name"))
    return name or None


def _parse_game_card_cell(cell) -> tuple[Optional[float], Optional[float]]:
    if cell is None:
        return None, None
    value_el = cell.select_one(".data-value")
    odds_el = cell.select_one(".data-odds")
    value = _parse_numeric_token(value_el.get_text(" ", strip=True) if value_el is not None else cell.get_text(" ", strip=True))
    odds = _parse_american_odds(odds_el.get_text(" ", strip=True) if odds_el is not None else None)
    return value, odds


def build_scoresandodds_games_frame(
    html: str,
    *,
    report_date: date,
    source_url: str = SCORESANDODDS_SCOREBOARD_URL,
    page_snapshot_at: Optional[str] = None,
) -> pd.DataFrame:
    snapshot_at = page_snapshot_at or _now_iso()
    report_date_str = report_date.isoformat()
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for card in soup.select("div.event-card[id^='nba.']"):
        fixture_token = _clean_text(card.get("id"))
        fixture_id = fixture_token.split(".", 1)[-1] if fixture_token else ""
        time_el = card.select_one("[data-role='localtime'][data-value]")
        commence_time = _clean_text(time_el.get("data-value") if time_el is not None else "")
        game_date = commence_time[:10]
        if game_date and game_date != report_date_str:
            continue

        matchup_url = ""
        for anchor in card.select("a[href]"):
            href = _clean_text(anchor.get("href"))
            if re.match(r"^/nba/[^/]+-vs-[^/]+$", href):
                matchup_url = _absolute_url(href)
                break

        game_rows = card.select("tr.event-card-row")
        if len(game_rows) < 2:
            continue
        away_row, home_row = game_rows[0], game_rows[1]
        away_team_raw = _scoreandodds_row_team(away_row)
        home_team_raw = _scoreandodds_row_team(home_row)
        away_abbrev = canonical_team_abbrev(away_team_raw) or _clean_text(away_team_raw)
        home_abbrev = canonical_team_abbrev(home_team_raw) or _clean_text(home_team_raw)
        if not away_abbrev or not home_abbrev:
            continue
        away_team = canonical_team_name_from_abbrev(away_abbrev) or _clean_text(away_team_raw)
        home_team = canonical_team_name_from_abbrev(home_abbrev) or _clean_text(home_team_raw)

        open_total, open_total_odds = _parse_game_card_cell(away_row.select_one("td.event-card-open"))
        open_home_spread, open_home_spread_odds = _parse_game_card_cell(home_row.select_one("td.event-card-open"))
        current_away_spread, current_away_spread_odds = _parse_game_card_cell(away_row.select_one("td[data-field='current-spread']"))
        current_home_spread, current_home_spread_odds = _parse_game_card_cell(home_row.select_one("td[data-field='current-spread']"))
        current_total_over, current_total_odds_over = _parse_game_card_cell(away_row.select_one("td[data-field='current-total']"))
        current_total_under, current_total_odds_under = _parse_game_card_cell(home_row.select_one("td[data-field='current-total']"))
        current_away_moneyline, _ = _parse_game_card_cell(away_row.select_one("td[data-field='current-moneyline']"))
        current_home_moneyline, _ = _parse_game_card_cell(home_row.select_one("td[data-field='current-moneyline']"))

        if open_home_spread is not None:
            open_away_spread = abs(open_home_spread)
            open_home_spread = float(open_home_spread)
            if open_home_spread > 0:
                open_away_spread = -open_home_spread
            else:
                open_away_spread = abs(open_home_spread)
        else:
            open_away_spread = None

        rows.append(
            {
                "fixture_id": fixture_id or stable_id(report_date_str, home_abbrev, away_abbrev, prefix="sao"),
                "game_id": stable_id(report_date_str, home_abbrev, away_abbrev, prefix="game"),
                "game_date": report_date_str,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "vegas_game_total": current_total_over,
                "vegas_home_spread": current_home_spread,
                "vegas_away_spread": current_away_spread,
                "vegas_abs_spread": abs(float(current_home_spread)) if current_home_spread is not None else (abs(float(current_away_spread)) if current_away_spread is not None else None),
                "open_game_total": open_total,
                "open_game_total_odds": open_total_odds,
                "open_home_spread": open_home_spread,
                "open_home_spread_odds": open_home_spread_odds,
                "open_away_spread": open_away_spread,
                "open_away_spread_odds": open_home_spread_odds,
                "current_total_odds_over": current_total_odds_over,
                "current_total_odds_under": current_total_odds_under,
                "current_home_moneyline": current_home_moneyline,
                "current_away_moneyline": current_away_moneyline,
                "matchup_url": matchup_url,
                "source_provider": "scoresandodds",
                "source_mode": PUBLIC_PAGE_SOURCE_MODE,
                "source_page_url": source_url,
                "page_snapshot_at": snapshot_at,
            }
        )

    if not rows:
        return pd.DataFrame(columns=GAME_LINES_COLUMNS + ["fixture_id", "game_id"])

    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=["fixture_id"], keep="last")


def build_scoresandodds_game_lines_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=GAME_LINES_COLUMNS)
    frame = games_df.copy()
    frame = frame.rename(columns={"fixture_id": "event_id"})
    return frame[[column for column in GAME_LINES_COLUMNS if column in frame.columns]].copy()


def _market_from_tbody(tbody) -> Optional[str]:
    joined = " ".join([_clean_text(tbody.get("id")), " ".join(tbody.get("class", []))]).lower()
    if "spread" in joined:
        return "game_spread"
    if "total" in joined:
        return "game_total"
    if "moneyline" in joined:
        return "game_moneyline"
    return None


def _parse_game_odds_cell(cell, *, market: str) -> tuple[Optional[float], Optional[float]]:
    if cell is None:
        return None, None
    value_el = cell.select_one(".data-value")
    odds_el = cell.select_one(".data-odds")
    value_text = _clean_text(value_el.get_text(" ", strip=True) if value_el is not None else cell.get_text(" ", strip=True))
    odds_text = _clean_text(odds_el.get_text(" ", strip=True) if odds_el is not None else "")

    if market == "game_moneyline":
        return 0.0, _parse_american_odds(value_text)
    return _parse_numeric_token(value_text), _parse_american_odds(odds_text)


def build_scoresandodds_matchup_snapshot_frame(
    html: str,
    *,
    game_row: pd.Series,
    source_url: str,
    page_snapshot_at: Optional[str] = None,
) -> pd.DataFrame:
    snapshot_at = page_snapshot_at or _now_iso()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="odds-table game-table")
    if table is None:
        return pd.DataFrame(columns=GAME_ODDS_COLUMNS)

    books = [
        _canonical_book(img.get("alt"))
        for img in table.select("thead th.book-logo img[alt]")
    ]
    rows = []
    fixture_id = str(game_row["fixture_id"])
    game_id = str(game_row["game_id"])
    game_date = str(game_row["game_date"])
    commence_time = str(game_row["commence_time"])
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])

    for tbody in table.find_all("tbody"):
        market = _market_from_tbody(tbody)
        if market is None:
            continue
        body_rows = tbody.find_all("tr")
        if len(body_rows) < 2:
            continue

        for row_index, tr in enumerate(body_rows[:2]):
            if market in {"game_spread", "game_moneyline"}:
                side = "away" if row_index == 0 else "home"
            else:
                side = "over" if row_index == 0 else "under"

            data_cells = tr.find_all("td", class_="game-odds")
            for book, cell in zip(books, data_cells):
                line_value, price = _parse_game_odds_cell(cell, market=market)
                if market != "game_moneyline" and (line_value is None or price is None):
                    continue
                if market == "game_moneyline" and price is None:
                    continue
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
                        "sportsbook": book,
                        "bookmaker_id": None,
                        "line_value": abs(float(line_value or 0.0)),
                        "price": float(price) if price is not None else None,
                        "market_id": None,
                        "market_name": market,
                        "is_historical": 0,
                        "source_url": source_url,
                        "snapshot_type": "intraday",
                        "captured_at": snapshot_at,
                        "source_provider": "scoresandodds",
                        "source_mode": PUBLIC_PAGE_SOURCE_MODE,
                        "source_page_url": source_url,
                        "source_book": book,
                        "is_consensus_quote": 0,
                        "page_snapshot_at": snapshot_at,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=GAME_ODDS_COLUMNS)
    return pd.DataFrame(rows).drop_duplicates(
        subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
        keep="last",
    )


def fetch_scoresandodds_game_frames(
    *,
    report_date: date,
    session: Optional[requests.Session] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    http = session or _session()
    page_snapshot_at = _now_iso()
    scoreboard_html = _get_html(SCORESANDODDS_SCOREBOARD_URL, session=http)
    games_df = build_scoresandodds_games_frame(
        scoreboard_html,
        report_date=report_date,
        source_url=SCORESANDODDS_SCOREBOARD_URL,
        page_snapshot_at=page_snapshot_at,
    )
    game_lines_df = build_scoresandodds_game_lines_frame(games_df)
    if games_df.empty:
        return pd.DataFrame(columns=GAME_ODDS_COLUMNS), game_lines_df

    frames = []
    for _, game_row in games_df.iterrows():
        matchup_url = _clean_text(game_row.get("matchup_url"))
        if not matchup_url:
            continue
        matchup_html = _get_html(matchup_url, session=http)
        frame = build_scoresandodds_matchup_snapshot_frame(
            matchup_html,
            game_row=game_row,
            source_url=matchup_url,
            page_snapshot_at=page_snapshot_at,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=GAME_ODDS_COLUMNS), game_lines_df
    return pd.concat(frames, ignore_index=True), game_lines_df


def _extract_espn_payload(html: str) -> dict:
    match = re.search(r"window\['__espnfitt__'\]\s*=\s*(\{.*?\});", html, re.S)
    if match is None:
        raise RuntimeError("Could not locate ESPN page payload")
    return json.loads(match.group(1))


def _iter_espn_lines(html: str) -> Iterable[dict]:
    payload = _extract_espn_payload(html)
    odds_content = payload.get("page", {}).get("content", {}).get("odds", {})
    groups = odds_content.get("odds") or []
    for group in groups:
        for line in group.get("lines") or []:
            yield line


def build_espn_game_lines_frame(
    html: str,
    *,
    report_date: date,
    source_url: str = ESPN_ODDS_URL,
    page_snapshot_at: Optional[str] = None,
) -> pd.DataFrame:
    snapshot_at = page_snapshot_at or _now_iso()
    report_date_str = report_date.isoformat()
    rows = []

    for line in _iter_espn_lines(html):
        game_date = str(line.get("date") or "")[:10]
        if game_date != report_date_str:
            continue

        competitors = line.get("competitors") or []
        home = next((item for item in competitors if str(item.get("homeAway")) == "home"), None)
        away = next((item for item in competitors if str(item.get("homeAway")) == "away"), None)
        if home is None or away is None:
            continue

        home_team = str(home.get("team", {}).get("displayName") or "")
        away_team = str(away.get("team", {}).get("displayName") or "")
        odds_entries = line.get("odds") or []
        if not odds_entries:
            continue
        odds_payload = odds_entries[0]
        point_spread = odds_payload.get("pointSpread") or {}
        totals = odds_payload.get("total") or {}
        moneyline = odds_payload.get("moneyline") or {}

        home_close = point_spread.get("home", {}).get("close", {})
        away_close = point_spread.get("away", {}).get("close", {})
        over_close = totals.get("over", {}).get("close", {})
        under_close = totals.get("under", {}).get("close", {})
        home_open = point_spread.get("home", {}).get("open", {})
        away_open = point_spread.get("away", {}).get("open", {})
        over_open = totals.get("over", {}).get("open", {})

        home_close_line = _parse_numeric_token(home_close.get("line"))
        away_close_line = _parse_numeric_token(away_close.get("line"))
        home_open_line = _parse_numeric_token(home_open.get("line"))
        away_open_line = _parse_numeric_token(away_open.get("line"))
        over_close_line = _parse_numeric_token(over_close.get("line"))
        over_open_line = _parse_numeric_token(over_open.get("line"))

        rows.append(
            {
                "event_id": str(line.get("id") or stable_id(game_date, home_team, away_team, prefix="espn")),
                "game_date": report_date_str,
                "commence_time": str(line.get("date") or ""),
                "home_team": home_team,
                "away_team": away_team,
                "vegas_game_total": over_close_line,
                "vegas_home_spread": home_close_line,
                "vegas_away_spread": away_close_line,
                "vegas_abs_spread": abs(float(home_close_line)) if home_close_line is not None else (abs(float(away_close_line)) if away_close_line is not None else None),
                "open_game_total": over_open_line,
                "open_game_total_odds": _parse_american_odds(over_open.get("odds")),
                "open_home_spread": home_open_line,
                "open_home_spread_odds": _parse_american_odds(home_open.get("odds")),
                "open_away_spread": away_open_line,
                "open_away_spread_odds": _parse_american_odds(away_open.get("odds")),
                "current_total_odds_over": _parse_american_odds(over_close.get("odds")),
                "current_total_odds_under": _parse_american_odds(under_close.get("odds")),
                "current_home_moneyline": _parse_american_odds(moneyline.get("home", {}).get("close", {}).get("odds")),
                "current_away_moneyline": _parse_american_odds(moneyline.get("away", {}).get("close", {}).get("odds")),
                "matchup_url": str(line.get("link", {}).get("href") or source_url),
                "source_provider": "espn",
                "source_mode": PUBLIC_PAGE_SOURCE_MODE,
                "source_page_url": source_url,
                "page_snapshot_at": snapshot_at,
            }
        )

    if not rows:
        return pd.DataFrame(columns=GAME_LINES_COLUMNS)
    return pd.DataFrame(rows)


def build_espn_game_odds_snapshot_frame(
    html: str,
    *,
    report_date: date,
    source_url: str = ESPN_ODDS_URL,
    page_snapshot_at: Optional[str] = None,
) -> pd.DataFrame:
    snapshot_at = page_snapshot_at or _now_iso()
    report_date_str = report_date.isoformat()
    rows = []

    for line in _iter_espn_lines(html):
        game_date = str(line.get("date") or "")[:10]
        if game_date != report_date_str:
            continue

        competitors = line.get("competitors") or []
        home = next((item for item in competitors if str(item.get("homeAway")) == "home"), None)
        away = next((item for item in competitors if str(item.get("homeAway")) == "away"), None)
        if home is None or away is None:
            continue

        home_team = str(home.get("team", {}).get("displayName") or "")
        away_team = str(away.get("team", {}).get("displayName") or "")
        home_abbrev = canonical_team_abbrev(home.get("team", {}).get("abbreviation") or home_team) or home_team
        away_abbrev = canonical_team_abbrev(away.get("team", {}).get("abbreviation") or away_team) or away_team
        fixture_id = str(line.get("id") or stable_id(game_date, home_abbrev, away_abbrev, prefix="espn"))
        game_id = stable_id(game_date, home_abbrev, away_abbrev, prefix="game")

        for odds_payload in line.get("odds") or []:
            sportsbook = _canonical_book(
                odds_payload.get("provider", {}).get("displayName")
                or odds_payload.get("provider", {}).get("name")
            ) or "draftkings"
            point_spread = odds_payload.get("pointSpread") or {}
            totals = odds_payload.get("total") or {}
            moneyline = odds_payload.get("moneyline") or {}

            home_close = point_spread.get("home", {}).get("close", {})
            away_close = point_spread.get("away", {}).get("close", {})
            over_close = totals.get("over", {}).get("close", {})
            under_close = totals.get("under", {}).get("close", {})
            home_ml = moneyline.get("home", {}).get("close", {})
            away_ml = moneyline.get("away", {}).get("close", {})

            market_rows = [
                ("game_spread", "home", _parse_numeric_token(home_close.get("line")), _parse_american_odds(home_close.get("odds"))),
                ("game_spread", "away", _parse_numeric_token(away_close.get("line")), _parse_american_odds(away_close.get("odds"))),
                ("game_total", "over", _parse_numeric_token(over_close.get("line")), _parse_american_odds(over_close.get("odds"))),
                ("game_total", "under", _parse_numeric_token(under_close.get("line")), _parse_american_odds(under_close.get("odds"))),
                ("game_moneyline", "home", 0.0, _parse_american_odds(home_ml.get("odds"))),
                ("game_moneyline", "away", 0.0, _parse_american_odds(away_ml.get("odds"))),
            ]
            for market, side, line_value, price in market_rows:
                if price is None:
                    continue
                if market != "game_moneyline" and line_value is None:
                    continue
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "game_id": game_id,
                        "game_date": report_date_str,
                        "commence_time": str(line.get("date") or ""),
                        "home_team": home_team,
                        "away_team": away_team,
                        "market": market,
                        "side": side,
                        "sportsbook": sportsbook,
                        "bookmaker_id": None,
                        "line_value": abs(float(line_value or 0.0)),
                        "price": float(price),
                        "market_id": None,
                        "market_name": market,
                        "is_historical": 0,
                        "source_url": source_url,
                        "snapshot_type": "intraday",
                        "captured_at": snapshot_at,
                        "source_provider": "espn",
                        "source_mode": PUBLIC_PAGE_SOURCE_MODE,
                        "source_page_url": source_url,
                        "source_book": sportsbook,
                        "is_consensus_quote": 0,
                        "page_snapshot_at": snapshot_at,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=GAME_ODDS_COLUMNS)
    return pd.DataFrame(rows).drop_duplicates(
        subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
        keep="last",
    )


def fetch_espn_game_frames(
    *,
    report_date: date,
    session: Optional[requests.Session] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    http = session or _session()
    page_snapshot_at = _now_iso()
    html = _get_html(ESPN_ODDS_URL, session=http)
    snapshot_df = build_espn_game_odds_snapshot_frame(
        html,
        report_date=report_date,
        source_url=ESPN_ODDS_URL,
        page_snapshot_at=page_snapshot_at,
    )
    game_lines_df = build_espn_game_lines_frame(
        html,
        report_date=report_date,
        source_url=ESPN_ODDS_URL,
        page_snapshot_at=page_snapshot_at,
    )
    return snapshot_df, game_lines_df
