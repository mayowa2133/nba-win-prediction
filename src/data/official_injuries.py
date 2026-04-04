"""Fetch and parse official NBA injury reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
import requests
from sqlalchemy import delete

from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev, team_name_mappings
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import InjuryReportRecord


OFFICIAL_NBA_BASE = "https://official.nba.com"
OFFICIAL_REPORT_LINK_RE = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf", re.IGNORECASE)
TEAM_ROW_RE = re.compile(
    r"^(?:(?P<game_date>\d{2}/\d{2}/\d{4})\s+)?(?:(?P<game_time>\d{1,2}:\d{2})\s+\(ET\)\s+)?(?:(?P<matchup>[A-Z]{2,3}@[A-Z]{2,3})\s+)?(?P<rest>.+)$"
)
GAME_HEADER_RE = re.compile(r"(?P<game_date>\d{2}/\d{2}/\d{4})\s+(?P<game_time>\d{1,2}:\d{2})\s+\(ET\)\s+(?P<matchup>[A-Z]{2,3}@[A-Z]{2,3})")
PLAYER_STATUS_RE = re.compile(
    r"^(?P<player>.+?)\s+(?P<status>Available|Probable|Questionable|Doubtful|Out)\b(?P<reason>.*)$",
    re.IGNORECASE,
)
TOKENIZED_PAGE_HEADER_RE = re.compile(
    r"Injury\s+Report:\s+\d{2}/\d{2}/\d{2}\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+Page\s+\d+\s+of\s+\d+",
    re.IGNORECASE,
)
TOKENIZED_COLUMN_HEADER_RE = re.compile(
    r"Game\s+Date\s+Game\s+Time\s+Matchup\s+Team\s+Player\s+Name\s+Current\s+Status\s+Reason",
    re.IGNORECASE,
)
TOKENIZED_PLAYER_RE = re.compile(
    r"(?P<player>[A-Za-z'’.\-]+(?:\s+[A-Za-z'’.\-]+)*,\s+[A-Za-z'’.\-]+(?:\s+[A-Za-z'’.\-]+)*)\s+"
    r"(?P<status>Available|Probable|Questionable|Doubtful|Out)\s+"
    r"(?P<reason>.*?)(?=(?:[A-Za-z'’.\-]+(?:\s+[A-Za-z'’.\-]+)*,\s+[A-Za-z'’.\-]+(?:\s+[A-Za-z'’.\-]+)*\s+"
    r"(?:Available|Probable|Questionable|Doubtful|Out)\b)|$)",
    re.IGNORECASE,
)
STATUS_TOKENS = {"Available", "Probable", "Questionable", "Doubtful", "Out"}
NON_NAME_START_TOKENS = {
    "assignment",
    "back",
    "big",
    "bursitis",
    "calf",
    "capsulitis",
    "competition",
    "contusion",
    "deep",
    "finger",
    "foot",
    "g",
    "hamstring",
    "hip",
    "illness",
    "injury",
    "injury/illness",
    "knee",
    "league",
    "left",
    "low",
    "lung",
    "not",
    "on",
    "pain",
    "pneumothorax",
    "reconditioning",
    "retrocalcaneal",
    "return",
    "right",
    "shoulder",
    "soreness",
    "sprain",
    "strain",
    "surgery",
    "team",
    "thrombosis",
    "to",
    "toe",
    "two",
    "two-",
    "quad",
    "vein",
    "way",
    "with",
}


def current_season_start_year(report_date: date) -> int:
    return report_date.year if report_date.month >= 10 else report_date.year - 1


def official_report_index_url(report_date: date) -> str:
    start_year = current_season_start_year(report_date)
    end_suffix = str(start_year + 1)[-2:]
    return f"{OFFICIAL_NBA_BASE}/nba-injury-report-{start_year}-{end_suffix}-season/"


def normalize_report_status(raw_status: str) -> str:
    status = str(raw_status or "").strip().lower()
    mapping = {
        "out": "out",
        "doubtful": "doubtful",
        "questionable": "questionable",
        "probable": "probable",
        "available": "available",
        "not yet submitted": "inactive_other",
    }
    return mapping.get(status, "inactive_other")


def projected_availability_for_status(normalized_status: str) -> str:
    if normalized_status in {"out", "doubtful", "inactive_other"}:
        return "unavailable"
    if normalized_status == "questionable":
        return "uncertain"
    if normalized_status in {"probable", "available"}:
        return "available"
    return "unknown"


def extract_pdf_links(index_html: str, *, report_date: date) -> List[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    target = report_date.isoformat()
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "Injury-Report_" not in href:
            continue
        absolute = href if href.startswith("http") else f"{OFFICIAL_NBA_BASE}{href}"
        match = OFFICIAL_REPORT_LINK_RE.search(absolute)
        if match and match.group(1) == target:
            links.append(absolute)
    return sorted(set(links))


def parse_report_timestamp_from_url(url: str) -> tuple[str, str]:
    match = OFFICIAL_REPORT_LINK_RE.search(url)
    if not match:
        return "", ""
    report_date = match.group(1)
    hour = int(match.group(2))
    minute = match.group(3)
    meridiem = match.group(4).upper()
    return report_date, f"{hour:02d}:{minute} {meridiem}"


def fetch_pdf_text(url: str, *, session: Optional[requests.Session] = None) -> str:
    from pypdf import PdfReader

    http = session or requests.Session()
    response = http.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    reader = PdfReader(BytesIO(response.content))
    texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(texts)


def _clean_report_lines(text: str) -> List[str]:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.startswith("NBA Injury Report"):
            continue
        if line.startswith("League") or line.startswith("Page "):
            continue
        if line.startswith("Team Injury Reports"):
            continue
        lines.append(line)
    return lines


def _canonical_full_team_names() -> List[str]:
    names = []
    for normalized, abbr in team_name_mappings().items():
        team_name = canonical_team_name_from_abbrev(abbr)
        if team_name:
            names.append(team_name)
    return sorted(set(names), key=len, reverse=True)


def _parse_team_entry(rest: str, team_names: Iterable[str]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    stripped = str(rest or "").strip()
    if not stripped:
        return None, None, None, None

    team_name = None
    for candidate in team_names:
        if stripped.startswith(candidate):
            team_name = candidate
            break
    if team_name is None:
        return None, None, None, None

    remainder = stripped[len(team_name):].strip()
    if not remainder:
        return team_name, "", "", ""

    if remainder.upper().startswith("NOT YET SUBMITTED"):
        return team_name, "", "NOT YET SUBMITTED", ""

    match = PLAYER_STATUS_RE.match(remainder)
    if not match:
        return team_name, remainder, "INACTIVE_OTHER", ""
    return (
        team_name,
        match.group("player").strip(),
        match.group("status").strip(),
        match.group("reason").strip(" -"),
    )


def parse_official_injury_report_text(
    text: str,
    *,
    source_url: str,
    pulled_at: str,
) -> pd.DataFrame:
    team_names = _canonical_full_team_names()
    collapsed_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if TOKENIZED_PAGE_HEADER_RE.search(collapsed_text) and GAME_HEADER_RE.search(collapsed_text):
        tokenized_df = _parse_tokenized_official_injury_report_text(
            text,
            source_url=source_url,
            pulled_at=pulled_at,
            team_names=team_names,
        )
        if not tokenized_df.empty:
            return tokenized_df

    rows: List[dict] = []
    current_game_date = ""
    current_game_time = ""
    current_matchup = ""
    last_row_index: Optional[int] = None

    for line in _clean_report_lines(text):
        parsed = TEAM_ROW_RE.match(line)
        if not parsed:
            if last_row_index is not None:
                rows[last_row_index]["raw_reason"] = (rows[last_row_index]["raw_reason"] + " " + line).strip()
            continue

        if parsed.group("game_date"):
            current_game_date = datetime.strptime(parsed.group("game_date"), "%m/%d/%Y").date().isoformat()
        if parsed.group("game_time"):
            current_game_time = parsed.group("game_time")
        if parsed.group("matchup"):
            current_matchup = parsed.group("matchup")

        rest = parsed.group("rest").strip()
        team_name, player_name, raw_status, raw_reason = _parse_team_entry(rest, team_names)
        if team_name is None:
            if last_row_index is not None:
                rows[last_row_index]["raw_reason"] = (rows[last_row_index]["raw_reason"] + " " + rest).strip()
            continue

        team_abbrev = canonical_team_abbrev(team_name) or team_name
        game_id = stable_id(current_game_date, current_matchup, prefix="game")
        normalized_status = normalize_report_status(raw_status)
        row_kind = "team_status" if not player_name else "player_status"

        rows.append(
            {
                "game_id": game_id,
                "game_date": current_game_date,
                "report_date": current_game_date,
                "report_time_et": current_game_time,
                "matchup": current_matchup,
                "row_kind": row_kind,
                "player_name": player_name,
                "team_abbrev": team_abbrev,
                "report_status": raw_status.lower() if raw_status else "",
                "raw_status": raw_status,
                "raw_reason": raw_reason.strip(),
                "normalized_status": normalized_status,
                "projected_availability": projected_availability_for_status(normalized_status),
                "source_url": source_url,
                "source": "official_nba",
                "reported_at": pulled_at,
                "pulled_at": pulled_at,
            }
        )
        last_row_index = len(rows) - 1

    if rows:
        return pd.DataFrame(rows)
    return _parse_tokenized_official_injury_report_text(
        text,
        source_url=source_url,
        pulled_at=pulled_at,
        team_names=team_names,
    )


def _empty_injury_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "game_date",
            "report_date",
            "report_time_et",
            "matchup",
            "row_kind",
            "player_name",
            "team_abbrev",
            "report_status",
            "raw_status",
            "raw_reason",
            "normalized_status",
            "projected_availability",
            "source_url",
            "source",
            "reported_at",
            "pulled_at",
        ]
    )


def _build_injury_row(
    *,
    game_date: str,
    game_time: str,
    matchup: str,
    team_name: str,
    player_name: str,
    raw_status: str,
    raw_reason: str,
    source_url: str,
    pulled_at: str,
) -> dict:
    team_abbrev = canonical_team_abbrev(team_name) or team_name
    normalized_status = normalize_report_status(raw_status)
    return {
        "game_id": stable_id(game_date, matchup, prefix="game"),
        "game_date": game_date,
        "report_date": game_date,
        "report_time_et": game_time,
        "matchup": matchup,
        "row_kind": "team_status" if not player_name else "player_status",
        "player_name": player_name,
        "team_abbrev": team_abbrev,
        "report_status": raw_status.lower() if raw_status else "",
        "raw_status": raw_status,
        "raw_reason": raw_reason.strip(),
        "normalized_status": normalized_status,
        "projected_availability": projected_availability_for_status(normalized_status),
        "source_url": source_url,
        "source": "official_nba",
        "reported_at": pulled_at,
        "pulled_at": pulled_at,
    }


def _parse_tokenized_official_injury_report_text(
    text: str,
    *,
    source_url: str,
    pulled_at: str,
    team_names: Iterable[str],
) -> pd.DataFrame:
    raw_tokens = re.sub(r"\s+", " ", str(text or "")).strip().split()
    if not raw_tokens:
        return _empty_injury_frame()

    header_tokens = ["Game", "Date", "Game", "Time", "Matchup", "Team", "Player", "Name", "Current", "Status", "Reason"]
    tokens: List[str] = []
    index = 0
    while index < len(raw_tokens):
        if (
            index + 8 < len(raw_tokens)
            and raw_tokens[index] == "Injury"
            and raw_tokens[index + 1] == "Report:"
            and re.match(r"^\d{2}/\d{2}/\d{2}$", raw_tokens[index + 2])
            and re.match(r"^\d{1,2}:\d{2}$", raw_tokens[index + 3])
            and raw_tokens[index + 4] in {"AM", "PM"}
            and raw_tokens[index + 5] == "Page"
            and raw_tokens[index + 7] == "of"
            and re.match(r"^\d+$", raw_tokens[index + 8])
        ):
            index += 9
            continue
        if raw_tokens[index:index + len(header_tokens)] == header_tokens:
            index += len(header_tokens)
            continue
        tokens.append(raw_tokens[index])
        index += 1

    if not tokens:
        return _empty_injury_frame()

    team_token_map = [
        (team_name, team_name.split())
        for team_name in team_names
    ]

    def is_date_token(token: str) -> bool:
        return bool(re.match(r"^\d{2}/\d{2}/\d{4}$", token))

    def is_time_token(token: str) -> bool:
        return bool(re.match(r"^\d{1,2}:\d{2}$", token))

    def is_matchup_token(token: str) -> bool:
        return bool(re.match(r"^[A-Z]{2,3}@[A-Z]{2,3}$", token))

    def is_name_token(token: str) -> bool:
        return bool(re.match(r"^(?=.*[A-Za-z])[A-Za-z'’.\-]+,?$", token))

    def is_game_header(index: int) -> bool:
        return (
            index + 3 < len(tokens)
            and is_date_token(tokens[index])
            and is_time_token(tokens[index + 1])
            and tokens[index + 2] == "(ET)"
            and is_matchup_token(tokens[index + 3])
        )

    def is_time_matchup_header(index: int) -> bool:
        return (
            index + 2 < len(tokens)
            and is_time_token(tokens[index])
            and tokens[index + 1] == "(ET)"
            and is_matchup_token(tokens[index + 2])
        )

    def is_matchup_only_header(index: int) -> bool:
        return is_matchup_token(tokens[index])

    def match_team(index: int) -> tuple[Optional[str], int]:
        for team_name, team_tokens in team_token_map:
            if tokens[index:index + len(team_tokens)] == team_tokens:
                return team_name, len(team_tokens)
        return None, 0

    def is_not_yet_submitted(index: int) -> bool:
        return tokens[index:index + 3] == ["NOT", "YET", "SUBMITTED"]

    def player_status_index(index: int) -> Optional[int]:
        first_token = tokens[index].strip(",").lower()
        if first_token in NON_NAME_START_TOKENS:
            return None
        max_index = min(index + 7, len(tokens))
        for candidate in range(index + 1, max_index):
            if tokens[candidate] not in STATUS_TOKENS:
                continue
            name_tokens = tokens[index:candidate]
            if not name_tokens or not any("," in token for token in name_tokens):
                continue
            if all(is_name_token(token) for token in name_tokens):
                return candidate
        return None

    def next_boundary(index: int) -> int:
        for candidate in range(index, len(tokens)):
            if is_game_header(candidate):
                return candidate
            if is_time_matchup_header(candidate):
                return candidate
            if is_matchup_only_header(candidate):
                return candidate
            team_name, _ = match_team(candidate)
            if team_name is not None:
                return candidate
            if is_not_yet_submitted(candidate):
                return candidate
            if player_status_index(candidate) is not None:
                return candidate
        return len(tokens)

    rows: List[dict] = []
    current_game_date = ""
    current_game_time = ""
    current_matchup = ""
    index = 0
    while index < len(tokens):
        if is_game_header(index):
            current_game_date = datetime.strptime(tokens[index], "%m/%d/%Y").date().isoformat()
            current_game_time = tokens[index + 1]
            current_matchup = tokens[index + 3]
            index += 4
            continue
        if is_time_matchup_header(index):
            current_game_time = tokens[index]
            current_matchup = tokens[index + 2]
            index += 3
            continue
        if is_matchup_only_header(index):
            current_matchup = tokens[index]
            index += 1
            continue

        team_name, team_token_count = match_team(index)
        if team_name is None:
            index += 1
            continue
        index += team_token_count

        if is_not_yet_submitted(index):
            rows.append(
                _build_injury_row(
                    game_date=current_game_date,
                    game_time=current_game_time,
                    matchup=current_matchup,
                    team_name=team_name,
                    player_name="",
                    raw_status="NOT YET SUBMITTED",
                    raw_reason="",
                    source_url=source_url,
                    pulled_at=pulled_at,
                )
            )
            index += 3
            continue

        while index < len(tokens):
            if is_game_header(index):
                break
            next_team_name, _ = match_team(index)
            if next_team_name is not None:
                break
            if is_not_yet_submitted(index):
                break

            status_index = player_status_index(index)
            if status_index is None:
                index += 1
                continue

            player_name = " ".join(tokens[index:status_index]).strip()
            raw_status = tokens[status_index]
            reason_start = status_index + 1
            reason_end = next_boundary(reason_start)
            raw_reason = " ".join(tokens[reason_start:reason_end]).strip()
            rows.append(
                _build_injury_row(
                    game_date=current_game_date,
                    game_time=current_game_time,
                    matchup=current_matchup,
                    team_name=team_name,
                    player_name=player_name,
                    raw_status=raw_status,
                    raw_reason=raw_reason,
                    source_url=source_url,
                    pulled_at=pulled_at,
                )
            )
            index = reason_end

    if not rows:
        return _empty_injury_frame()
    return pd.DataFrame(rows)


def fetch_official_injury_reports(
    *,
    report_date: date,
    latest_only: bool = True,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    http = session or requests.Session()
    index_url = official_report_index_url(report_date)
    response = http.get(index_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    links = extract_pdf_links(response.text, report_date=report_date)
    if latest_only and links:
        links = [links[-1]]

    frames = []
    pulled_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for link in links:
        text = fetch_pdf_text(link, session=http)
        report_df = parse_official_injury_report_text(text, source_url=link, pulled_at=pulled_at)
        if not report_df.empty:
            report_df["report_date"], report_df["report_time_et"] = parse_report_timestamp_from_url(link)
        frames.append(report_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def persist_official_injury_reports(
    df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
    database_url: Optional[str] = None,
) -> int:
    if df.empty:
        return 0

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=[
                    "source_url",
                    "game_id",
                    "team_abbrev",
                    "player_name",
                    "report_status",
                    "raw_reason",
                ],
                keep="last",
            )
        else:
            combined = df.copy()
        combined.to_csv(output_path, index=False)

    init_database(database_url)
    with session_scope(database_url) as session:
        urls = sorted({str(value) for value in df["source_url"].dropna().unique().tolist()})
        if urls:
            session.execute(delete(InjuryReportRecord).where(InjuryReportRecord.source_url.in_(urls)))

        count = 0
        for row in df.fillna("").to_dict(orient="records"):
            session.add(
                InjuryReportRecord(
                    game_id=str(row.get("game_id") or ""),
                    game_date=str(row.get("game_date") or ""),
                    report_date=str(row.get("report_date") or ""),
                    report_time_et=str(row.get("report_time_et") or ""),
                    matchup=str(row.get("matchup") or ""),
                    row_kind=str(row.get("row_kind") or "player_status"),
                    player_name=str(row.get("player_name") or ""),
                    team_abbrev=str(row.get("team_abbrev") or ""),
                    report_status=str(row.get("report_status") or ""),
                    raw_status=str(row.get("raw_status") or ""),
                    raw_reason=str(row.get("raw_reason") or ""),
                    normalized_status=str(row.get("normalized_status") or ""),
                    projected_availability=str(row.get("projected_availability") or ""),
                    source_url=str(row.get("source_url") or ""),
                    source=str(row.get("source") or "official_nba"),
                    reported_at=str(row.get("reported_at") or ""),
                    pulled_at=str(row.get("pulled_at") or ""),
                )
            )
            count += 1
    return count
