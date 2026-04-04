"""Helpers for normalizing NBA team names and abbreviations."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, Optional


def _normalize(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace(".", "")
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def team_name_mappings() -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    try:
        from nba_api.stats.static import teams as nba_teams

        for team in nba_teams.get_teams():
            abbr = str(team.get("abbreviation", "")).upper()
            full_name = str(team.get("full_name", ""))
            city = str(team.get("city", ""))
            nickname = str(team.get("nickname", ""))
            if not abbr:
                continue
            for value in {abbr, full_name, city, nickname, f"{city} {nickname}".strip()}:
                normalized = _normalize(value)
                if normalized:
                    mapping[normalized] = abbr
    except Exception:
        fallback = {
            "atlanta hawks": "ATL",
            "boston celtics": "BOS",
            "brooklyn nets": "BKN",
            "charlotte hornets": "CHA",
            "chicago bulls": "CHI",
            "cleveland cavaliers": "CLE",
            "dallas mavericks": "DAL",
            "denver nuggets": "DEN",
            "detroit pistons": "DET",
            "golden state warriors": "GSW",
            "houston rockets": "HOU",
            "indiana pacers": "IND",
            "la clippers": "LAC",
            "los angeles clippers": "LAC",
            "los angeles lakers": "LAL",
            "la lakers": "LAL",
            "memphis grizzlies": "MEM",
            "miami heat": "MIA",
            "milwaukee bucks": "MIL",
            "minnesota timberwolves": "MIN",
            "new orleans pelicans": "NOP",
            "new york knicks": "NYK",
            "oklahoma city thunder": "OKC",
            "orlando magic": "ORL",
            "philadelphia 76ers": "PHI",
            "phoenix suns": "PHX",
            "portland trail blazers": "POR",
            "sacramento kings": "SAC",
            "san antonio spurs": "SAS",
            "toronto raptors": "TOR",
            "utah jazz": "UTA",
            "washington wizards": "WAS",
        }
        mapping.update({_normalize(name): abbr for name, abbr in fallback.items()})
        mapping.update({abbr.lower(): abbr for abbr in fallback.values()})

    extras = {
        "brooklyn": "BKN",
        "brooklyn nets": "BKN",
        "new jersey nets": "BKN",
        "new orleans": "NOP",
        "new orleans hornets": "NOP",
        "new orleans oklahoma city hornets": "NOP",
        "okc": "OKC",
        "oklahoma city": "OKC",
        "seattle supersonics": "OKC",
        "seattle": "OKC",
        "ny knicks": "NYK",
        "76ers": "PHI",
        "sixers": "PHI",
        "trail blazers": "POR",
        "wolves": "MIN",
        "charlotte bobcats": "CHA",
        "vancouver grizzlies": "MEM",
        "njn": "BKN",
        "sea": "OKC",
        "nok": "NOP",
        "cho": "CHA",
    }
    mapping.update({_normalize(name): abbr for name, abbr in extras.items()})
    return mapping


def canonical_team_abbrev(value: str | None) -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize(value)
    if not normalized:
        return None
    if len(normalized) == 3 and normalized.isalpha():
        return normalized.upper()
    return team_name_mappings().get(normalized)


def canonical_team_name_from_abbrev(abbrev: str | None) -> Optional[str]:
    if not abbrev:
        return None
    abbr = str(abbrev).upper()
    try:
        from nba_api.stats.static import teams as nba_teams

        for team in nba_teams.get_teams():
            if str(team.get("abbreviation", "")).upper() == abbr:
                return str(team.get("full_name", ""))
    except Exception:
        pass
    return abbr


def match_team_prefix(line: str, team_names: Iterable[str]) -> Optional[str]:
    normalized_line = _normalize(line)
    candidates = sorted({_normalize(name) for name in team_names if name}, key=len, reverse=True)
    for candidate in candidates:
        if normalized_line.startswith(candidate):
            return candidate
    return None
