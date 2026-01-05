#!/usr/bin/env python
"""
build_player_game_logs_from_balldontlie.py

Fetch per-game player stats from the balldontlie API and build
data/player_game_logs.csv in the format expected by our prop scripts.

Usage:
    python build_player_game_logs_from_balldontlie.py
"""

import os
import time
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

import requests
import pandas as pd


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Seasons you want (balldontlie uses season start year, e.g. 2018 = 2018-19)
SEASONS = list(range(2015, 2026))

# Where to save
OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "player_game_logs.csv"

# balldontlie base URL
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# API key from env
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")

# Simple team name -> abbrev map (you can extend if needed)
TEAM_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def bdl_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET wrapper for balldontlie with optional API key.

    Tries two auth styles:
      1) Authorization: Bearer <KEY>
      2) Authorization: <KEY>

    If both fail (401 or other error), prints debug info and raises.
    """
    url = f"{BALLDONTLIE_BASE}/{path}"

    # If you somehow run without a key, warn loudly
    if not BALLDONTLIE_API_KEY:
        print("\n[balldontlie WARNING] BALLDONTLIE_API_KEY is not set in the environment.")
        print("Requests will be sent WITHOUT an Authorization header — most endpoints will 401.")
    
    auth_styles = []
    if BALLDONTLIE_API_KEY:
        # Try Bearer first, then raw
        auth_styles = ["bearer", "raw"]
    else:
        # No key → one attempt with no auth header at all
        auth_styles = ["none"]

    last_resp = None

    for style in auth_styles:
        headers: Dict[str, str] = {}

        if style == "bearer":
            headers["Authorization"] = f"Bearer {BALLDONTLIE_API_KEY}"
        elif style == "raw":
            headers["Authorization"] = BALLDONTLIE_API_KEY
        # style == "none" → no Authorization header at all

        resp = requests.get(url, params=params, headers=headers, timeout=20)
        last_resp = resp

        # If we got 401 using "Bearer", automatically retry with raw key
        if resp.status_code == 401 and style == "bearer" and BALLDONTLIE_API_KEY:
            print("Got 401 with 'Bearer <API_KEY>'. Retrying once with 'Authorization: <API_KEY>' ...")
            continue

        # Any 4xx/5xx → print debug, then raise
        if resp.status_code >= 400:
            print("\n[balldontlie ERROR]")
            print(f"  URL:    {resp.url}")
            print(f"  Params: {params}")
            print(f"  Status: {resp.status_code}")
            try:
                print("  Body (JSON):", resp.json())
            except Exception:
                print("  Body (text):", resp.text[:500])
            resp.raise_for_status()

        # Success
        return resp.json()

    # If we somehow exit the loop without returning (e.g. repeated 401s),
    # raise on the last response.
    if last_resp is not None:
        print("\n[balldontlie ERROR] All auth styles tried, still unauthorized.")
        print(f"  URL:    {last_resp.url}")
        print(f"  Status: {last_resp.status_code}")
        try:
            print("  Body (JSON):", last_resp.json())
        except Exception:
            print("  Body (text):", last_resp.text[:500])
        last_resp.raise_for_status()
    else:
        raise RuntimeError("bdl_get: no HTTP response obtained.")


def parse_minutes(min_str: str) -> float:
    """Convert 'MM:SS' to float minutes. If None/empty, return 0."""
    if not min_str:
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    parts = str(min_str).split(":")
    if len(parts) != 2:
        return 0.0
    try:
        m = int(parts[0])
        s = int(parts[1])
        return m + s / 60.0
    except ValueError:
        return 0.0


def extract_logs_for_season(season: int) -> List[Dict[str, Any]]:
    """
    Pull all regular-season player game logs for a season from balldontlie.
    Returns list of dict rows for our CSV.
    """
    print(f"\n=== Season {season} ===")

    page = 1
    per_page = 100
    all_rows: List[Dict[str, Any]] = []

    while True:
        params = {
            "seasons[]": season,
            "per_page": per_page,
            "page": page,
            "postseason": "false",  # regular season only
        }
        data = bdl_get("stats", params)
        stats = data.get("data", [])
        meta = data.get("meta", {})

        if not stats:
            break

        print(f"  Season {season}: page {page}/{meta.get('total_pages','?')} -> {len(stats)} rows")

        for s in stats:
            game = s.get("game", {}) or {}
            team = s.get("team", {}) or {}
            player = s.get("player", {}) or {}

            # Game metadata
            game_id = game.get("id")
            game_date_str = game.get("date")  # e.g. "2019-10-22T00:00:00.000Z"
            try:
                game_dt = dt.datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                game_date = game_dt.date().isoformat()
            except Exception:
                game_date = None

            home_team_id = game.get("home_team_id")
            visitor_team_id = game.get("visitor_team_id")
            home_score = game.get("home_team_score")
            visitor_score = game.get("visitor_team_score")

            team_id = team.get("id")
            team_name = team.get("full_name") or team.get("name")
            team_abbrev = team.get("abbreviation")

            # Derive opponent info and is_home flag
            if team_id == home_team_id:
                is_home = 1
                opp_team_id = visitor_team_id
                team_score = home_score
                opp_score = visitor_score
            else:
                is_home = 0
                opp_team_id = home_team_id
                team_score = visitor_score
                opp_score = home_score

            # Try to get opponent abbrev from embedded home_team/visitor_team if present
            home_team = game.get("home_team") or {}
            visitor_team = game.get("visitor_team") or {}
            home_abbrev = home_team.get("abbreviation") or TEAM_NAME_TO_ABBREV.get(home_team.get("full_name", ""), None)
            visitor_abbrev = visitor_team.get("abbreviation") or TEAM_NAME_TO_ABBREV.get(visitor_team.get("full_name", ""), None)

            if team_id == home_team_id:
                opp_abbrev = visitor_abbrev
            else:
                opp_abbrev = home_abbrev

            # Player info
            player_id = player.get("id")
            first_name = player.get("first_name") or ""
            last_name = player.get("last_name") or ""
            player_name = (first_name + " " + last_name).strip()

            # Box score stats
            min_played = parse_minutes(s.get("min"))
            pts = s.get("pts", 0)
            reb = s.get("reb", 0)
            ast = s.get("ast", 0)
            stl = s.get("stl", 0)
            blk = s.get("blk", 0)
            tov = s.get("turnover", 0)
            fg3m = s.get("fg3m", 0)
            fg3a = s.get("fg3a", 0)
            fga = s.get("fga", 0)
            fgm = s.get("fgm", 0)
            fta = s.get("fta", 0)
            ftm = s.get("ftm", 0)
            pf = s.get("pf", 0)

            row = {
                "game_id": game_id,
                "season": season,
                "game_date": game_date,
                "player_id": player_id,
                "player_name": player_name,
                "team_abbrev": team_abbrev,
                "opp_abbrev": opp_abbrev,
                "is_home": is_home,
                "minutes": min_played,
                "pts": pts,
                "reb": reb,
                "ast": ast,
                "stl": stl,
                "blk": blk,
                "tov": tov,
                "fg3m": fg3m,
                "fg3a": fg3a,
                "fga": fga,
                "fgm": fgm,
                "fta": fta,
                "ftm": ftm,
                "pf": pf,
                "team_score": team_score,
                "opp_score": opp_score,
                "spread_close": None,
                "total_close": None,
                "ml_team": None,
                "ml_opp": None,
            }
            all_rows.append(row)

        total_pages = meta.get("total_pages", page)
        if page >= total_pages:
            break

        page += 1
        # small pause to be nice to the API
        time.sleep(0.2)

    return all_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for season in SEASONS:
        rows = extract_logs_for_season(season)
        all_rows.extend(rows)

    if not all_rows:
        print("No rows fetched. Check API key, base URL, or seasons config.")
        return

    df = pd.DataFrame(all_rows)

    # Basic sanity checks
    print(f"\nFetched {len(df)} player-game rows across seasons {SEASONS[0]}–{SEASONS[-1]}")
    print("Columns:", list(df.columns))

    # Sort for nice ordering
    df = df.sort_values(by=["season", "game_date", "team_abbrev", "player_name"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved player game logs to {OUT_CSV}")

    # Show a small sample
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
