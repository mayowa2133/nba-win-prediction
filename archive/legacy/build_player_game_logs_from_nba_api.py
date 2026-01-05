#!/usr/bin/env python
"""
build_player_game_logs_from_nba_api.py

Fetch per-game **player** box score stats from the NBA Stats API
via `nba_api` and build data/player_game_logs.csv in the same
column format we used before.

Usage:
    python build_player_game_logs_from_nba_api.py

Requirements:
    pip install nba_api pandas
"""

import time
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# First season start year you care about.
# 2015 -> "2015-16", 2016 -> "2016-17", etc.
START_SEASON_START_YEAR = 2015

OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "player_game_logs.csv"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def get_current_season_start_year(today: Optional[dt.date] = None) -> int:
    """
    Infer the most recent NBA season start year.

    NBA seasons are roughly:
        - Start: Oct of year Y
        - End: Apr/Jun of year Y+1

    If we're in Oct or later, current season start year is this year.
    Otherwise, it's last year.
    """
    if today is None:
        today = dt.date.today()

    if today.month >= 10:
        return today.year
    else:
        return today.year - 1


def iter_season_strings(start_year: int, end_year: int) -> List[str]:
    """
    Convert start years into NBA season strings, e.g.
      2015 -> "2015-16"
      2016 -> "2016-17"
    """
    seasons = []
    for year in range(start_year, end_year + 1):
        seasons.append(f"{year}-{str(year + 1)[-2:]}")
    return seasons


def parse_minutes(min_str: Any) -> float:
    """
    Convert 'MM:SS' to float minutes. If None/empty/invalid, return 0.0.
    """
    if min_str is None:
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)

    s = str(min_str).strip()
    if s == "":
        return 0.0

    # Sometimes it's "38" (whole minutes) instead of "38:15"
    if ":" not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0

    parts = s.split(":")
    if len(parts) != 2:
        return 0.0

    try:
        m = int(parts[0])
        sec = int(parts[1])
        return m + sec / 60.0
    except ValueError:
        return 0.0


def parse_matchup(matchup: str, team_abbrev: str) -> Tuple[Optional[str], Optional[int]]:
    """
    From a MATCHUP string like:
        "GSW vs. CLE"  -> GSW home vs CLE
        "LAL @ BOS"    -> LAL away @ BOS
    infer:
        opp_abbrev, is_home (1 for home, 0 for away, None if unknown)
    """
    if matchup is None:
        return None, None

    m = str(matchup)

    # Most common formats seen in nba_api:
    #   "GSW vs. CLE" or "GSW vs CLE"
    #   "LAL @ BOS"
    sep_home = " vs. "
    sep_home_alt = " vs "
    sep_away = " @ "

    is_home = None
    left = None
    right = None

    if sep_home in m:
        left, right = m.split(sep_home, 1)
        is_home = 1
    elif sep_home_alt in m:
        left, right = m.split(sep_home_alt, 1)
        is_home = 1
    elif sep_away in m:
        left, right = m.split(sep_away, 1)
        is_home = 0

    if left is None or right is None:
        return None, None

    left = left.strip()
    right = right.strip()

    # Sanity: team_abbrev should be one of the two sides;
    # pick the *other* one as opponent.
    if team_abbrev == left:
        opp = right
    elif team_abbrev == right:
        opp = left
    else:
        # Fallback – assume left is this team, right is opp.
        opp = right

    return opp, is_home


def fetch_season_raw(season_str: str) -> pd.DataFrame:
    """
    Call NBA Stats 'LeagueGameLog' in player mode for a specific season.
    Returns the raw DataFrame from nba_api.
    """
    print(f"\n=== Fetching player game logs for season {season_str} ===")

    # This hits stats.nba.com via nba_api
    gl = LeagueGameLog(
        season=season_str,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",  # 'P' = players, 'T' = teams
    )
    df = gl.get_data_frames()[0]
    print(f"  -> got {len(df)} rows")
    return df


def transform_raw_to_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform nba_api LeagueGameLog output into our desired schema:

    Columns:
      game_id, season, game_date,
      player_id, player_name,
      team_abbrev, opp_abbrev, is_home,
      minutes,
      pts, reb, ast, stl, blk, tov,
      fg3m, fg3a, fga, fgm, fta, ftm, pf,
      team_score, opp_score,
      spread_close, total_close, ml_team, ml_opp
    """
    df = df_raw.copy()

    # season: pull last 4 digits from SEASON_ID (e.g. '22023' -> 2023)
    if "SEASON_ID" in df.columns:
        df["season"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)
    else:
        # Fallback if SEASON_ID ever missing
        df["season"] = None

    # game_date: store as 'YYYY-MM-DD'
    df["game_date"] = pd.to_datetime(df["GAME_DATE"]).dt.date.astype(str)

    # minutes
    df["minutes"] = df["MIN"].apply(parse_minutes)

    # opp_abbrev & is_home from MATCHUP
    opps: List[Optional[str]] = []
    homes: List[Optional[int]] = []
    for _, row in df.iterrows():
        team_abbrev = row.get("TEAM_ABBREVIATION", None)
        matchup = row.get("MATCHUP", None)
        opp_abbrev, is_home = parse_matchup(matchup, str(team_abbrev) if team_abbrev else "")
        opps.append(opp_abbrev)
        homes.append(is_home)

    df["opp_abbrev"] = opps
    df["is_home"] = homes

    # Map to our output columns
    col_map = {
        "game_id": "GAME_ID",
        "season": "season",
        "game_date": "game_date",
        "player_id": "PLAYER_ID",
        "player_name": "PLAYER_NAME",
        "team_abbrev": "TEAM_ABBREVIATION",
        "opp_abbrev": "opp_abbrev",
        "is_home": "is_home",
        "minutes": "minutes",
        "pts": "PTS",
        "reb": "REB",
        "ast": "AST",
        "stl": "STL",
        "blk": "BLK",
        "tov": "TOV",
        "fg3m": "FG3M",
        "fg3a": "FG3A",
        "fga": "FGA",
        "fgm": "FGM",
        "fta": "FTA",
        "ftm": "FTM",
        "pf": "PF",
        "oreb": "OREB",
    }

    missing = [src for src in col_map.values() if src not in df.columns]
    if missing:
        print("WARNING: missing expected columns from nba_api response:", missing)

    df_out = df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})

    # We don't have team final scores or betting lines from nba_api here,
    # so fill them as None for now. You can merge them later from another source.
    df_out["team_score"] = None
    df_out["opp_score"] = None
    df_out["spread_close"] = None
    df_out["total_close"] = None
    df_out["ml_team"] = None
    df_out["ml_opp"] = None

    return df_out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    last_start_year = get_current_season_start_year()
    seasons = iter_season_strings(START_SEASON_START_YEAR, last_start_year)
    print("Seasons to fetch:", ", ".join(seasons))

    all_frames: List[pd.DataFrame] = []

    for season_str in seasons:
        df_raw = fetch_season_raw(season_str)
        if df_raw.empty:
            print(f"  (no data returned for {season_str}, skipping)")
            continue

        df_season = transform_raw_to_schema(df_raw)
        all_frames.append(df_season)

        # Small delay between seasons to be nice to the API
        time.sleep(1.0)

    if not all_frames:
        print("No data fetched. Check your network / nba_api / season config.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)

    # Sort nicely
    df_all = df_all.sort_values(
        by=["season", "game_date", "team_abbrev", "player_name", "game_id"]
    ).reset_index(drop=True)

    # Save
    df_all.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df_all)} player-game rows to {OUT_CSV}")

    print("\nSample rows:")
    print(df_all.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
