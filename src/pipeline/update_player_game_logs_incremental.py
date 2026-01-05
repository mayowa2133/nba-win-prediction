#!/usr/bin/env python
"""
update_player_game_logs_incremental.py

Lightweight updater for data/player_game_logs.csv.

- Assumes build_player_game_logs_from_nba_api.py has already created
  data/player_game_logs.csv at least once.
- Only fetches **new games** for the current NBA season using nba_api
  and appends them to the existing CSV.

Usage:
    python update_player_game_logs_incremental.py
"""

import datetime as dt
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

# Constants and helpers (previously from build_player_game_logs_from_nba_api)
OUT_CSV = Path("data/player_game_logs.csv")


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


def parse_minutes(min_str) -> float:
    """Convert 'MM:SS' to float minutes."""
    if min_str is None:
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    s = str(min_str).strip()
    if s == "":
        return 0.0
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
    
    # Sanity: team_abbrev should be one of the two sides; pick the *other* one as opponent.
    if team_abbrev == left:
        opp = right
    elif team_abbrev == right:
        opp = left
    else:
        # Fallback – assume left is this team, right is opp.
        opp = right
    
    return opp, is_home


def transform_raw_to_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform nba_api LeagueGameLog output into our desired schema.
    
    Columns:
      game_id, season, game_date,
      player_id, player_name,
      team_abbrev, opp_abbrev, is_home,
      minutes,
      pts, reb, ast, stl, blk, tov,
      fg3m, fg3a, fga, fgm, fta, ftm, pf,
    """
    df = df_raw.copy()
    
    # season: pull last 4 digits from SEASON_ID (e.g. '22023' -> 2023)
    if "SEASON_ID" in df.columns:
        df["season"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)
    else:
        df["season"] = None
    
    # game_date: store as 'YYYY-MM-DD' string
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
    }
    
    missing = [src for src in col_map.values() if src not in df.columns and src not in ["season", "game_date", "opp_abbrev", "is_home", "minutes"]]
    if missing:
        print("WARNING: missing expected columns from nba_api response:", missing)
    
    df_out = df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})
    
    return df_out


def season_start_date_for_year(start_year: int) -> dt.date:
    """
    Rough guess for season start date; NBA regular season typically starts in October.
    We use Oct 1 of start_year as a safe lower bound.
    """
    return dt.date(start_year, 10, 1)


def season_int_to_str(season_start_year: int) -> str:
    """
    Convert an integer season start year (e.g. 2025) to NBA season string "2025-26".
    """
    return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"


def fetch_new_games_for_season(
    season_start_year: int,
    date_from: dt.date,
    date_to: dt.date,
) -> pd.DataFrame:
    """
    Call nba_api.LeagueGameLog for the given season & date range
    in PLAYER mode and return the raw DataFrame.
    """
    season_str = season_int_to_str(season_start_year)
    print(
        f"\n=== Fetching incremental player logs for season {season_str} "
        f"from {date_from} to {date_to} ==="
    )

    # nba_api expects dates as MM/DD/YYYY strings
    date_from_str = date_from.strftime("%m/%d/%Y")
    date_to_str = date_to.strftime("%m/%d/%Y")

    gl = LeagueGameLog(
        season=season_str,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
        date_from_nullable=date_from_str,
        date_to_nullable=date_to_str,
    )
    df_raw = gl.get_data_frames()[0]
    print(f"  -> got {len(df_raw)} rows")
    return df_raw


def main():
    # Ensure base CSV exists
    if not OUT_CSV.exists():
        print(
            f"Existing logs file {OUT_CSV} not found.\n"
            "Run build_player_game_logs_from_nba_api.py once to create it."
        )
        return

    df_existing = pd.read_csv(OUT_CSV)
    if df_existing.empty:
        print(
            f"{OUT_CSV} is empty. Run the full build script first to seed historical data."
        )
        return

    # Figure out the current NBA season start year (e.g. 2025 → season "2025-26")
    current_season_start_year = get_current_season_start_year()
    print(f"Current season start year inferred as {current_season_start_year}")

    # Check if we already have rows for this season
    if "season" not in df_existing.columns:
        print("Existing logs do not have a 'season' column — something is off.")
        return

    df_existing["season"] = df_existing["season"].astype(int)
    df_current_season = df_existing[df_existing["season"] == current_season_start_year]

    today = dt.date.today()

    if df_current_season.empty:
        # No games yet stored for this season: start from season start date
        date_from = season_start_date_for_year(current_season_start_year)
        print(
            f"No rows found for season {current_season_start_year} yet. "
            f"Will fetch from {date_from} to {today}."
        )
    else:
        # We already have some games; fetch only after the latest stored game_date
        df_current_season["game_date_parsed"] = pd.to_datetime(
            df_current_season["game_date"]
        )
        last_game_date = df_current_season["game_date_parsed"].max().date()
        print(
            f"Latest stored game_date for season {current_season_start_year}: "
            f"{last_game_date}"
        )
        if last_game_date >= today:
            print(
                "Logs already up-to-date for the current season "
                f"(last_game_date={last_game_date}, today={today}). Nothing to do."
            )
            return
        date_from = last_game_date + dt.timedelta(days=1)
        print(f"Will fetch new games from {date_from} to {today}.")

    # If date_from is after today, nothing to fetch
    if date_from > today:
        print(
            f"date_from ({date_from}) is after today ({today}); nothing to fetch."
        )
        return

    # Fetch new games for the current season
    df_new_raw = fetch_new_games_for_season(
        season_start_year=current_season_start_year,
        date_from=date_from,
        date_to=today,
    )

    if df_new_raw.empty:
        print("No new player logs returned for that date range. Nothing to append.")
        return

    # Transform into our schema using the same function as the full builder
    df_new = transform_raw_to_schema(df_new_raw)

    # Combine and drop duplicates (in case of overlap)
    combined = pd.concat([df_existing, df_new], ignore_index=True)

    # A safe uniqueness key is (season, game_id, player_id)
    if not {"season", "game_id", "player_id"}.issubset(combined.columns):
        print(
            "Missing one of ['season', 'game_id', 'player_id'] in combined logs "
            "— cannot safely de-duplicate."
        )
        return

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["season", "game_id", "player_id"], keep="last"
    ).reset_index(drop=True)
    after = len(combined)
    print(f"De-duplicated rows: {before} -> {after}")

    # Sort in the same way as the full builder
    sort_cols: List[str] = [
        "season",
        "game_date",
        "team_abbrev",
        "player_name",
        "game_id",
    ]
    existing_cols = [c for c in sort_cols if c in combined.columns]
    combined = combined.sort_values(by=existing_cols).reset_index(drop=True)

    # Save back
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nUpdated logs saved to {OUT_CSV}")
    print("Sample of newest rows:")
    print(
        combined.tail(10)[
            ["season", "game_date", "team_abbrev", "player_name", "pts", "minutes"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()