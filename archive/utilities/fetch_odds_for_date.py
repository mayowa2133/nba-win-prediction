#!/usr/bin/env python
"""
fetch_odds_for_date.py

Fetch NBA odds (moneyline, spread, total) from The Odds API
for a given date and save them to a CSV file.

Usage:
  python fetch_odds_for_date.py           # uses today's NBA local date (America/New_York)
  python fetch_odds_for_date.py 2025-11-24
"""

import os
import sys
import json
import requests
import datetime as dt
from pathlib import Path

from zoneinfo import ZoneInfo
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# Markets & config
MARKETS = "h2h,spreads,totals"
REGIONS = "us"          # US books
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"
BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,pointsbetus"

# Map The Odds API team names -> your abbreviations
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


def get_api_key() -> str:
    """Read ODDS_API_KEY from environment or .env."""
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY not found in environment. Make sure you either:\n"
            "  - put ODDS_API_KEY=YOUR_KEY in your .env file, OR\n"
            "  - export ODDS_API_KEY=YOUR_KEY in your shell before running."
        )
    return api_key


def parse_target_date(argv) -> dt.date:
    """
    Parse the NBA local date (America/New_York) from CLI,
    or default to today's local date.
    """
    tz = ZoneInfo("America/New_York")
    if len(argv) > 1:
        # Interpret the provided date as local NBA date (no timezone)
        return dt.date.fromisoformat(argv[1])
    # Default: today's date in NBA local timezone
    return dt.datetime.now(tz).date()


def select_bookmaker(bookmakers, preferred_books=None):
    """
    Given the 'bookmakers' list from The Odds API, choose one bookmaker's markets.

    Returns:
      dict: { 'h2h': market_dict_or_None, 'spreads': ..., 'totals': ... }
    """
    if preferred_books is None:
        preferred_books = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]

    if not bookmakers:
        return {"h2h": None, "spreads": None, "totals": None}

    ordered = sorted(
        bookmakers,
        key=lambda b: preferred_books.index(b["key"])
        if b["key"] in preferred_books
        else len(preferred_books),
    )

    best = ordered[0]
    by_key = {m["key"]: m for m in best.get("markets", [])}
    return {
        "h2h": by_key.get("h2h"),
        "spreads": by_key.get("spreads"),
        "totals": by_key.get("totals"),
    }


def extract_row_from_event(event: dict, target_local_date: dt.date, tz: ZoneInfo) -> dict | None:
    """
    Convert a single Odds API event into a row dict for our CSV.
    Filters out events not on target_local_date (NBA local time).
    """
    commence_iso = event.get("commence_time")
    if not commence_iso:
        return None

    # Parse commence_time as UTC
    try:
        # e.g. "2025-11-25T00:10:00Z"
        utc_dt = dt.datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    except Exception:
        return None

    local_dt = utc_dt.astimezone(tz)
    local_date = local_dt.date()

    # Filter by NBA local date
    if local_date != target_local_date:
        return None

    # v4 response: home_team and away_team exist as separate fields
    home_team = event.get("home_team")
    away_team = event.get("away_team")

    # Fallback in case some sport uses 'teams' + 'home_team'
    if (not away_team) and "teams" in event:
        teams = event.get("teams", [])
        if len(teams) == 2:
            if home_team in teams:
                away_team = teams[0] if teams[1] == home_team else teams[1]
            else:
                # If home_team not in teams for some reason, just pick second as away
                away_team = teams[1]

    if not home_team or not away_team:
        # If we can't reliably identify both teams, skip this event
        return None

    # Choose bookmaker and extract markets
    bm_markets = select_bookmaker(event.get("bookmakers", []))

    home_ml = None
    away_ml = None
    home_spread = None
    home_spread_odds = None
    away_spread = None
    away_spread_odds = None
    total_points = None
    over_odds = None
    under_odds = None

    # Moneyline (h2h)
    h2h = bm_markets["h2h"]
    if h2h is not None:
        for outcome in h2h.get("outcomes", []):
            name = outcome.get("name")
            price = outcome.get("price")
            if name == home_team:
                home_ml = price
            elif name == away_team:
                away_ml = price

    # Spreads
    spreads = bm_markets["spreads"]
    if spreads is not None:
        for outcome in spreads.get("outcomes", []):
            name = outcome.get("name")
            point = outcome.get("point")
            price = outcome.get("price")
            if name == home_team:
                home_spread = point
                home_spread_odds = price
            elif name == away_team:
                away_spread = point
                away_spread_odds = price

    # Totals (Over/Under)
    totals = bm_markets["totals"]
    if totals is not None:
        for outcome in totals.get("outcomes", []):
            name = outcome.get("name")
            point = outcome.get("point")
            price = outcome.get("price")
            if name == "Over":
                total_points = point
                over_odds = price
            elif name == "Under":
                total_points = point
                under_odds = price

    row = {
        # This is actually the NBA local date we filtered on
        "date_utc": target_local_date.isoformat(),
        "commence_time": commence_iso,
        "home_team": home_team,
        "away_team": away_team,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "home_spread": home_spread,
        "home_spread_odds": home_spread_odds,
        "away_spread": away_spread,
        "away_spread_odds": away_spread_odds,
        "total_points": total_points,
        "over_odds": over_odds,
        "under_odds": under_odds,
    }

    return row


def main(argv=None):
    if argv is None:
        argv = sys.argv

    tz = ZoneInfo("America/New_York")
    target_local_date = parse_target_date(argv)

    print(
        f"Target local NBA date filter: {target_local_date.isoformat()} "
        f"in America/New_York"
    )

    api_key = get_api_key()

    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
        "bookmakers": BOOKMAKERS,
    }

    print(f"Requesting NBA odds for upcoming games from {ODDS_API_BASE_URL}")
    resp = requests.get(ODDS_API_BASE_URL, params=params, timeout=15)

    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print("Error from The Odds API:")
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
        sys.exit(1)

    data = resp.json()
    print(f"Received {len(data)} events from The Odds API\n")

    # Debug: show all event times in local NBA time
    print("Debug: event commence times in NBA local time:")
    for ev in data:
        commence_iso = ev.get("commence_time")
        if not commence_iso:
            continue
        try:
            utc_dt = dt.datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            local_dt = utc_dt.astimezone(tz)
            local_date = local_dt.date()
            print(
                f"  {commence_iso} -> {local_dt.strftime('%Y-%m-%d %H:%M')} "
                f"(EST) local_date={local_date}"
            )
        except Exception:
            print(f"  {commence_iso} -> (could not parse)")

    rows = []
    for event in data:
        row = extract_row_from_event(event, target_local_date, tz)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nNo games on this local date with odds available (after filtering).")
        return

    df = pd.DataFrame(rows)

    # Map team names -> abbreviations for merging with your model
    df["home_team_abbrev"] = df["home_team"].map(TEAM_NAME_TO_ABBREV)
    df["away_team_abbrev"] = df["away_team"].map(TEAM_NAME_TO_ABBREV)

    missing_home = df[df["home_team_abbrev"].isna()]["home_team"].unique()
    missing_away = df[df["away_team_abbrev"].isna()]["away_team"].unique()
    if len(missing_home) or len(missing_away):
        print("WARNING: Missing abbrevs for some teams:")
        print("  home:", missing_home)
        print("  away:", missing_away)

    cols_order = [
        "date_utc",
        "commence_time",
        "home_team",
        "away_team",
        "home_team_abbrev",
        "away_team_abbrev",
        "home_ml",
        "away_ml",
        "home_spread",
        "home_spread_odds",
        "away_spread",
        "away_spread_odds",
        "total_points",
        "over_odds",
        "under_odds",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    out_dir = Path(__file__).resolve().parent / "odds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"odds_nba_{target_local_date.isoformat()}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} games with odds to {out_path}")

    print("\nSample odds:")
    for _, r in df.head().iterrows():
        print(
            f"{r['away_team']} @ {r['home_team']} | "
            f"ML home={r['home_ml']} away={r['away_ml']} | "
            f"spread home={r['home_spread']} ({r['home_spread_odds']}) | "
            f"total={r['total_points']} O({r['over_odds']}) U({r['under_odds']})"
        )


if __name__ == "__main__":
    main()
