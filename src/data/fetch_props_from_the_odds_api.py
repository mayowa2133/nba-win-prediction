#!/usr/bin/env python
"""
fetch_props_from_the_odds_api.py

Pulls NBA player prop odds from The Odds API and writes a CSV "slate"-style file
that you can feed into scan_slate.py (or inspect manually).

Core behavior:
  1. Fetch today's NBA games from The Odds API.
  2. For each game, query player prop markets (e.g. player_points).
  3. Flatten props into rows like:
       player,line,side,odds,book,...
  4. Save to a CSV (default: data/odds_slate.csv).
  5. NEW: Also fetches game-level lines (spreads, totals) and saves to
     data/game_lines.csv for use as features in the points model.

Assumptions about The Odds API v4:
  - Base URL: https://api.the-odds-api.com/v4
  - List odds for a sport:
        /sports/{sport_key}/odds
    with params: apiKey, regions, markets, oddsFormat, dateFormat...
  - Get event-specific odds (needed for props):
        /sports/{sport_key}/events/{event_id}/odds
    with params: apiKey, regions, markets, oddsFormat, dateFormat...

  - For player props, each "market" will look something like:
        {
          "key": "player_points",
          "outcomes": [
            {
              "name": "LeBron James",
              "description": "Over",
              "price": -115,
              "point": 27.5
            },
            {
              "name": "LeBron James",
              "description": "Under",
              "price": -105,
              "point": 27.5
            }
          ]
        }

    Where:
      - name         -> player name
      - description  -> "Over" or "Under"
      - price        -> American odds (if oddsFormat=american)
      - point        -> line (e.g. 27.5)

If The Odds API changes their schema, you may need to tweak the parsing logic
in `parse_props_from_event_odds`.

Usage examples:

  export ODDS_API_KEY="your_key_here"

  # Basic: today's NBA, player_points props from US books, all books
  python fetch_props_from_the_odds_api.py

  # Only DraftKings & FanDuel, save to a custom file
  python fetch_props_from_the_odds_api.py \
      --bookmakers draftkings,fanduel \
      --output data/odds_slate_dk_fd.csv

  # Multiple markets (points + assists):
  python fetch_props_from_the_odds_api.py \
      --markets player_points,player_assists

Then you can run:

  python scan_slate.py --input data/odds_slate.csv --min-edge 3.0
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import csv
import sys
from datetime import datetime
import json


BASE_URL = "https://api.the-odds-api.com/v4"

DEFAULT_SPORT_KEY = "basketball_nba"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "player_points"  # comma-separated in CLI
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"


def get_api_key(cli_key: Optional[str]) -> str:
    """
    Get The Odds API key from CLI or environment.
    """
    key = cli_key or os.getenv("ODDS_API_KEY")
    if not key:
        print(
            "ERROR: No API key provided. Set ODDS_API_KEY env var or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def request_json(endpoint: str, params: Dict[str, Any]) -> Any:
    """
    Helper to make a GET request to The Odds API and return parsed JSON.
    Raises on non-200 or JSON parse issues.
    """
    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        # Try to parse structured error response so callers can handle quota gracefully
        payload = None
        try:
            payload = resp.json()
        except Exception:
            payload = {"message": resp.text}

        err_code = str(payload.get("error_code") or "").upper()
        msg = payload.get("message") or resp.text
        raise RuntimeError(
            json.dumps(
                {
                    "url": url,
                    "status": resp.status_code,
                    "error_code": err_code,
                    "message": msg,
                }
            )
        )

    try:
        data = resp.json()
    except Exception as e:
        print(f"ERROR: Failed to parse JSON from {url}: {e}", file=sys.stderr)
        print(resp.text[:1000], file=sys.stderr)
        sys.exit(1)

    # Log remaining quota if headers exist
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None and used is not None:
        print(f"  [API usage] used={used}, remaining={remaining}")

    return data


def fetch_events(
    api_key: str,
    sport_key: str,
    regions: str,
) -> List[Dict[str, Any]]:
    """
    Fetch today's events for the given sport.
    We call the general /odds endpoint with a cheap market (e.g. h2h) just to get events.

    Endpoint: /sports/{sport_key}/odds
    """
    print(f"Fetching events for sport={sport_key}, regions={regions} ...")

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",  # minimal, we just want the event list
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }

    endpoint = f"/sports/{sport_key}/odds"
    data = request_json(endpoint, params)

    if not isinstance(data, list):
        print("ERROR: Expected a list of events from The Odds API.", file=sys.stderr)
        print(repr(data)[:1000])
        sys.exit(1)

    print(f"  -> Got {len(data)} events.")
    return data


def fetch_game_lines(
    api_key: str,
    sport_key: str,
    regions: str,
) -> List[Dict[str, Any]]:
    """
    Fetch game-level odds (spreads, totals) for today's games.
    
    Returns a list of events with spreads and totals markets included.
    
    Endpoint: /sports/{sport_key}/odds
    """
    print(f"Fetching game lines (spreads, totals) for sport={sport_key} ...")
    
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "spreads,totals",
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }
    
    endpoint = f"/sports/{sport_key}/odds"
    data = request_json(endpoint, params)
    
    if not isinstance(data, list):
        print("ERROR: Expected a list of events from The Odds API.", file=sys.stderr)
        return []
    
    print(f"  -> Got game lines for {len(data)} events.")
    return data


def parse_game_lines(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse game-level lines (spreads and totals) from events data.
    
    For each game, extracts:
      - vegas_game_total: the Over/Under total points line
      - vegas_home_spread: the home team spread (negative = favored)
      - vegas_away_spread: the away team spread
    
    Returns one row per game with consensus (median) lines across books.
    """
    import statistics
    
    game_rows: List[Dict[str, Any]] = []
    
    for event in events:
        event_id = event.get("id")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        
        if not all([event_id, home_team, away_team]):
            continue
        
        # Collect all totals and spreads across bookmakers
        totals_lines: List[float] = []
        home_spreads: List[float] = []
        away_spreads: List[float] = []
        
        bookmakers = event.get("bookmakers", [])
        for bk in bookmakers:
            markets = bk.get("markets", [])
            for market in markets:
                market_key = market.get("key")
                outcomes = market.get("outcomes", [])
                
                if market_key == "totals":
                    for out in outcomes:
                        point = out.get("point")
                        if point is not None:
                            try:
                                totals_lines.append(float(point))
                            except (ValueError, TypeError):
                                pass
                
                elif market_key == "spreads":
                    for out in outcomes:
                        name = out.get("name", "")
                        point = out.get("point")
                        if point is not None:
                            try:
                                spread_val = float(point)
                                if name == home_team:
                                    home_spreads.append(spread_val)
                                elif name == away_team:
                                    away_spreads.append(spread_val)
                            except (ValueError, TypeError):
                                pass
        
        # Compute consensus (median) values
        vegas_game_total = None
        vegas_home_spread = None
        vegas_away_spread = None
        vegas_abs_spread = None
        
        if totals_lines:
            vegas_game_total = statistics.median(totals_lines)
        
        if home_spreads:
            vegas_home_spread = statistics.median(home_spreads)
            vegas_abs_spread = abs(vegas_home_spread)
        
        if away_spreads:
            vegas_away_spread = statistics.median(away_spreads)
            if vegas_abs_spread is None:
                vegas_abs_spread = abs(vegas_away_spread)
        
        # Parse game date from commence_time
        game_date = None
        if commence_time:
            try:
                # Parse ISO format and convert to date
                dt_obj = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                game_date = dt_obj.strftime("%Y-%m-%d")
            except Exception:
                pass
        
        row = {
            "event_id": event_id,
            "game_date": game_date,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "vegas_game_total": vegas_game_total,
            "vegas_home_spread": vegas_home_spread,
            "vegas_away_spread": vegas_away_spread,
            "vegas_abs_spread": vegas_abs_spread,
        }
        game_rows.append(row)
    
    return game_rows


def write_game_lines_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write game lines to CSV.
    """
    if not rows:
        print("No game lines to write.", file=sys.stderr)
        return
    
    fieldnames = [
        "event_id",
        "game_date", 
        "commence_time",
        "home_team",
        "away_team",
        "vegas_game_total",
        "vegas_home_spread",
        "vegas_away_spread",
        "vegas_abs_spread",
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"Wrote {len(rows)} game lines to {output_path}")


def fetch_event_props(
    api_key: str,
    sport_key: str,
    event_id: str,
    regions: str,
    markets: List[str],
    bookmakers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fetch player prop odds for a single event.

    Endpoint: /sports/{sport_key}/events/{event_id}/odds

    Returns the full JSON for that event (including bookmakers & markets).
    """
    endpoint = f"/sports/{sport_key}/events/{event_id}/odds"

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    print(f"  Fetching props for event_id={event_id} ...")
    data = request_json(endpoint, params)

    # /events/{id}/odds returns a single event object (not a list)
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    elif isinstance(data, dict):
        return data
    else:
        print(
            "WARN: Unexpected response shape for event odds; "
            "expected dict or single-element list.",
            file=sys.stderr,
        )
        return data


def parse_props_from_event_odds(
    event_odds: Dict[str, Any],
    wanted_markets: List[str],
) -> List[Dict[str, Any]]:
    """
    Flatten the props from a single event odds JSON into rows.

    Assumes The Odds API structure like:

      {
        "id": "...",
        "sport_key": "basketball_nba",
        "commence_time": "2025-11-28T01:30:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Dallas Mavericks",
        "bookmakers": [
          {
            "key": "draftkings",
            "title": "DraftKings",
            "markets": [
              {
                "key": "player_points",
                "outcomes": [
                  {
                    "name": "LeBron James",
                    "description": "Over",
                    "price": -115,
                    "point": 27.5
                  },
                  ...
                ]
              },
              ...
            ]
          },
          ...
        ]
      }

    We return a list of rows with fields:
      - player       (str)
      - line         (float)
      - side         ('over'|'under')
      - odds         (int)      # American odds
      - book         (str)      # e.g. DraftKings
      - sport_key
      - event_id
      - commence_time
      - home_team
      - away_team
      - market_key
      - book_key
      - book_title
    """
    rows: List[Dict[str, Any]] = []

    event_id = event_odds.get("id")
    sport_key = event_odds.get("sport_key")
    commence_time = event_odds.get("commence_time")
    home_team = event_odds.get("home_team")
    away_team = event_odds.get("away_team")

    bookmakers = event_odds.get("bookmakers", [])
    if not bookmakers:
        return rows

    for bk in bookmakers:
        bk_key = bk.get("key")
        bk_title = bk.get("title") or bk_key
        markets = bk.get("markets", [])
        for m in markets:
            m_key = m.get("key")
            if m_key not in wanted_markets:
                continue

            outcomes = m.get("outcomes", [])
            for out in outcomes:
                # This is based on The Odds API docs assumption.
                player_name = out.get("name") or ""
                side = (out.get("description") or "").lower().strip()
                odds = out.get("price")
                line = out.get("point")

                # Basic sanity checks
                if not player_name:
                    continue
                if side not in ("over", "under"):
                    # In case the schema is reversed (name=Over, description=player)
                    # try to recover:
                    maybe_side = (out.get("name") or "").lower().strip()
                    maybe_player = out.get("description") or ""
                    if maybe_side in ("over", "under") and maybe_player:
                        side = maybe_side
                        player_name = maybe_player
                        odds = out.get("price")
                        line = out.get("point")
                    else:
                        continue

                if odds is None or line is None:
                    continue

                try:
                    odds_int = int(odds)
                except Exception:
                    # If odds are decimal, you'd convert; but we requested american.
                    # If not convertible, skip.
                    continue

                try:
                    line_float = float(line)
                except Exception:
                    continue

                row = {
                    "player": player_name,
                    "line": line_float,
                    "side": side,
                    "odds": odds_int,
                    "book": bk_title,
                    "sport_key": sport_key,
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_key": m_key,
                    "book_key": bk_key,
                    "book_title": bk_title,
                }
                rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write rows to CSV. Ensures at least the columns your scan_slate.py expects:
      player,line,side,odds,book
    plus any extra fields we collected.
    """
    # Always create the output file so downstream steps can decide how to handle it.
    # This matters when API quota is exhausted (we want a valid, empty slate file).
    default_fieldnames = [
        "player",
        "line",
        "side",
        "odds",
        "book",
        "sport_key",
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
        "market_key",
        "book_key",
        "book_title",
    ]

    # Collect all keys across rows so we don't lose extra info
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    else:
        fieldnames = default_fieldnames

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NBA player prop odds from The Odds API and write a slate CSV."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The Odds API key. If omitted, uses ODDS_API_KEY env var.",
    )
    parser.add_argument(
        "--sport-key",
        type=str,
        default=DEFAULT_SPORT_KEY,
        help=f"Sport key (default: {DEFAULT_SPORT_KEY})",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=DEFAULT_REGIONS,
        help=f"Regions to pull odds from (default: {DEFAULT_REGIONS}). "
             "Example: us,uk,eu",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=DEFAULT_MARKETS,
        help="Comma-separated list of markets (default: player_points). "
             "Examples: player_points,player_assists,player_rebounds",
    )
    parser.add_argument(
        "--bookmakers",
        type=str,
        default=None,
        help="Optional comma-separated list of bookmakers to include "
             "(e.g. draftkings,fanduel). If omitted, includes all.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/odds_slate.csv",
        help="Output CSV path (default: data/odds_slate.csv)",
    )
    parser.add_argument(
        "--game-lines-output",
        type=str,
        default="data/game_lines.csv",
        help="Output CSV path for game-level lines (default: data/game_lines.csv)",
    )
    parser.add_argument(
        "--skip-game-lines",
        action="store_true",
        help="Skip fetching game-level lines (spreads, totals).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = get_api_key(args.api_key)

    sport_key = args.sport_key
    regions = args.regions
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bookmakers = (
        [b.strip() for b in args.bookmakers.split(",") if b.strip()]
        if args.bookmakers
        else None
    )
    output_path = Path(args.output)
    game_lines_path = Path(args.game_lines_output)

    print("=== Fetching props from The Odds API ===")
    print(f"Sport:      {sport_key}")
    print(f"Regions:    {regions}")
    print(f"Markets:    {markets}")
    print(f"Bookmakers: {bookmakers if bookmakers else '(all available)'}")
    print(f"Output:     {output_path}")
    print(f"Game lines: {game_lines_path}")
    print("========================================\n")

    # 1) Get today's events (games)
    try:
        events = fetch_events(api_key=api_key, sport_key=sport_key, regions=regions)
    except Exception as e:
        msg = str(e)
        if "OUT_OF_USAGE_CREDITS" in msg:
            print(
                "[WARN] Odds API quota exhausted while fetching events. "
                f"Writing empty slate to {output_path} and exiting cleanly.",
                file=sys.stderr,
            )
            write_csv([], output_path)
            sys.exit(0)
        raise

    if not events:
        print("No events returned. Is it off-season or did you hit a limit?", file=sys.stderr)
        write_csv([], output_path)
        sys.exit(0)

    # 1b) Fetch game-level lines (spreads, totals) if not skipped
    if not args.skip_game_lines:
        try:
            game_lines_events = fetch_game_lines(
                api_key=api_key,
                sport_key=sport_key,
                regions=regions,
            )
            game_lines_rows = parse_game_lines(game_lines_events)
            if game_lines_rows:
                write_game_lines_csv(game_lines_rows, game_lines_path)
            else:
                print("No game lines were parsed.", file=sys.stderr)
        except Exception as e:
            msg = str(e)
            if "OUT_OF_USAGE_CREDITS" in msg:
                print(
                    "[WARN] Odds API quota exhausted while fetching game lines. "
                    "Continuing without updating game_lines.csv.",
                    file=sys.stderr,
                )
            else:
                print(f"[WARN] Failed to fetch game lines: {e}", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []

    # 2) For each event, get prop odds and parse them
    for ev in events:
        event_id = ev.get("id")
        if not event_id:
            continue

        try:
            ev_props = fetch_event_props(
                api_key=api_key,
                sport_key=sport_key,
                event_id=event_id,
                regions=regions,
                markets=markets,
                bookmakers=bookmakers,
            )

            rows = parse_props_from_event_odds(ev_props, wanted_markets=markets)
            if rows:
                print(f"  -> Parsed {len(rows)} prop rows for event_id={event_id}")
                all_rows.extend(rows)
        except Exception as e:
            # If we hit usage quota mid-run, keep partial data and exit cleanly.
            msg = str(e)
            if "OUT_OF_USAGE_CREDITS" in msg:
                print(
                    f"[WARN] Odds API quota exhausted while fetching event_id={event_id}. "
                    f"Writing partial slate with {len(all_rows)} rows and exiting cleanly.",
                    file=sys.stderr,
                )
                break
            print(f"[WARN] Failed to fetch props for event_id={event_id}: {e}", file=sys.stderr)
            continue

    if not all_rows:
        print(
            "[WARN] No prop rows were parsed (possibly due to API limits).",
            file=sys.stderr,
        )
        sys.exit(0)

    # 3) Write output CSV
    write_csv(all_rows, output_path)


if __name__ == "__main__":
    main()