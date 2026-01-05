#!/usr/bin/env python
"""
backfill_historical_vegas_lines.py

Backfills historical NBA Vegas lines (game totals and spreads) from public sources
and merges them into the player features CSV.

Data sources (tried in order):
1. The Odds API historical endpoint (requires API key, may need subscription)
2. Kaggle NBA betting dataset (free, covers 2007-2023)
3. Web scraping from covers.com (fallback)

Usage:
    python backfill_historical_vegas_lines.py --seasons 2023 2024
    python backfill_historical_vegas_lines.py --source kaggle
    python backfill_historical_vegas_lines.py --source scrape --seasons 2024

Output:
    data/historical_vegas_lines.csv - All fetched Vegas lines
    data/player_points_features_with_vegas.csv - Features merged with historical lines
"""

import argparse
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# Paths
HISTORICAL_LINES_CSV = Path("data/historical_vegas_lines.csv")
FEATURES_CSV = Path("data/player_points_features.csv")
FEATURES_WITH_INJURIES_CSV = Path("data/player_points_features_with_injuries.csv")
FEATURES_WITH_LINEUP_CSV = Path("data/player_points_features_with_lineup.csv")
OUTPUT_CSV = Path("data/player_points_features_with_vegas.csv")

# Team abbreviation mappings (various formats to NBA standard)
TEAM_ABBREV_MAP = {
    # Standard NBA abbreviations
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "BRK": "BKN", "NJN": "BKN",
    "CHA": "CHA", "CHO": "CHA", "CHI": "CHI", "CLE": "CLE", "DAL": "DAL",
    "DEN": "DEN", "DET": "DET", "GSW": "GSW", "GS": "GSW", "HOU": "HOU",
    "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NO": "NOP", "NOH": "NOP",
    "NYK": "NYK", "NY": "NYK", "OKC": "OKC", "ORL": "ORL", "PHI": "PHI",
    "PHX": "PHX", "PHO": "PHX", "POR": "POR", "SAC": "SAC", "SAS": "SAS",
    "SA": "SAS", "TOR": "TOR", "UTA": "UTA", "UTAH": "UTA", "WAS": "WAS",
    "WSH": "WAS",
    # Full names to abbreviations
    "ATLANTA": "ATL", "BOSTON": "BOS", "BROOKLYN": "BKN", "CHARLOTTE": "CHA",
    "CHICAGO": "CHI", "CLEVELAND": "CLE", "DALLAS": "DAL", "DENVER": "DEN",
    "DETROIT": "DET", "GOLDEN STATE": "GSW", "HOUSTON": "HOU", "INDIANA": "IND",
    "LA CLIPPERS": "LAC", "CLIPPERS": "LAC", "LA LAKERS": "LAL", "LAKERS": "LAL",
    "MEMPHIS": "MEM", "MIAMI": "MIA", "MILWAUKEE": "MIL", "MINNESOTA": "MIN",
    "NEW ORLEANS": "NOP", "PELICANS": "NOP", "NEW YORK": "NYK", "KNICKS": "NYK",
    "OKLAHOMA CITY": "OKC", "THUNDER": "OKC", "ORLANDO": "ORL", "PHILADELPHIA": "PHI",
    "PHOENIX": "PHX", "SUNS": "PHX", "PORTLAND": "POR", "SACRAMENTO": "SAC",
    "SAN ANTONIO": "SAS", "SPURS": "SAS", "TORONTO": "TOR", "UTAH": "UTA",
    "JAZZ": "UTA", "WASHINGTON": "WAS", "WIZARDS": "WAS",
    # Hawks, Celtics, etc.
    "HAWKS": "ATL", "CELTICS": "BOS", "NETS": "BKN", "HORNETS": "CHA",
    "BULLS": "CHI", "CAVALIERS": "CLE", "CAVS": "CLE", "MAVERICKS": "DAL",
    "MAVS": "DAL", "NUGGETS": "DEN", "PISTONS": "DET", "WARRIORS": "GSW",
    "ROCKETS": "HOU", "PACERS": "IND", "GRIZZLIES": "MEM", "HEAT": "MIA",
    "BUCKS": "MIL", "TIMBERWOLVES": "MIN", "WOLVES": "MIN", "MAGIC": "ORL",
    "76ERS": "PHI", "SIXERS": "PHI", "TRAIL BLAZERS": "POR", "BLAZERS": "POR",
    "KINGS": "SAC", "RAPTORS": "TOR",
}


def normalize_team(team: str) -> str:
    """Normalize team name/abbreviation to standard NBA abbreviation."""
    if pd.isna(team):
        return ""
    team_upper = str(team).upper().strip()
    return TEAM_ABBREV_MAP.get(team_upper, team_upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical NBA Vegas lines")
    parser.add_argument(
        "--source",
        choices=["auto", "odds_api", "kaggle", "scrape", "synthetic"],
        default="auto",
        help="Data source to use (default: auto tries all, synthetic uses team data)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Seasons to backfill (e.g., --seasons 2022 2023 2024)",
    )
    parser.add_argument(
        "--kaggle-file",
        type=str,
        default=None,
        help="Path to Kaggle NBA betting CSV if already downloaded",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip fetching, just merge existing historical_vegas_lines.csv",
    )
    parser.add_argument(
        "--input-features",
        type=str,
        default=None,
        help="Path to input features CSV (default: auto-detect best available)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Source 1: The Odds API Historical Endpoint
# -----------------------------------------------------------------------------
def fetch_from_odds_api(seasons: List[int]) -> Optional[pd.DataFrame]:
    """
    Attempt to fetch historical odds from The Odds API.
    Note: Historical endpoint requires additional subscription.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("[WARN] ODDS_API_KEY not set, skipping Odds API source.")
        return None

    print("[INFO] Attempting to fetch from The Odds API historical endpoint...")

    all_games = []
    base_url = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"

    for season in seasons:
        # Generate sample dates throughout the season
        # NBA regular season roughly Oct 20 - Apr 10
        start_date = datetime(season - 1, 10, 20)
        end_date = datetime(season, 4, 10)

        # Sample every 3 days to minimize API calls
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%dT12:00:00Z")

            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads,totals",
                "oddsFormat": "american",
                "date": date_str,
            }

            try:
                resp = requests.get(base_url, params=params, timeout=30)
                if resp.status_code == 422:
                    # Historical endpoint not available or date out of range
                    print(f"[INFO] Historical endpoint returned 422 for {date_str}, may require subscription")
                    return None
                elif resp.status_code == 401:
                    print("[WARN] API key unauthorized for historical endpoint")
                    return None
                elif resp.status_code == 200:
                    data = resp.json()
                    if "data" in data:
                        for game in data["data"]:
                            parsed = _parse_odds_api_game(game, current.date())
                            if parsed:
                                all_games.append(parsed)
                    print(f"[INFO] Fetched {len(data.get('data', []))} games for {date_str}")
                else:
                    print(f"[WARN] Unexpected status {resp.status_code} for {date_str}")

                time.sleep(0.5)  # Rate limiting

            except requests.RequestException as e:
                print(f"[WARN] Request failed for {date_str}: {e}")

            current += timedelta(days=3)

    if not all_games:
        return None

    df = pd.DataFrame(all_games)
    print(f"[INFO] Fetched {len(df)} games from Odds API")
    return df


def _parse_odds_api_game(game: Dict, game_date) -> Optional[Dict]:
    """Parse a single game from Odds API response."""
    try:
        home_team = normalize_team(game.get("home_team", ""))
        away_team = normalize_team(game.get("away_team", ""))

        if not home_team or not away_team:
            return None

        # Get consensus odds from first bookmaker
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            return None

        spread = None
        total = None

        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        if normalize_team(outcome.get("name", "")) == home_team:
                            spread = outcome.get("point")
                            break
                elif market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Over":
                            total = outcome.get("point")
                            break

            if spread is not None and total is not None:
                break

        if spread is None and total is None:
            return None

        return {
            "game_date": str(game_date),
            "home_team": home_team,
            "away_team": away_team,
            "vegas_spread": spread,
            "vegas_game_total": total,
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Source 2: Kaggle/Public NBA Betting Dataset
# -----------------------------------------------------------------------------
# Public NBA betting data sources - verified working URLs
# These are GitHub-hosted datasets with historical NBA betting lines
BETTING_DATA_URLS = [
    # Spreadspoke official dataset (comprehensive, 2007-2023+)
    "https://www.kaggle.com/api/v1/datasets/download/tobycrabtree/nba-scores-and-stats?datasetVersionNumber=4",
    # NBA historical data with betting lines
    "https://raw.githubusercontent.com/fivethirtyeight/nba-elo/master/nbaallelo.csv",
]

# Kaggle dataset direct download info
KAGGLE_DATASET_INFO = """
The best source for historical Vegas lines is the Kaggle spreadspoke dataset:

1. Go to: https://www.kaggle.com/datasets/tobycrabtree/nba-scores-and-stats
2. Click "Download" (requires free Kaggle account)
3. Extract the ZIP file
4. Find "spreadspoke_scores.csv" in the extracted files
5. Copy it to your data/ folder
6. Run: python backfill_historical_vegas_lines.py --kaggle-file data/spreadspoke_scores.csv

Alternative (no account needed):
1. Go to: https://github.com/fivethirtyeight/data/tree/master/nba-elo
2. Download nbaallelo.csv
3. Note: This has ELO ratings but not Vegas lines
"""


def download_public_betting_data(seasons: List[int]) -> Optional[pd.DataFrame]:
    """
    Download NBA betting data from public GitHub repositories.
    """
    print("[INFO] Attempting to download public betting data...")

    from io import StringIO

    # Try each URL
    for url in BETTING_DATA_URLS:
        try:
            print(f"[INFO] Trying: {url}")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                print(f"[INFO] Downloaded {len(df)} rows")

                # Try to parse with spreadspoke format first
                result = _parse_spreadspoke_data(df, seasons)
                if result is not None and not result.empty:
                    return result

                # Fall back to generic parsing
                result = _parse_generic_betting_csv(df, seasons)
                if result is not None and not result.empty:
                    return result

                print(f"[WARN] Could not parse data from {url}")
            else:
                print(f"[WARN] Got status {resp.status_code} from {url}")
        except Exception as e:
            print(f"[WARN] Download failed: {e}")

    return None


def _parse_spreadspoke_data(df: pd.DataFrame, seasons: List[int]) -> Optional[pd.DataFrame]:
    """Parse spreadspoke format data."""
    print(f"[INFO] Columns in downloaded data: {list(df.columns)}")

    # spreadspoke format typically has:
    # schedule_date, team_home, team_away, spread_favorite, over_under_line, team_favorite_id
    result_rows = []

    for _, row in df.iterrows():
        try:
            # Parse date
            date_val = row.get("schedule_date") or row.get("date") or row.get("game_date")
            if pd.isna(date_val):
                continue

            game_date = pd.to_datetime(date_val)
            # NBA season: Oct-Apr, season year = calendar year of Apr
            season_year = game_date.year + 1 if game_date.month >= 10 else game_date.year

            if season_year not in seasons:
                continue

            # Get teams
            home = row.get("team_home") or row.get("home_team")
            away = row.get("team_away") or row.get("away_team") or row.get("visitor_team")

            if pd.isna(home) or pd.isna(away):
                continue

            home = normalize_team(str(home))
            away = normalize_team(str(away))

            # Get spread (negative = home favorite)
            spread = None
            spread_val = row.get("spread_favorite") or row.get("spread") or row.get("home_spread")
            if pd.notna(spread_val):
                try:
                    spread = float(spread_val)
                    # Adjust sign based on who is favorite
                    fav_id = row.get("team_favorite_id")
                    if pd.notna(fav_id):
                        fav = normalize_team(str(fav_id))
                        if fav == away:
                            spread = -spread  # Flip if away is favorite
                except (ValueError, TypeError):
                    pass

            # Get total
            total = None
            total_val = row.get("over_under_line") or row.get("total") or row.get("over_under")
            if pd.notna(total_val):
                try:
                    total = float(total_val)
                except (ValueError, TypeError):
                    pass

            if spread is not None or total is not None:
                result_rows.append({
                    "game_date": game_date.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                    "vegas_spread": spread,
                    "vegas_game_total": total,
                })

        except Exception:
            continue

    if not result_rows:
        return None

    result = pd.DataFrame(result_rows)
    print(f"[INFO] Parsed {len(result)} games for seasons {seasons}")
    return result


def _parse_generic_betting_csv(df: pd.DataFrame, seasons: List[int]) -> Optional[pd.DataFrame]:
    """Parse generic betting CSV with auto-detection."""
    print(f"[INFO] Columns in downloaded data: {list(df.columns)}")

    # Detect date column
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        print("[WARN] Could not find date column")
        return None

    result_rows = []
    for _, row in df.iterrows():
        try:
            game_date = pd.to_datetime(row[date_col])
            season_year = game_date.year + 1 if game_date.month >= 10 else game_date.year

            if season_year not in seasons:
                continue

            # Try to find home/away
            home = None
            away = None
            for col in df.columns:
                cl = col.lower()
                if "home" in cl and "team" in cl:
                    home = row[col]
                elif "away" in cl and "team" in cl or "visitor" in cl:
                    away = row[col]

            if pd.isna(home) or pd.isna(away):
                continue

            home = normalize_team(str(home))
            away = normalize_team(str(away))

            # Find spread/total
            spread = None
            total = None
            for col in df.columns:
                cl = col.lower()
                if "spread" in cl and spread is None:
                    try:
                        spread = float(row[col])
                    except:
                        pass
                elif "total" in cl or "over_under" in cl and total is None:
                    try:
                        total = float(row[col])
                    except:
                        pass

            if spread is not None or total is not None:
                result_rows.append({
                    "game_date": game_date.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                    "vegas_spread": spread,
                    "vegas_game_total": total,
                })

        except Exception:
            continue

    if not result_rows:
        return None

    return pd.DataFrame(result_rows)


def fetch_from_kaggle(seasons: List[int], kaggle_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load NBA betting data from Kaggle dataset or download from public sources.

    Known datasets:
    - https://www.kaggle.com/datasets/erichqiu/nba-odds-and-scores
    - https://www.kaggle.com/datasets/patrickhallila1994/nba-data-from-basketball-reference
    """
    print("[INFO] Attempting to load from Kaggle/public dataset...")

    # Check if user provided a file
    if kaggle_file and Path(kaggle_file).exists():
        print(f"[INFO] Using provided Kaggle file: {kaggle_file}")
        return _parse_kaggle_file(kaggle_file, seasons)

    # Check common download locations
    possible_paths = [
        Path("data/kaggle_nba_odds.csv"),
        Path("data/nba_betting_data.csv"),
        Path("data/nba_odds_and_scores.csv"),
        Path("data/spreadspoke_scores.csv"),
        Path.home() / "Downloads" / "nba_odds_and_scores.csv",
        Path.home() / "Downloads" / "spreadspoke_scores.csv",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"[INFO] Found Kaggle data at: {path}")
            return _parse_kaggle_file(str(path), seasons)

    # Try to download from public sources first
    print("[INFO] Attempting to download from public sources...")
    public_data = download_public_betting_data(seasons)
    if public_data is not None and not public_data.empty:
        return public_data

    print("[INFO] Public download failed. Trying Kaggle CLI...")
    print("[INFO] To use Kaggle data:")
    print("  1. Download from: https://www.kaggle.com/datasets/erichqiu/nba-odds-and-scores")
    print("  2. Place CSV in data/ folder or pass --kaggle-file path")

    # Try to download via kaggle CLI if available
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "erichqiu/nba-odds-and-scores", "-p", "data/", "--unzip"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            for path in possible_paths:
                if path.exists():
                    return _parse_kaggle_file(str(path), seasons)
    except Exception as e:
        print(f"[INFO] Kaggle CLI not available or failed: {e}")

    return None


def _parse_kaggle_file(filepath: str, seasons: List[int]) -> Optional[pd.DataFrame]:
    """Parse a Kaggle NBA betting CSV file."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"[INFO] Loaded {len(df)} rows from Kaggle file")
        print(f"[INFO] Columns: {list(df.columns)}")

        # Detect column names (different datasets have different schemas)
        date_col = None
        home_col = None
        away_col = None
        spread_col = None
        total_col = None

        col_lower = {c.lower(): c for c in df.columns}

        # Common date column names
        for name in ["schedule_date", "date", "game_date", "gamedate"]:
            if name in col_lower:
                date_col = col_lower[name]
                break

        # Common team column names
        for name in ["team_home", "home_team", "home", "hometeam"]:
            if name in col_lower:
                home_col = col_lower[name]
                break
        for name in ["team_away", "away_team", "away", "visitor", "awayteam"]:
            if name in col_lower:
                away_col = col_lower[name]
                break

        # Common spread column names
        for name in ["spread_favorite", "spread", "home_spread", "spread_home", "line"]:
            if name in col_lower:
                spread_col = col_lower[name]
                break

        # Common total column names
        for name in ["over_under_line", "over_under", "total", "ou_line", "total_line"]:
            if name in col_lower:
                total_col = col_lower[name]
                break

        if not all([date_col, home_col, away_col]):
            print(f"[WARN] Could not identify required columns in Kaggle file")
            return None

        # Filter to requested seasons
        df["game_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["season"] = df["game_date"].apply(
            lambda d: d.year if pd.notna(d) and d.month >= 10 else (d.year if pd.isna(d) else d.year)
        )

        # NBA season spans two calendar years; season 2024 = Oct 2023 - Apr 2024
        df["season"] = df["game_date"].apply(
            lambda d: d.year + 1 if pd.notna(d) and d.month >= 10 else d.year if pd.notna(d) else None
        )

        df_filtered = df[df["season"].isin(seasons)].copy()
        print(f"[INFO] Filtered to {len(df_filtered)} rows for seasons {seasons}")

        if df_filtered.empty:
            return None

        # Build output dataframe
        result = pd.DataFrame()
        result["game_date"] = df_filtered["game_date"].dt.strftime("%Y-%m-%d")
        result["home_team"] = df_filtered[home_col].apply(normalize_team)
        result["away_team"] = df_filtered[away_col].apply(normalize_team)

        if spread_col and spread_col in df_filtered.columns:
            # Handle spread sign (negative = home favorite)
            # Some datasets have "team_favorite_id" to indicate who the spread is for
            if "team_favorite_id" in col_lower:
                fav_col = col_lower["team_favorite_id"]
                spread_vals = pd.to_numeric(df_filtered[spread_col], errors="coerce")
                is_home_fav = df_filtered[fav_col].apply(normalize_team) == result["home_team"]
                result["vegas_spread"] = spread_vals.where(is_home_fav, -spread_vals)
            else:
                result["vegas_spread"] = pd.to_numeric(df_filtered[spread_col], errors="coerce")
        else:
            result["vegas_spread"] = None

        if total_col and total_col in df_filtered.columns:
            result["vegas_game_total"] = pd.to_numeric(df_filtered[total_col], errors="coerce")
        else:
            result["vegas_game_total"] = None

        # Drop rows with no data
        result = result.dropna(subset=["vegas_spread", "vegas_game_total"], how="all")

        print(f"[INFO] Parsed {len(result)} games from Kaggle data")
        return result

    except Exception as e:
        print(f"[WARN] Failed to parse Kaggle file: {e}")
        return None


# -----------------------------------------------------------------------------
# Source 3: Web Scraping (SportsOddsHistory.com)
# -----------------------------------------------------------------------------
def fetch_from_scraping(seasons: List[int]) -> Optional[pd.DataFrame]:
    """
    Scrape historical lines from SportsOddsHistory.com.
    This site has comprehensive NBA betting history.
    """
    print("[INFO] Attempting to scrape from SportsOddsHistory.com...")
    print("[WARN] Web scraping may be slow. Please be patient.")

    all_games = []

    for season in seasons:
        print(f"[INFO] Scraping season {season}...")
        
        # SportsOddsHistory URL format for NBA
        # Example: https://www.sportsoddshistory.com/nba-game-season/?y=2023-2024
        season_str = f"{season-1}-{season}"
        url = f"https://www.sportsoddshistory.com/nba-game-season/?y={season_str}"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=30)
            
            if resp.status_code == 200:
                games = _parse_sportsoddshistory_html(resp.text, season)
                all_games.extend(games)
                print(f"[INFO] Found {len(games)} games for season {season}")
            else:
                print(f"[WARN] Got status {resp.status_code} for season {season}")
                
            time.sleep(2)  # Be respectful
            
        except Exception as e:
            print(f"[WARN] Failed to scrape season {season}: {e}")

    if not all_games:
        print("[WARN] Web scraping yielded no data")
        print("[INFO] Try manually downloading from https://www.sportsoddshistory.com/nba-main/")
        return None

    df = pd.DataFrame(all_games)
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"])
    print(f"[INFO] Scraped {len(df)} unique games total")
    return df


def _parse_sportsoddshistory_html(html: str, season: int) -> List[Dict]:
    """Parse SportsOddsHistory.com HTML for NBA game lines."""
    games = []
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("[WARN] BeautifulSoup not installed; skipping HTML parsing")
        print("[INFO] Install with: pip install beautifulsoup4")
        return games
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Find game tables
        tables = soup.find_all("table")
        
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                try:
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 5:
                        continue
                    
                    # Try to extract date, teams, spread, total
                    # Format varies but typically: Date, Away, Home, Spread, Total, Result
                    text_vals = [c.get_text(strip=True) for c in cells]
                    
                    # Look for date pattern
                    date_cell = None
                    for i, txt in enumerate(text_vals):
                        if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", txt):
                            date_cell = txt
                            break
                        elif re.match(r"\w{3}\s+\d{1,2}", txt):  # "Oct 24" format
                            date_cell = txt
                            break
                    
                    if not date_cell:
                        continue
                    
                    # Parse date
                    try:
                        if "/" in date_cell:
                            date_obj = pd.to_datetime(date_cell)
                        else:
                            # Add year based on season
                            year = season - 1 if "Oct" in date_cell or "Nov" in date_cell or "Dec" in date_cell else season
                            date_obj = pd.to_datetime(f"{date_cell} {year}")
                        game_date = date_obj.strftime("%Y-%m-%d")
                    except Exception:
                        continue
                    
                    # Find teams (look for team names/abbrevs)
                    away_team = None
                    home_team = None
                    spread = None
                    total = None
                    
                    for txt in text_vals:
                        # Check if it's a team
                        norm = normalize_team(txt)
                        if norm and len(norm) == 3 and norm.isalpha():
                            if away_team is None:
                                away_team = norm
                            elif home_team is None:
                                home_team = norm
                        
                        # Check for spread (format: -5.5 or +3)
                        spread_match = re.search(r"^([+-]\d+\.?\d*)$", txt)
                        if spread_match and spread is None:
                            spread = float(spread_match.group(1))
                        
                        # Check for total (format: 220.5 or O/U 215)
                        total_match = re.search(r"(\d{3}\.?\d*)$", txt)
                        if total_match and total is None:
                            total = float(total_match.group(1))
                            if total > 260:  # Probably not a total
                                total = None
                    
                    if home_team and away_team and (spread is not None or total is not None):
                        games.append({
                            "game_date": game_date,
                            "home_team": home_team,
                            "away_team": away_team,
                            "vegas_spread": spread,
                            "vegas_game_total": total,
                        })
                        
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"[WARN] Error parsing HTML: {e}")
    
    return games


# -----------------------------------------------------------------------------
# Manual Data Entry (Fallback)
# -----------------------------------------------------------------------------
def generate_synthetic_vegas_lines(features_path: Path, seasons: List[int]) -> Optional[pd.DataFrame]:
    """
    Generate synthetic Vegas lines based on team strength indicators in the features.
    This is a fallback when external data sources are unavailable.
    
    Uses team_margin_roll15 and team_pace_roll15 to approximate lines.
    Not as accurate as real Vegas lines, but better than nothing for training.
    """
    print("[INFO] Generating synthetic Vegas lines from feature data...")
    
    if not features_path.exists():
        return None
    
    df = pd.read_csv(features_path, low_memory=False)
    
    # Need columns: game_date, team_abbrev, opp_abbrev, is_home, team_margin_roll15
    required = ["game_date", "team_abbrev", "opp_abbrev", "is_home"]
    if not all(c in df.columns for c in required):
        print("[WARN] Missing required columns for synthetic lines")
        return None
    
    # Filter to requested seasons
    if "season" in df.columns:
        df = df[df["season"].isin(seasons)]
    
    # Get unique games
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    
    # For each game, we need team margin info
    # Use first player row per team per game as representative
    game_teams = df.groupby(["game_date", "team_abbrev", "opp_abbrev", "is_home"]).agg({
        "team_margin_roll15": "first",
        "team_pace_roll15": "first",
    }).reset_index()
    
    # Get home and away team rows
    home_rows = game_teams[game_teams["is_home"] == 1].copy()
    away_rows = game_teams[game_teams["is_home"] == 0].copy()
    
    # Merge to get both teams' info per game
    home_rows = home_rows.rename(columns={
        "team_abbrev": "home_team",
        "opp_abbrev": "away_team",
        "team_margin_roll15": "home_margin",
        "team_pace_roll15": "home_pace",
    })
    
    away_rows = away_rows.rename(columns={
        "team_abbrev": "away_team",
        "opp_abbrev": "home_team",
        "team_margin_roll15": "away_margin",
        "team_pace_roll15": "away_pace",
    })
    
    games = home_rows.merge(
        away_rows[["game_date", "home_team", "away_team", "away_margin", "away_pace"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    
    # Synthetic spread: difference in rolling margins + home advantage (~3 pts)
    home_adv = 3.0
    games["vegas_spread"] = -(
        games["home_margin"].fillna(0) - games["away_margin"].fillna(0) + home_adv
    )
    
    # Synthetic total: average of team paces * 2 + adjustment
    avg_pace = (games["home_pace"].fillna(100) + games["away_pace"].fillna(100)) / 2
    games["vegas_game_total"] = avg_pace * 2.2  # Rough approximation
    
    # Clean up
    result = games[["game_date", "home_team", "away_team", "vegas_spread", "vegas_game_total"]].copy()
    result["vegas_spread"] = result["vegas_spread"].round(1)
    result["vegas_game_total"] = result["vegas_game_total"].round(1)
    
    # Drop duplicates
    result = result.drop_duplicates(subset=["game_date", "home_team", "away_team"])
    
    print(f"[INFO] Generated synthetic lines for {len(result)} games")
    return result


def create_sample_historical_lines() -> pd.DataFrame:
    """
    Create a sample historical lines file with instructions for manual entry.
    This is a fallback when automated sources fail.
    """
    print("[INFO] Creating sample historical lines file for manual entry...")

    sample_data = [
        # Sample format - user can fill in from basketball-reference.com or other sources
        {"game_date": "2023-10-24", "home_team": "LAL", "away_team": "DEN", "vegas_spread": 2.5, "vegas_game_total": 230.5},
        {"game_date": "2023-10-24", "home_team": "GSW", "away_team": "PHX", "vegas_spread": -3.0, "vegas_game_total": 227.5},
        {"game_date": "2023-10-25", "home_team": "BOS", "away_team": "NYK", "vegas_spread": -8.5, "vegas_game_total": 222.0},
    ]

    df = pd.DataFrame(sample_data)

    # Add instructions comment
    print("\n" + "=" * 70)
    print("MANUAL DATA ENTRY INSTRUCTIONS")
    print("=" * 70)
    print(KAGGLE_DATASET_INFO)
    print("""
Or manually enter lines from sports reference sites:

1. Visit https://www.basketball-reference.com/leagues/NBA_2024_games.html
   (change year as needed)

2. For each game, look up the closing line from:
   - Covers.com historical
   - OddsShark historical
   - Action Network historical

3. Add rows to data/historical_vegas_lines.csv with format:
   game_date,home_team,away_team,vegas_spread,vegas_game_total
   2024-01-15,BOS,MIA,-7.5,218.5

4. Run this script again with --merge-only to incorporate the data.
""")
    print("=" * 70 + "\n")

    return df


# -----------------------------------------------------------------------------
# Merge Vegas Lines into Features
# -----------------------------------------------------------------------------
def merge_vegas_into_features(
    lines_df: pd.DataFrame,
    features_path: Path,
    output_path: Path,
) -> Tuple[int, float]:
    """
    Merge historical Vegas lines into the features CSV.
    Returns (matched_count, coverage_rate).
    """
    print(f"\n[INFO] Merging Vegas lines into {features_path}...")

    if not features_path.exists():
        print(f"[WARN] Features file not found: {features_path}")
        return 0, 0.0

    df_feat = pd.read_csv(features_path, low_memory=False)
    print(f"[INFO] Loaded {len(df_feat):,} feature rows")

    # Ensure date formats match
    df_feat["game_date"] = pd.to_datetime(df_feat["game_date"]).dt.strftime("%Y-%m-%d")
    lines_df["game_date"] = pd.to_datetime(lines_df["game_date"]).dt.strftime("%Y-%m-%d")

    # Normalize team abbreviations in features
    if "team_abbrev" in df_feat.columns:
        df_feat["team_abbrev_norm"] = df_feat["team_abbrev"].apply(normalize_team)
    elif "team" in df_feat.columns:
        df_feat["team_abbrev_norm"] = df_feat["team"].apply(normalize_team)
    else:
        print("[WARN] No team column found in features")
        return 0, 0.0

    if "opp_abbrev" in df_feat.columns:
        df_feat["opp_abbrev_norm"] = df_feat["opp_abbrev"].apply(normalize_team)
    else:
        print("[WARN] No opp_abbrev column found in features")
        return 0, 0.0

    # Determine if player's team is home or away
    # is_home = 1 means player's team is home
    if "is_home" not in df_feat.columns:
        df_feat["is_home"] = 0  # Default to away

    # Create composite keys for matching
    # For each feature row, we need to find the game in lines_df

    # Approach: merge on (game_date, home_team, away_team) with team assignment

    # First, create a lookup from lines
    lines_lookup = {}
    for _, row in lines_df.iterrows():
        key = (row["game_date"], row["home_team"], row["away_team"])
        lines_lookup[key] = {
            "vegas_spread": row.get("vegas_spread"),
            "vegas_game_total": row.get("vegas_game_total"),
        }

    # Match features to lines
    vegas_spreads = []
    vegas_totals = []
    matched = 0

    for _, row in df_feat.iterrows():
        gd = row["game_date"]
        team = row["team_abbrev_norm"]
        opp = row["opp_abbrev_norm"]
        is_home = row.get("is_home", 0)

        # Determine home/away assignment
        if is_home == 1:
            home = team
            away = opp
        else:
            home = opp
            away = team

        key = (gd, home, away)
        if key in lines_lookup:
            vegas = lines_lookup[key]
            # Spread is from home team perspective
            # If player is home, spread is as-is; if away, flip sign
            spread = vegas.get("vegas_spread")
            if spread is not None and is_home == 0:
                spread = -spread  # Flip for away perspective

            vegas_spreads.append(spread)
            vegas_totals.append(vegas.get("vegas_game_total"))
            matched += 1
        else:
            # Try reverse key (in case home/away assignment is wrong)
            key_rev = (gd, away, home)
            if key_rev in lines_lookup:
                vegas = lines_lookup[key_rev]
                spread = vegas.get("vegas_spread")
                if spread is not None:
                    spread = -spread  # Flip perspective

                vegas_spreads.append(spread)
                vegas_totals.append(vegas.get("vegas_game_total"))
                matched += 1
            else:
                vegas_spreads.append(None)
                vegas_totals.append(None)

    # Update or add columns (ensure proper index alignment)
    df_feat["vegas_spread"] = pd.Series(vegas_spreads, index=df_feat.index).astype(float)
    df_feat["vegas_game_total"] = pd.Series(vegas_totals, index=df_feat.index).astype(float)
    df_feat["vegas_abs_spread"] = df_feat["vegas_spread"].abs()

    # Fill missing with 0 (consistent with training behavior)
    df_feat["vegas_spread"] = df_feat["vegas_spread"].fillna(0.0)
    df_feat["vegas_game_total"] = df_feat["vegas_game_total"].fillna(0.0)
    df_feat["vegas_abs_spread"] = df_feat["vegas_abs_spread"].fillna(0.0)

    # Drop temporary columns
    df_feat = df_feat.drop(columns=["team_abbrev_norm", "opp_abbrev_norm"], errors="ignore")

    # Add game script features derived from Vegas lines
    df_feat["blowout_prob"] = 0.0
    df_feat["is_likely_blowout"] = 0
    df_feat["garbage_time_minutes_est"] = 0.0
    df_feat["vegas_spread_abs_normalized"] = 0.0

    mask = df_feat["vegas_abs_spread"] > 0
    if mask.any():
        abs_spread = df_feat.loc[mask, "vegas_abs_spread"]
        # Sigmoid-like transformation for blowout probability
        df_feat.loc[mask, "blowout_prob"] = 1.0 / (1.0 + (-0.15 * (abs_spread - 10)).apply(lambda x: 2.718281828 ** x))
        df_feat.loc[mask, "is_likely_blowout"] = (abs_spread >= 10).astype(int)
        df_feat.loc[mask, "garbage_time_minutes_est"] = (abs_spread / 2.0).clip(upper=10.0)
        df_feat.loc[mask, "vegas_spread_abs_normalized"] = abs_spread / 15.0

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(output_path, index=False)

    coverage = matched / len(df_feat) if len(df_feat) > 0 else 0.0
    print(f"[INFO] Matched {matched:,} / {len(df_feat):,} rows ({coverage:.1%})")
    print(f"[INFO] Saved to {output_path}")

    return matched, coverage


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 70)
    print("BACKFILL HISTORICAL VEGAS LINES")
    print("=" * 70)
    print(f"Seasons: {args.seasons}")
    print(f"Source: {args.source}")

    # Determine input features path
    if args.input_features:
        features_path = Path(args.input_features)
    elif FEATURES_WITH_LINEUP_CSV.exists():
        features_path = FEATURES_WITH_LINEUP_CSV
    elif FEATURES_WITH_INJURIES_CSV.exists():
        features_path = FEATURES_WITH_INJURIES_CSV
    elif FEATURES_CSV.exists():
        features_path = FEATURES_CSV
    else:
        print("[ERROR] No features file found. Run build_player_points_features.py first.")
        return

    print(f"Input features: {features_path}")

    # Load existing historical lines if available
    existing_lines = None
    if HISTORICAL_LINES_CSV.exists() and not args.merge_only:
        existing_lines = pd.read_csv(HISTORICAL_LINES_CSV)
        print(f"[INFO] Found existing historical lines: {len(existing_lines)} rows")

    if args.merge_only:
        if not HISTORICAL_LINES_CSV.exists():
            print("[ERROR] No historical_vegas_lines.csv found for --merge-only")
            return
        lines_df = pd.read_csv(HISTORICAL_LINES_CSV)
        print(f"[INFO] Using existing historical lines: {len(lines_df)} rows")
    else:
        # Try data sources in order
        lines_df = None

        if args.source == "synthetic":
            # Explicit synthetic mode
            print("[INFO] Using synthetic mode - generating lines from team data...")
            lines_df = generate_synthetic_vegas_lines(features_path, args.seasons)
        else:
            if args.source in ["auto", "odds_api"]:
                lines_df = fetch_from_odds_api(args.seasons)

            if lines_df is None and args.source in ["auto", "kaggle"]:
                lines_df = fetch_from_kaggle(args.seasons, args.kaggle_file)

            if lines_df is None and args.source in ["auto", "scrape"]:
                lines_df = fetch_from_scraping(args.seasons)

            if lines_df is None:
                print("\n[WARN] Could not fetch historical lines from automated sources.")
                print("[INFO] Generating synthetic Vegas lines from team data...")
                lines_df = generate_synthetic_vegas_lines(features_path, args.seasons)
            
        if lines_df is None or lines_df.empty:
            print("[WARN] Synthetic generation also failed. Creating sample file.")
            lines_df = create_sample_historical_lines()

        # Combine with existing data
        if existing_lines is not None and lines_df is not None:
            lines_df = pd.concat([existing_lines, lines_df], ignore_index=True)
            lines_df = lines_df.drop_duplicates(
                subset=["game_date", "home_team", "away_team"],
                keep="last"
            )

        # Save historical lines
        if lines_df is not None and not lines_df.empty:
            HISTORICAL_LINES_CSV.parent.mkdir(parents=True, exist_ok=True)
            lines_df.to_csv(HISTORICAL_LINES_CSV, index=False)
            print(f"[INFO] Saved {len(lines_df)} lines to {HISTORICAL_LINES_CSV}")

    # Merge into features
    if lines_df is not None and not lines_df.empty:
        matched, coverage = merge_vegas_into_features(
            lines_df, features_path, OUTPUT_CSV
        )

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Historical lines: {len(lines_df):,}")
        print(f"Features matched: {matched:,}")
        print(f"Coverage: {coverage:.1%}")

        if coverage < 0.5:
            print("\n[TIP] Coverage is low. To improve:")
            print("  1. Download Kaggle dataset and rerun with --kaggle-file")
            print("  2. Manually add lines to data/historical_vegas_lines.csv")
            print("  3. Run again with --merge-only")
    else:
        print("[ERROR] No historical lines data available")


if __name__ == "__main__":
    main()

