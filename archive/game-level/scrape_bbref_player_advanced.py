#!/usr/bin/env python
"""
scrape_bbref_player_advanced.py

Scrapes Basketball-Reference advanced boxscore stats for all players
for a range of seasons and builds a unified "talent" table with one
row per (season, player), preferring the TOT row when players changed
teams.

Output: bbref_player_advanced_talent_2015_2025.csv
"""

import time
from typing import List, Optional

import requests
import pandas as pd

BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"

# Seasons we care about (aligned with your games dataset)
SEASONS: List[int] = list(range(2015, 2026))

OUT_CSV = "bbref_player_advanced_talent_2015_2025.csv"


def fetch_season_advanced_table(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch a single season's advanced stats table from BBRef and return as a DataFrame.

    Normalizes:
      - Flattens multi-index headers if present
      - Renames 'Tm' -> 'Team' (if needed)
      - Drops repeated header rows
      - Converts MP to numeric
      - Adds 'season' column
    """
    url = BASE_URL.format(season=season)
    print(f"\n=== Fetching season {season} from {url} ===")

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code} for season {season}")
        return None

    # Parse tables from HTML
    try:
        tables = pd.read_html(resp.text)
    except ValueError as e:
        print(f"  ERROR parsing HTML for season {season}: {e}")
        return None

    if not tables:
        print(f"  No tables found for season {season}")
        return None

    # First table is the advanced stats table
    df = tables[0]

    print(f"  Raw rows: {len(df)} columns: {list(df.columns)}")

    # Flatten multi-index header if necessary
    if isinstance(df.columns, pd.MultiIndex):
        # Usually something like ('Rk', 'Rk'), ('Player', 'Player') etc.
        df.columns = [c[1] if isinstance(c, tuple) and c[1] != "" else c[0] for c in df.columns]

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Some older code expected 'Tm'; current BBRef uses 'Team'
    # Normalize so we always work with 'Team'
    col_rename = {}
    if "Tm" in df.columns and "Team" not in df.columns:
        col_rename["Tm"] = "Team"
    df = df.rename(columns=col_rename)

    # Sanity check for Team column
    if "Team" not in df.columns:
        print(f"  WARNING: season {season} missing 'Team' column. Columns: {list(df.columns)}")

    # Remove repeated header rows (BBRef repeats header in the body)
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]

    # Drop rows with missing Player
    if "Player" in df.columns:
        df = df[df["Player"].notna()]

    # Convert MP to numeric so we can pick the row with most minutes
    if "MP" in df.columns:
        df["MP"] = pd.to_numeric(df["MP"], errors="coerce")

    # Add season column
    df["season"] = season

    return df


def pick_tot_row(group: pd.DataFrame) -> pd.Series:
    """
    For a given (season, Player) group:
      - If there's a 'TOT' row in Team, return that.
      - Else, pick the row with maximum MP.
      - As a fallback, return the first row.
    """
    df = group

    if "Team" in df.columns:
        tot = df[df["Team"] == "TOT"]
        if len(tot) == 1:
            return tot.iloc[0]
        elif len(tot) > 1 and "MP" in df.columns:
            # Rare, but pick the TOT row with highest MP
            idx = tot["MP"].idxmax()
            return tot.loc[idx]

    # No TOT row: pick row with max minutes if available
    if "MP" in df.columns and df["MP"].notna().any():
        idx = df["MP"].idxmax()
        return df.loc[idx]

    # Fallback: just take the first
    return df.iloc[0]


def build_unified_talent_table(seasons: List[int]) -> pd.DataFrame:
    """
    Fetches and concatenates advanced stats for all seasons,
    then deduplicates to one row per (season, Player) using pick_tot_row().
    """
    dfs: List[pd.DataFrame] = []

    for season in seasons:
        df_season = fetch_season_advanced_table(season)
        if df_season is not None and not df_season.empty:
            dfs.append(df_season)
        # Small delay to be polite to BBRef
        time.sleep(1.5)

    if not dfs:
        raise RuntimeError("No data fetched for any season.")

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined raw rows across seasons: {len(all_df)}")

    # Keep only columns we care about
    desired_cols = [
        "season",
        "Player",
        "Team",   # unified team column
        "Pos",
        "Age",
        "G",
        "MP",
        "PER",
        "TS%",
        "WS",
        "WS/48",
        "BPM",
        "VORP",
    ]

    keep_cols = [c for c in desired_cols if c in all_df.columns]
    talent = all_df[keep_cols].copy()

    # Group by season & Player, pick TOT row or max MP row
    grouped = talent.groupby(["season", "Player"], as_index=False, group_keys=False)
    dedup_rows = grouped.apply(pick_tot_row)

    print(f"Final deduped rows (one per season/player): {len(dedup_rows)}")

    # Rename advanced metrics with clearer prefixes so we don't clash with other features
    rename_map = {}
    if "PER" in dedup_rows.columns:
        rename_map["PER"] = "bbref_PER"
    if "WS" in dedup_rows.columns:
        rename_map["WS"] = "bbref_WS"
    if "WS/48" in dedup_rows.columns:
        rename_map["WS/48"] = "bbref_WS_per48"
    if "BPM" in dedup_rows.columns:
        rename_map["BPM"] = "bbref_BPM"
    if "VORP" in dedup_rows.columns:
        rename_map["VORP"] = "bbref_VORP"
    if "TS%" in dedup_rows.columns:
        rename_map["TS%"] = "bbref_TS_pct"

    talent_final = dedup_rows.rename(columns=rename_map)

    return talent_final


def main() -> None:
    print(
        "Building player talent table from BBRef advanced stats for seasons: "
        f"{SEASONS}"
    )
    talent = build_unified_talent_table(SEASONS)

    print("\nSample rows after dedup:")
    print(talent.head(10))

    out_path = OUT_CSV
    talent.to_csv(out_path, index=False)
    print(f"\nSaved BBRef player talent table to {out_path}")


if __name__ == "__main__":
    main()
