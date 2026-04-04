#!/usr/bin/env python
"""
fetch_injury_data.py

Fetch and process injury/availability data for NBA players.

Phase 1: Infer injuries from game logs (DNP - Did Not Play patterns)
Phase 2: Can be extended with external APIs (ESPN, Rotowire, etc.)

Output:
  data/injury_data.csv with columns:
    - player_id
    - player_name
    - game_date
    - season
    - is_injured (1 if DNP when expected, 0 otherwise)
    - injury_status (probable/questionable/out based on patterns)
    - days_since_last_dnp
    - dnp_count_last_10_games
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PLAYER_LOGS_CSV = Path("data/player_game_logs.csv")
OUTPUT_CSV = Path("data/injury_data.csv")


def infer_injuries_from_logs(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Infer injury status from game logs by detecting DNP (Did Not Play) patterns.
    
    Logic:
    - If a player has 0 minutes in a game where they normally play, likely injured
    - If they've played in recent games but suddenly DNP, likely injured
    - Track consecutive DNPs as more severe injury
    """
    df = df_logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    
    # Sort by player and date
    df = df.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    
    # Calculate rolling averages for expected minutes (using last 10 games)
    df["minutes_roll10"] = df.groupby("player_id")["minutes"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    )
    
    # Calculate recent average (last 5 games) for more sensitive detection
    df["minutes_roll5"] = df.groupby("player_id")["minutes"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
    )
    
    # DNP detection: Multiple criteria
    # 1. Zero minutes when expected > 5 (more sensitive)
    # 2. Minutes dropped by >50% from recent average
    # 3. Zero minutes in consecutive games
    df["is_dnp_zero"] = (df["minutes"] == 0) & (df["minutes_roll10"] > 5)
    df["is_dnp_reduced"] = (
        (df["minutes"] < df["minutes_roll5"] * 0.5) & 
        (df["minutes_roll5"] > 15) & 
        (df["minutes"] < 10)
    )
    
    # Combine DNP signals
    df["is_dnp"] = df["is_dnp_zero"] | df["is_dnp_reduced"]
    
    # Injury status based on patterns
    df["injury_status"] = "healthy"
    
    # Count DNPs in last 10 games (calculate before using)
    df["dnp_count_last_10"] = (
        df.groupby("player_id")["is_dnp"]
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=0).sum())
    )
    
    # Track consecutive DNPs
    def calc_dnp_streak(group):
        streak = 0
        streaks = []
        for is_dnp in group:
            if is_dnp:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        return pd.Series(streaks, index=group.index)
    
    df["dnp_streak"] = df.groupby("player_id")["is_dnp"].transform(calc_dnp_streak)
    
    # Classify injury status based on patterns
    # Out: 3+ consecutive DNPs or multiple DNPs in last 10 games
    df.loc[df["dnp_streak"] >= 3, "injury_status"] = "out"
    df.loc[(df["dnp_count_last_10"] >= 3) & (df["is_dnp"]), "injury_status"] = "out"
    
    # Questionable: 1-2 DNPs or significant minutes reduction
    df.loc[(df["dnp_streak"] >= 1) & (df["dnp_streak"] < 3) & (df["injury_status"] == "healthy"), "injury_status"] = "questionable"
    df.loc[df["is_dnp_reduced"] & (df["injury_status"] == "healthy"), "injury_status"] = "questionable"
    
    # Probable: Single DNP but high usage player
    df.loc[df["is_dnp"] & (df["minutes_roll10"] > 20) & (df["injury_status"] == "healthy"), "injury_status"] = "probable"
    
    # Days since last DNP
    def calc_days_since_dnp(group):
        last_dnp_date = None
        days_list = []
        for idx, (date, is_dnp) in enumerate(zip(group["game_date"], group["is_dnp"])):
            if is_dnp:
                last_dnp_date = date
                days_list.append(0)
            elif last_dnp_date is not None:
                days_list.append((date - last_dnp_date).days)
            else:
                days_list.append(999)  # Never had a DNP
        return pd.Series(days_list, index=group.index)
    
    df["days_since_last_dnp"] = (
        df.groupby("player_id")
        .apply(lambda g: calc_days_since_dnp(g))
        .reset_index(level=0, drop=True)
    )
    
    # Binary injury indicator (1 if DNP when expected, 0 otherwise)
    df["is_injured"] = df["is_dnp"].astype(int)
    
    # Select output columns
    output_cols = [
        "player_id",
        "player_name",
        "game_date",
        "season",
        "is_injured",
        "injury_status",
        "days_since_last_dnp",
        "dnp_count_last_10",
    ]
    
    return df[output_cols].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and process injury/availability data for NBA players."
    )
    parser.add_argument(
        "--logs-csv",
        type=str,
        default=str(PLAYER_LOGS_CSV),
        help="Path to player_game_logs.csv (default: data/player_game_logs.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help="Output CSV path (default: data/injury_data.csv)",
    )
    args = parser.parse_args()
    
    logs_path = Path(args.logs_csv)
    output_path = Path(args.output)
    
    if not logs_path.exists():
        raise FileNotFoundError(f"Player logs file not found: {logs_path}")
    
    print(f"Loading player logs from {logs_path} ...")
    df_logs = pd.read_csv(logs_path)
    print(f"Loaded {len(df_logs):,} game log rows.")
    
    print("\nInferring injury status from game logs...")
    df_injuries = infer_injuries_from_logs(df_logs)
    
    # Summary stats
    injured_count = df_injuries["is_injured"].sum()
    total_games = len(df_injuries)
    print(f"\nInjury detection summary:")
    print(f"  Total games: {total_games:,}")
    print(f"  Detected injuries (DNPs): {injured_count:,} ({100*injured_count/total_games:.2f}%)")
    print(f"  Injury status breakdown:")
    status_counts = df_injuries["injury_status"].value_counts()
    for status, count in status_counts.items():
        print(f"    {status}: {count:,} ({100*count/total_games:.2f}%)")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_injuries.to_csv(output_path, index=False)
    print(f"\nSaved injury data to {output_path}")
    print(f"  Rows: {len(df_injuries):,}")
    print(f"  Columns: {list(df_injuries.columns)}")


if __name__ == "__main__":
    main()

