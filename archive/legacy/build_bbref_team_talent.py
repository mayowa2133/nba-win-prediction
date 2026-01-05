#!/usr/bin/env python
"""
build_bbref_team_talent.py

Reads the BBRef player advanced talent table (one row per player+season)
and aggregates it into per-team, per-season talent features.

Input:
    bbref_player_advanced_talent_2015_2025.csv
Output:
    bbref_team_talent_2015_2025.csv

Team abbreviations are mapped from BBRef style (BRK/CHO/PHO, etc.)
to your BallDontLie-style abbreviations (BKN/CHA/PHX, etc.).

We DROP multi-team aggregates like '2TM', '3TM', '4TM' because they
don't correspond to a specific franchise.
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PLAYER_TALENT_PATH = "bbref_player_advanced_talent_2015_2025.csv"
DEFAULT_TEAM_TALENT_PATH = "bbref_team_talent_2015_2025.csv"


BBREF_TO_BDL_TEAM_MAP = {
    # Nets
    "BRK": "BKN",
    "NJN": "BKN",  # older, but harmless to support

    # Hornets
    "CHO": "CHA",

    # Suns
    "PHO": "PHX",

    # Pelicans (older names, in case they show up)
    "NOH": "NOP",
    "NOK": "NOP",
    # Everything else: keep as-is
}

MULTI_TEAM_LABELS = {"2TM", "3TM", "4TM", "5TM", "6TM", "7TM"}


def build_team_talent(player_csv: Path) -> pd.DataFrame:
    print(f"Loading player talent from {player_csv} ...")
    df = pd.read_csv(player_csv)

    required_cols = ["season", "Player", "Team", "MP", "bbref_PER", "bbref_BPM", "bbref_VORP"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Player talent CSV is missing required columns: {missing}")

    # Drop multi-team aggregate labels like '2TM', '3TM', etc.
    print(f"Original rows: {len(df)}")
    df = df[~df["Team"].isin(MULTI_TEAM_LABELS)].copy()
    print(f"After dropping multi-team aggregates {MULTI_TEAM_LABELS}: {len(df)} rows")

    # Map BBRef team codes to BallDontLie codes
    df["team_abbrev"] = df["Team"].map(BBREF_TO_BDL_TEAM_MAP).fillna(df["Team"])

    # Ensure numeric
    for col in ["bbref_PER", "bbref_BPM", "bbref_VORP", "MP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group by season + team and compute MP-weighted averages
    rows = []
    for (season, team), g in df.groupby(["season", "team_abbrev"]):
        total_mp = g["MP"].sum()
        if total_mp <= 0:
            # No minutes, skip this team-season
            continue

        avg_per = (g["bbref_PER"] * g["MP"]).sum() / total_mp
        avg_bpm = (g["bbref_BPM"] * g["MP"]).sum() / total_mp
        total_vorp = g["bbref_VORP"].sum()

        rows.append(
            {
                "season": season,
                "team_abbrev": team,
                "bbref_team_PER": avg_per,
                "bbref_team_BPM": avg_bpm,
                "bbref_team_VORP_sum": total_vorp,
                "total_MP": total_mp,
            }
        )

    team_talent = pd.DataFrame(rows)
    team_counts = team_talent.groupby("season")["team_abbrev"].nunique()

    print("\nPer-season unique team counts:")
    for season, n in sorted(team_counts.items()):
        print(f"  {season}: {n} teams")

    print("\nSample of team talent table:")
    print(team_talent.sort_values(["season", "team_abbrev"]).head(20))

    return team_talent


def main():
    parser = argparse.ArgumentParser(description="Build team-level BBRef talent features.")
    parser.add_argument(
        "--player_csv",
        type=str,
        default=DEFAULT_PLAYER_TALENT_PATH,
        help=f"Input player talent CSV (default: {DEFAULT_PLAYER_TALENT_PATH})",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=DEFAULT_TEAM_TALENT_PATH,
        help=f"Output team talent CSV (default: {DEFAULT_TEAM_TALENT_PATH})",
    )
    args = parser.parse_args()

    player_csv = Path(args.player_csv)
    out_csv = Path(args.out_csv)

    team_talent = build_team_talent(player_csv)
    team_talent.to_csv(out_csv, index=False)
    print(f"\nSaved team talent table to {out_csv}")


if __name__ == "__main__":
    main()
