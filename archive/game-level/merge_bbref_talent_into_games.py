#!/usr/bin/env python

"""
merge_bbref_talent_into_games.py

Takes:
  - games_all_2015_2025_features_rolling_last10_prevstats_odds.csv
  - bbref_team_talent_2015_2025.csv

and merges BBRef team talent onto each game row for both home and away teams.

Output:
  - games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv
"""

import pandas as pd
from pathlib import Path

GAMES_CSV = "games_all_2015_2025_features_rolling_last10_prevstats_odds.csv"
TEAM_TALENT_CSV = "bbref_team_talent_2015_2025.csv"
OUTPUT_CSV = "games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv"


def main():
    games_path = Path(GAMES_CSV)
    talent_path = Path(TEAM_TALENT_CSV)

    # ------------------------------------------------
    # 1. Load games
    # ------------------------------------------------
    print(f"Loading games from {games_path} ...")
    games = pd.read_csv(games_path)
    print(f"Games shape: {games.shape}")
    print("Games columns (first 25):")
    print(games.columns.tolist()[:25])

    required_cols = ["season", "home_team", "away_team"]
    missing = [c for c in required_cols if c not in games.columns]
    if missing:
        raise KeyError(f"Games CSV is missing required columns: {missing}")

    # Make sure keys are clean
    games["season"] = games["season"].astype(int)
    games["home_team"] = games["home_team"].astype(str).str.upper()
    games["away_team"] = games["away_team"].astype(str).str.upper()

    # ------------------------------------------------
    # 2. Load team talent
    # ------------------------------------------------
    print(f"\nLoading team talent from {talent_path} ...")
    talent = pd.read_csv(talent_path)
    print(f"Talent shape: {talent.shape}")
    print("Talent columns:", talent.columns.tolist())

    required_talent_cols = [
        "season",
        "team_abbrev",
        "bbref_team_PER",
        "bbref_team_BPM",
        "bbref_team_VORP_sum",
        "total_MP",
    ]
    missing_talent = [c for c in required_talent_cols if c not in talent.columns]
    if missing_talent:
        raise KeyError(f"Team talent CSV is missing required columns: {missing_talent}")

    talent["season"] = talent["season"].astype(int)
    talent["team_abbrev"] = talent["team_abbrev"].astype(str).str.upper()

    # ------------------------------------------------
    # 3. Build home-side talent table and merge
    # ------------------------------------------------
    home_talent = talent.rename(
        columns={
            "team_abbrev": "home_team",
            "bbref_team_PER": "bbref_home_team_PER",
            "bbref_team_BPM": "bbref_home_team_BPM",
            "bbref_team_VORP_sum": "bbref_home_team_VORP_sum",
            "total_MP": "bbref_home_team_total_MP",
        }
    )

    games = games.merge(
        home_talent[
            [
                "season",
                "home_team",
                "bbref_home_team_PER",
                "bbref_home_team_BPM",
                "bbref_home_team_VORP_sum",
                "bbref_home_team_total_MP",
            ]
        ],
        on=["season", "home_team"],
        how="left",
    )

    # ------------------------------------------------
    # 4. Build away-side talent table and merge
    # ------------------------------------------------
    away_talent = talent.rename(
        columns={
            "team_abbrev": "away_team",
            "bbref_team_PER": "bbref_away_team_PER",
            "bbref_team_BPM": "bbref_away_team_BPM",
            "bbref_team_VORP_sum": "bbref_away_team_VORP_sum",
            "total_MP": "bbref_away_team_total_MP",
        }
    )

    games = games.merge(
        away_talent[
            [
                "season",
                "away_team",
                "bbref_away_team_PER",
                "bbref_away_team_BPM",
                "bbref_away_team_VORP_sum",
                "bbref_away_team_total_MP",
            ]
        ],
        on=["season", "away_team"],
        how="left",
    )

    # ------------------------------------------------
    # 5. Sanity checks + save
    # ------------------------------------------------
    print("\nAfter merge, shape:", games.shape)

    missing_home = games["bbref_home_team_PER"].isna().sum()
    missing_away = games["bbref_away_team_PER"].isna().sum()
    print(f"Rows missing home talent: {missing_home}")
    print(f"Rows missing away talent: {missing_away}")

    print("\nSample of talent columns:")
    print(
        games[
            [
                "season",
                "home_team",
                "away_team",
                "bbref_home_team_PER",
                "bbref_away_team_PER",
                "bbref_home_team_BPM",
                "bbref_away_team_BPM",
                "bbref_home_team_VORP_sum",
                "bbref_away_team_VORP_sum",
            ]
        ].head(10)
    )

    out_path = Path(OUTPUT_CSV)
    games.to_csv(out_path, index=False)
    print(f"\nSaved merged file with BBRef team talent to {out_path}")


if __name__ == "__main__":
    main()
