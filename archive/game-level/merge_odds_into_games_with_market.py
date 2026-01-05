# merge_odds_into_games_with_market.py

import math
from pathlib import Path

import pandas as pd


GAMES_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats.csv")
ODDS_PATH = Path("oddsData.csv")
OUT_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")


def american_to_prob(odds_val: float) -> float:
    """
    Convert American moneyline odds to implied win probability (with vig).
    Returns NaN if odds_val is NaN.
    """
    if pd.isna(odds_val):
        return math.nan
    # positive odds: +150 -> 100 / (150 + 100)
    if odds_val > 0:
        return 100.0 / (odds_val + 100.0)
    # negative odds: -200 -> 200 / (200 + 100)
    return -odds_val / (-odds_val + 100.0)


def main():
    print(f"Loading games from {GAMES_PATH}...")
    games = pd.read_csv(GAMES_PATH)

    print(f"Loading odds from {ODDS_PATH}...")
    odds = pd.read_csv(ODDS_PATH)

    # Only keep rows where the listed team is at HOME (home/visitor == 'vs')
    odds_home = odds[odds["home/visitor"] == "vs"].copy()

    # Map Kaggle team names -> BallDontLie 3-letter codes
    team_map = {
        "Atlanta": "ATL",
        "Boston": "BOS",
        "Brooklyn": "BKN",
        "Charlotte": "CHA",
        "Chicago": "CHI",
        "Cleveland": "CLE",
        "Dallas": "DAL",
        "Denver": "DEN",
        "Detroit": "DET",
        "Golden State": "GSW",
        "Houston": "HOU",
        "Indiana": "IND",
        "LA Clippers": "LAC",
        "LA Lakers": "LAL",
        "Memphis": "MEM",
        "Miami": "MIA",
        "Milwaukee": "MIL",
        "Minnesota": "MIN",
        "New Jersey": "BKN",   # old Nets → BKN
        "New Orleans": "NOP",
        "New York": "NYK",
        "Oklahoma City": "OKC",
        "Orlando": "ORL",
        "Philadelphia": "PHI",
        "Phoenix": "PHX",
        "Portland": "POR",
        "Sacramento": "SAC",
        "San Antonio": "SAS",
        "Seattle": "OKC",      # Sonics → OKC
        "Toronto": "TOR",
        "Utah": "UTA",
        "Washington": "WAS",
    }

    # Attach BDL-style team codes
    odds_home["home_team"] = odds_home["team"].map(team_map)
    odds_home["away_team"] = odds_home["opponent"].map(team_map)

    # Safety check
    if odds_home["home_team"].isna().any() or odds_home["away_team"].isna().any():
        missing = odds_home[odds_home["home_team"].isna() | odds_home["away_team"].isna()]
        print("WARNING: Some teams were not mapped correctly:")
        print(missing[["date", "team", "opponent"]].head())
        print("Fix team_map before continuing.")
        return

    # Normalize dates to datetime
    odds_home["date_dt"] = pd.to_datetime(odds_home["date"])
    games["date_dt"] = pd.to_datetime(games["date"])

    # Keep only odds columns we care about
    odds_join = odds_home[
        [
            "date_dt",
            "home_team",
            "away_team",
            "moneyLine",
            "opponentMoneyLine",
            "total",
            "spread",
        ]
    ].copy()

    print("Merging odds into games (left join on date + home_team + away_team)...")
    merged = games.merge(
        odds_join,
        on=["date_dt", "home_team", "away_team"],
        how="left",
        validate="m:1",  # each game should match at most one odds row
    )

    # Compute implied probabilities from American odds
    merged["home_ml"] = merged["moneyLine"]
    merged["away_ml"] = merged["opponentMoneyLine"]

    merged["home_prob_raw"] = merged["home_ml"].apply(american_to_prob)
    merged["away_prob_raw"] = merged["away_ml"].apply(american_to_prob)

    # Remove the bookmaker vig to get "fair" market probabilities
    overround = merged["home_prob_raw"] + merged["away_prob_raw"]
    merged["market_home_prob"] = merged["home_prob_raw"] / overround
    merged["market_away_prob"] = merged["away_prob_raw"] / overround

    # Just rename these for clarity
    merged["market_spread"] = merged["spread"]
    merged["market_total"] = merged["total"]

    # Quick coverage stats
    coverage = (
        merged.groupby("season")["market_home_prob"].apply(lambda s: s.notna().mean())
    )
    print("\nCoverage (fraction of games with odds) by season:")
    for season, frac in coverage.sort_index().items():
        print(f"  {season}: {frac:.3f}")

    # Save
    print(f"\nSaving merged dataset with odds to {OUT_PATH} ...")
    merged.to_csv(OUT_PATH, index=False)
    print(f"Done. Rows: {len(merged)}")


if __name__ == "__main__":
    main()
