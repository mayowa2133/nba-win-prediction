import pandas as pd

GAMES_CSV = "games_all_2015_2025_features_basic.csv"
TEAM_STATS_CSV = "team_stats_2015_2025.csv"
OUTPUT_CSV = "games_all_2015_2025_features_teamstats.csv"

BASE_TEAM_COLS = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "PACE",
    "EFG_PCT",
    "TS_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "W_PCT",
    "PIE",
]

# Mapping from NBA.com team names -> abbreviations used in your games CSV
NAME_TO_ABBREV = {
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
    "LA Clippers": "LAC",
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


def main():
    games = pd.read_csv(GAMES_CSV)
    team_stats = pd.read_csv(TEAM_STATS_CSV)

    # Ensure we have TEAM_ABBREVIATION; if not, derive from TEAM_NAME
    if "TEAM_ABBREVIATION" not in team_stats.columns:
        team_stats["TEAM_ABBREVIATION"] = team_stats["TEAM_NAME"].map(NAME_TO_ABBREV)
        if team_stats["TEAM_ABBREVIATION"].isna().any():
            missing = team_stats[team_stats["TEAM_ABBREVIATION"].isna()][
                "TEAM_NAME"
            ].unique()
            print("Missing mappings for team names:", missing)
            raise SystemExit(
                "Some team names have no abbreviation mapping. Add them to NAME_TO_ABBREV."
            )

    # Normalize season column name
    team_stats = team_stats.rename(columns={"SEASON_INT": "season"})

    # Keep only relevant columns
    cols_available = ["season", "TEAM_ABBREVIATION"] + [
        c for c in BASE_TEAM_COLS if c in team_stats.columns
    ]
    team_stats = team_stats[cols_available].copy()

    # ----- Home merge -----
    home_stats = team_stats.rename(
        columns={
            "TEAM_ABBREVIATION": "home_team",
            **{c: f"home_{c.lower()}" for c in BASE_TEAM_COLS if c in team_stats.columns},
        }
    )

    games = games.merge(
        home_stats,
        how="left",
        on=["season", "home_team"],
        validate="m:1",
    )

    # ----- Away merge -----
    away_stats = team_stats.rename(
        columns={
            "TEAM_ABBREVIATION": "away_team",
            **{c: f"away_{c.lower()}" for c in BASE_TEAM_COLS if c in team_stats.columns},
        }
    )

    games = games.merge(
        away_stats,
        how="left",
        on=["season", "away_team"],
        validate="m:1",
    )

    # ----- Diff features: home - away -----
    diff_cols = []
    for col in BASE_TEAM_COLS:
        h_col = f"home_{col.lower()}"
        a_col = f"away_{col.lower()}"
        if h_col in games.columns and a_col in games.columns:
            diff_name = f"{col.lower()}_diff"
            games[diff_name] = games[h_col] - games[a_col]
            diff_cols.append(diff_name)

    print("Created diff feature columns:", diff_cols)

    games.to_csv(OUTPUT_CSV, index=False)
    print(
        f"Saved enriched dataset with team stats to {OUTPUT_CSV} with {len(games)} games."
    )


if __name__ == "__main__":
    main()
