# build_prev_season_teamstats_features.py

import pandas as pd

GAMES_ROLLING_CSV = "games_all_2015_2025_features_rolling_last10.csv"
TEAM_STATS_CSV = "team_stats_2015_2025.csv"
OUT_CSV = "games_all_2015_2025_features_rolling_last10_prevstats.csv"


def main():
    print(f"Loading games from {GAMES_ROLLING_CSV}...")
    games = pd.read_csv(GAMES_ROLLING_CSV)

    print(f"Loading team stats from {TEAM_STATS_CSV}...")
    team_stats = pd.read_csv(TEAM_STATS_CSV)

    # Map NBA.com TEAM_NAME -> BallDontLie abbreviations
    name_to_abbrev = {
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

    team_stats["TEAM_ABBREVIATION"] = team_stats["TEAM_NAME"].map(name_to_abbrev)
    if team_stats["TEAM_ABBREVIATION"].isna().any():
        missing = team_stats[team_stats["TEAM_ABBREVIATION"].isna()]["TEAM_NAME"].unique()
        raise RuntimeError(f"Missing abbreviation mapping for: {missing}")

    # Keep only the columns we care about and rename SEASON_INT -> season
    ts = team_stats[
        [
            "SEASON_INT",
            "TEAM_ABBREVIATION",
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "EFG_PCT",
            "TS_PCT",
            "PACE",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "TM_TOV_PCT",
            "W_PCT",
            "PIE",
        ]
    ].copy()
    ts = ts.rename(columns={"SEASON_INT": "season"})

    # For each game in season S, we want team stats from season S-1
    games["prev_season"] = games["season"] - 1

    # ---- Merge home team prev-season stats ----
    home_prev = ts.add_prefix("home_prev_")
    merged = games.merge(
        home_prev,
        left_on=["prev_season", "home_team"],
        right_on=["home_prev_season", "home_prev_TEAM_ABBREVIATION"],
        how="left",
    )

    # ---- Merge away team prev-season stats ----
    away_prev = ts.add_prefix("away_prev_")
    merged = merged.merge(
        away_prev,
        left_on=["prev_season", "away_team"],
        right_on=["away_prev_season", "away_prev_TEAM_ABBREVIATION"],
        how="left",
    )

    # ---- Build diff features (home - away) ----
    merged["prev_off_rating_diff"] = (
        merged["home_prev_OFF_RATING"] - merged["away_prev_OFF_RATING"]
    )
    merged["prev_def_rating_diff"] = (
        merged["home_prev_DEF_RATING"] - merged["away_prev_DEF_RATING"]
    )
    merged["prev_net_rating_diff"] = (
        merged["home_prev_NET_RATING"] - merged["away_prev_NET_RATING"]
    )
    merged["prev_efg_pct_diff"] = merged["home_prev_EFG_PCT"] - merged["away_prev_EFG_PCT"]
    merged["prev_ts_pct_diff"] = merged["home_prev_TS_PCT"] - merged["away_prev_TS_PCT"]
    merged["prev_pace_diff"] = merged["home_prev_PACE"] - merged["away_prev_PACE"]
    merged["prev_oreb_pct_diff"] = (
        merged["home_prev_OREB_PCT"] - merged["away_prev_OREB_PCT"]
    )
    merged["prev_dreb_pct_diff"] = (
        merged["home_prev_DREB_PCT"] - merged["away_prev_DREB_PCT"]
    )
    merged["prev_reb_pct_diff"] = (
        merged["home_prev_REB_PCT"] - merged["away_prev_REB_PCT"]
    )
    merged["prev_tm_tov_pct_diff"] = (
        merged["home_prev_TM_TOV_PCT"] - merged["away_prev_TM_TOV_PCT"]
    )
    merged["prev_w_pct_diff"] = merged["home_prev_W_PCT"] - merged["away_prev_W_PCT"]
    merged["prev_pie_diff"] = merged["home_prev_PIE"] - merged["away_prev_PIE"]

    # We don't have prev-season stats for 2015 (because team_stats start at 2015),
    # so those rows will have NaNs in these new columns.
    prev_diff_cols = [c for c in merged.columns if c.startswith("prev_") and c.endswith("_diff")]
    before = len(merged)
    merged_clean = merged.dropna(subset=prev_diff_cols)
    after = len(merged_clean)
    print(f"Dropped {before - after} games with no prev-season stats (mostly 2015).")
    print(
        "Remaining games by season:\n",
        merged_clean["season"].value_counts().sort_index(),
    )

    merged_clean.to_csv(OUT_CSV, index=False)
    print(
        f"Saved enriched dataset with prev-season team stats to {OUT_CSV} "
        f"with {len(merged_clean)} games."
    )


if __name__ == "__main__":
    main()
