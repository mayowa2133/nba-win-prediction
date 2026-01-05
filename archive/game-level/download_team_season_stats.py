import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

START_SEASON = 2015
END_SEASON = 2025  # inclusive


def season_int_to_nba_str(season_int: int) -> str:
    next_year_two = str(season_int + 1)[-2:]
    return f"{season_int}-{next_year_two}"


def fetch_team_advanced_stats_for_season(season_int: int) -> pd.DataFrame:
    season_str = season_int_to_nba_str(season_int)
    print(f"Fetching team advanced stats for NBA season {season_str}...")

    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season_str,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )

    df = stats.get_data_frames()[0]

    # IMPORTANT: include TEAM_ABBREVIATION here
    cols_to_keep = [
        "TEAM_ID",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "GP",
        "W",
        "L",
        "W_PCT",
        "MIN",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "AST_PCT",
        "AST_RATIO",
        "OREB_PCT",
        "DREB_PCT",
        "REB_PCT",
        "TM_TOV_PCT",
        "EFG_PCT",
        "TS_PCT",
        "PACE",
        "PIE",
    ]

    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep].copy()

    df.insert(0, "SEASON_INT", season_int)
    return df


if __name__ == "__main__":
    all_seasons = []

    for season in range(START_SEASON, END_SEASON + 1):
        df_season = fetch_team_advanced_stats_for_season(season)
        out_path = f"team_stats_{season}.csv"
        df_season.to_csv(out_path, index=False)
        print(f"Saved {len(df_season)} teams to {out_path}")
        all_seasons.append(df_season)

    all_stats = pd.concat(all_seasons, ignore_index=True)
    all_stats.to_csv(f"team_stats_{START_SEASON}_{END_SEASON}.csv", index=False)
    print(f"Saved combined team stats to team_stats_{START_SEASON}_{END_SEASON}.csv with {len(all_stats)} rows.")
