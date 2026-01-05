import pandas as pd
import math

INPUT_CSV = "games_all_2015_2025_with_elo.csv"
OUTPUT_CSV = "games_all_2015_2025_features_basic.csv"

def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    # ensure sorted by date
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    last_play_date = {}  # team -> last date

    home_rest_days = []
    away_rest_days = []
    home_b2b = []
    away_b2b = []

    for _, row in df.iterrows():
        d = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # home team rest
        if home in last_play_date:
            rest_home = (d - last_play_date[home]).days
        else:
            rest_home = None

        # away team rest
        if away in last_play_date:
            rest_away = (d - last_play_date[away]).days
        else:
            rest_away = None

        # back-to-back flags (played yesterday)
        home_b2b_flag = 1 if rest_home == 1 else 0
        away_b2b_flag = 1 if rest_away == 1 else 0

        home_rest_days.append(rest_home)
        away_rest_days.append(rest_away)
        home_b2b.append(home_b2b_flag)
        away_b2b.append(away_b2b_flag)

        # update last play dates
        last_play_date[home] = d
        last_play_date[away] = d

    df["home_rest_days"] = home_rest_days
    df["away_rest_days"] = away_rest_days
    df["home_b2b"] = home_b2b
    df["away_b2b"] = away_b2b

    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    # add rest features
    df = add_rest_features(df)

    # basic Elo-based feature
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    # postseason flag as int
    df["is_postseason"] = df["postseason"].astype(int)

    # drop games where we don't know rest days (first game of each team)
    df = df.dropna(subset=["home_rest_days", "away_rest_days"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved basic feature dataset to {OUTPUT_CSV} with {len(df)} games.")
