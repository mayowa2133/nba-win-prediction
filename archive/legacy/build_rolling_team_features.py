import pandas as pd

INPUT_CSV = "games_all_2015_2025_features_basic.csv"
OUTPUT_CSV = "games_all_2015_2025_features_rolling.csv"


def build_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team & season, compute pre-game rolling stats using ONLY past games:
      - simple_off (avg points scored per game)
      - simple_def (avg points allowed per game)
      - simple_net (avg point diff per game)
      - simple_win_pct (wins / games_played)
      - games_played_so_far

    Then add home/away versions and diff features to the dataframe.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # state[(season, team)] = dict with rolling totals
    state = {}

    home_simple_off = []
    home_simple_def = []
    home_simple_net = []
    home_simple_win_pct = []
    home_games_so_far = []

    away_simple_off = []
    away_simple_def = []
    away_simple_net = []
    away_simple_win_pct = []
    away_games_so_far = []

    for _, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        key_home = (season, home)
        key_away = (season, away)

        # --- HOME pre-game stats ---
        st_h = state.get(key_home, {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0})
        if st_h["gp"] > 0:
            h_off = st_h["pts_for"] / st_h["gp"]
            h_def = st_h["pts_against"] / st_h["gp"]
            h_net = (st_h["pts_for"] - st_h["pts_against"]) / st_h["gp"]
            h_win_pct = st_h["wins"] / st_h["gp"]
        else:
            h_off = None
            h_def = None
            h_net = None
            h_win_pct = None

        home_simple_off.append(h_off)
        home_simple_def.append(h_def)
        home_simple_net.append(h_net)
        home_simple_win_pct.append(h_win_pct)
        home_games_so_far.append(st_h["gp"])

        # --- AWAY pre-game stats ---
        st_a = state.get(key_away, {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0})
        if st_a["gp"] > 0:
            a_off = st_a["pts_for"] / st_a["gp"]
            a_def = st_a["pts_against"] / st_a["gp"]
            a_net = (st_a["pts_for"] - st_a["pts_against"]) / st_a["gp"]
            a_win_pct = st_a["wins"] / st_a["gp"]
        else:
            a_off = None
            a_def = None
            a_net = None
            a_win_pct = None

        away_simple_off.append(a_off)
        away_simple_def.append(a_def)
        away_simple_net.append(a_net)
        away_simple_win_pct.append(a_win_pct)
        away_games_so_far.append(st_a["gp"])

        # --- update state AFTER the game ---

        # home
        if key_home not in state:
            state[key_home] = {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0}
        state[key_home]["gp"] += 1
        state[key_home]["pts_for"] += home_score
        state[key_home]["pts_against"] += away_score
        if home_score > away_score:
            state[key_home]["wins"] += 1

        # away
        if key_away not in state:
            state[key_away] = {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0}
        state[key_away]["gp"] += 1
        state[key_away]["pts_for"] += away_score
        state[key_away]["pts_against"] += home_score
        if away_score > home_score:
            state[key_away]["wins"] += 1

    # attach rolling features
    df["home_simple_off"] = home_simple_off
    df["home_simple_def"] = home_simple_def
    df["home_simple_net"] = home_simple_net
    df["home_simple_win_pct"] = home_simple_win_pct
    df["home_games_so_far"] = home_games_so_far

    df["away_simple_off"] = away_simple_off
    df["away_simple_def"] = away_simple_def
    df["away_simple_net"] = away_simple_net
    df["away_simple_win_pct"] = away_simple_win_pct
    df["away_games_so_far"] = away_games_so_far

    # diff features (home - away)
    df["simple_off_diff"] = df["home_simple_off"] - df["away_simple_off"]
    df["simple_def_diff"] = df["home_simple_def"] - df["away_simple_def"]
    df["simple_net_diff"] = df["home_simple_net"] - df["away_simple_net"]
    df["simple_win_pct_diff"] = df["home_simple_win_pct"] - df["away_simple_win_pct"]
    df["games_played_diff"] = df["home_games_so_far"] - df["away_games_so_far"]

    # drop games where one of the teams had no prior games (first game of season)
    df = df.dropna(
        subset=[
            "home_simple_off",
            "away_simple_off",
            "home_simple_def",
            "away_simple_def",
        ]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df = build_rolling_stats(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved rolling feature dataset to {OUTPUT_CSV} with {len(df)} games.")
