import pandas as pd

INPUT_CSV = "games_all_2015_2025_features_basic.csv"
OUTPUT_CSV = "games_all_2015_2025_features_rolling_last10.csv"

LAST_N = 10


def build_rolling_with_lastN(df: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:
    """
    For each (season, team), compute pre-game rolling stats using ONLY past games:

    Season-to-date:
      - simple_off (avg points scored per game)
      - simple_def (avg points allowed per game)
      - simple_net (avg point diff per game)
      - simple_win_pct (wins / games_played)
      - games_so_far

    Last-N-games:
      - lastN_off (avg points scored over last N games)
      - lastN_def
      - lastN_net
      - lastN_win_pct
      - lastN_games (how many games in that window, <= N)

    Then add home/away versions and diff features to the dataframe.
    """

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # State for season-to-date
    full_state = {}  # (season, team) -> {gp, pts_for, pts_against, wins}

    # State for last N games
    last_state = {}  # (season, team) -> {pts_for: [], pts_against: [], wins: []}

    # --- containers for features ---

    # Season-to-date (home)
    home_simple_off = []
    home_simple_def = []
    home_simple_net = []
    home_simple_win_pct = []
    home_games_so_far = []

    # Season-to-date (away)
    away_simple_off = []
    away_simple_def = []
    away_simple_net = []
    away_simple_win_pct = []
    away_games_so_far = []

    # Last-N (home)
    home_lastN_off = []
    home_lastN_def = []
    home_lastN_net = []
    home_lastN_win_pct = []
    home_lastN_games = []

    # Last-N (away)
    away_lastN_off = []
    away_lastN_def = []
    away_lastN_net = []
    away_lastN_win_pct = []
    away_lastN_games = []

    for _, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        key_home = (season, home)
        key_away = (season, away)

        # ---------- HOME pre-game: season-to-date ----------
        st_h = full_state.get(key_home, {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0})
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

        # ---------- AWAY pre-game: season-to-date ----------
        st_a = full_state.get(key_away, {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0})
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

        # ---------- HOME pre-game: last N ----------
        lst_h = last_state.get(key_home, {"pts_for": [], "pts_against": [], "wins": []})
        if len(lst_h["pts_for"]) > 0:
            h_lastN_off = sum(lst_h["pts_for"]) / len(lst_h["pts_for"])
            h_lastN_def = sum(lst_h["pts_against"]) / len(lst_h["pts_against"])
            h_lastN_net = sum(
                pf - pa for pf, pa in zip(lst_h["pts_for"], lst_h["pts_against"])
            ) / len(lst_h["pts_for"])
            h_lastN_win_pct = sum(lst_h["wins"]) / len(lst_h["wins"])
            h_lastN_games = len(lst_h["pts_for"])
        else:
            h_lastN_off = None
            h_lastN_def = None
            h_lastN_net = None
            h_lastN_win_pct = None
            h_lastN_games = 0

        home_lastN_off.append(h_lastN_off)
        home_lastN_def.append(h_lastN_def)
        home_lastN_net.append(h_lastN_net)
        home_lastN_win_pct.append(h_lastN_win_pct)
        home_lastN_games.append(h_lastN_games)

        # ---------- AWAY pre-game: last N ----------
        lst_a = last_state.get(key_away, {"pts_for": [], "pts_against": [], "wins": []})
        if len(lst_a["pts_for"]) > 0:
            a_lastN_off = sum(lst_a["pts_for"]) / len(lst_a["pts_for"])
            a_lastN_def = sum(lst_a["pts_against"]) / len(lst_a["pts_against"])
            a_lastN_net = sum(
                pf - pa for pf, pa in zip(lst_a["pts_for"], lst_a["pts_against"])
            ) / len(lst_a["pts_for"])
            a_lastN_win_pct = sum(lst_a["wins"]) / len(lst_a["wins"])
            a_lastN_games = len(lst_a["pts_for"])
        else:
            a_lastN_off = None
            a_lastN_def = None
            a_lastN_net = None
            a_lastN_win_pct = None
            a_lastN_games = 0

        away_lastN_off.append(a_lastN_off)
        away_lastN_def.append(a_lastN_def)
        away_lastN_net.append(a_lastN_net)
        away_lastN_win_pct.append(a_lastN_win_pct)
        away_lastN_games.append(a_lastN_games)

        # ---------- UPDATE STATES AFTER GAME ----------

        # full_state: season-to-date
        if key_home not in full_state:
            full_state[key_home] = {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0}
        if key_away not in full_state:
            full_state[key_away] = {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0}

        full_state[key_home]["gp"] += 1
        full_state[key_home]["pts_for"] += home_score
        full_state[key_home]["pts_against"] += away_score
        if home_score > away_score:
            full_state[key_home]["wins"] += 1

        full_state[key_away]["gp"] += 1
        full_state[key_away]["pts_for"] += away_score
        full_state[key_away]["pts_against"] += home_score
        if away_score > home_score:
            full_state[key_away]["wins"] += 1

        # last_state: last N
        if key_home not in last_state:
            last_state[key_home] = {"pts_for": [], "pts_against": [], "wins": []}
        if key_away not in last_state:
            last_state[key_away] = {"pts_for": [], "pts_against": [], "wins": []}

        # append home
        last_state[key_home]["pts_for"].append(home_score)
        last_state[key_home]["pts_against"].append(away_score)
        last_state[key_home]["wins"].append(1 if home_score > away_score else 0)
        if len(last_state[key_home]["pts_for"]) > last_n:
            last_state[key_home]["pts_for"].pop(0)
            last_state[key_home]["pts_against"].pop(0)
            last_state[key_home]["wins"].pop(0)

        # append away
        last_state[key_away]["pts_for"].append(away_score)
        last_state[key_away]["pts_against"].append(home_score)
        last_state[key_away]["wins"].append(1 if away_score > home_score else 0)
        if len(last_state[key_away]["pts_for"]) > last_n:
            last_state[key_away]["pts_for"].pop(0)
            last_state[key_away]["pts_against"].pop(0)
            last_state[key_away]["wins"].pop(0)

    # Attach season-to-date rolling features
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

    df["simple_off_diff"] = df["home_simple_off"] - df["away_simple_off"]
    df["simple_def_diff"] = df["home_simple_def"] - df["away_simple_def"]
    df["simple_net_diff"] = df["home_simple_net"] - df["away_simple_net"]
    df["simple_win_pct_diff"] = df["home_simple_win_pct"] - df["away_simple_win_pct"]
    df["games_played_diff"] = df["home_games_so_far"] - df["away_games_so_far"]

    # Attach last-N rolling features
    df["home_lastN_off"] = home_lastN_off
    df["home_lastN_def"] = home_lastN_def
    df["home_lastN_net"] = home_lastN_net
    df["home_lastN_win_pct"] = home_lastN_win_pct
    df["home_lastN_games"] = home_lastN_games

    df["away_lastN_off"] = away_lastN_off
    df["away_lastN_def"] = away_lastN_def
    df["away_lastN_net"] = away_lastN_net
    df["away_lastN_win_pct"] = away_lastN_win_pct
    df["away_lastN_games"] = away_lastN_games

    df["lastN_off_diff"] = df["home_lastN_off"] - df["away_lastN_off"]
    df["lastN_def_diff"] = df["home_lastN_def"] - df["away_lastN_def"]
    df["lastN_net_diff"] = df["home_lastN_net"] - df["away_lastN_net"]
    df["lastN_win_pct_diff"] = df["home_lastN_win_pct"] - df["away_lastN_win_pct"]
    df["lastN_games_diff"] = df["home_lastN_games"] - df["away_lastN_games"]

    # Drop games where we don't have any prior data for either team
    df = df.dropna(
        subset=["home_simple_off", "away_simple_off", "home_lastN_off", "away_lastN_off"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df = build_rolling_with_lastN(df, last_n=LAST_N)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved rolling+last{LAST_N} feature dataset to {OUTPUT_CSV} with {len(df)} games.")
