import json
import os
import sys
from datetime import datetime, date, timedelta

import joblib
import pandas as pd
import requests

from team_state_snapshot import TeamStateSnapshot

# === Paths ===
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_prevstats.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_prevstats_config.json")

# Basic games file (for Elo + rolling state)
GAMES_CSV = "games_all_2015_2025_features_basic.csv"

# Team season advanced stats (NBA API dump you created earlier)
TEAM_STATS_CSV = "team_stats_2015_2025.csv"

BALLDONTLIE_URL = "https://api.balldontlie.io/v1/games"


def load_model_and_config():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}.")

    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    feature_cols = config["feature_cols"]
    return model, feature_cols


def get_target_date_from_args_or_data(games_df: pd.DataFrame) -> date:
    """
    If user passes a date (YYYY-MM-DD) as argv[1], use that.
    Otherwise, use (max date in data + 1 day) as a proxy for 'tomorrow'.
    """
    if len(sys.argv) > 1:
        target_date_str = sys.argv[1]
        return datetime.strptime(target_date_str, "%Y-%m-%d").date()
    else:
        games_df["date"] = pd.to_datetime(games_df["date"]).dt.date
        max_date = games_df["date"].max()
        target_date = max_date + timedelta(days=1)
        print(f"No date arg provided, using dataset max_date+1 = {target_date}")
        return target_date


def fetch_schedule_from_balldontlie(target_date: date):
    """
    Call BallDontLie to get the games on target_date.
    Returns a list of dicts: {season, date, home_team, away_team, postseason}.
    """
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError("BALLDONTLIE_API_KEY env var not set.")

    target_str = target_date.strftime("%Y-%m-%d")

    headers = {"Authorization": api_key}
    params = {"dates[]": target_str, "per_page": 100}

    resp = requests.get(BALLDONTLIE_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    games = []
    for g in data:
        season = g["season"]
        d_str = g["date"]  # "YYYY-MM-DD"
        home_abbr = g["home_team"]["abbreviation"]
        away_abbr = g["visitor_team"]["abbreviation"]
        is_postseason = g.get("postseason", False)

        games.append(
            {
                "season": season,
                "date": d_str,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "postseason": is_postseason,
            }
        )

    return games


def load_prev_team_stats():
    """
    Load previous-season team advanced stats into a lookup:
    (season_int, team_abbrev) -> dict of advanced stats.
    """
    if not os.path.exists(TEAM_STATS_CSV):
        raise FileNotFoundError(f"{TEAM_STATS_CSV} not found.")

    team_stats = pd.read_csv(TEAM_STATS_CSV)

    name_to_abbrev = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Charlotte Bobcats": "CHA",
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
        "New Orleans Hornets": "NOP",
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

    cols = [
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
    ts = team_stats[cols].copy()

    lookup = {}
    for _, row in ts.iterrows():
        season_int = int(row["SEASON_INT"])
        abbr = row["TEAM_ABBREVIATION"]
        lookup[(season_int, abbr)] = {
            "OFF_RATING": row["OFF_RATING"],
            "DEF_RATING": row["DEF_RATING"],
            "NET_RATING": row["NET_RATING"],
            "EFG_PCT": row["EFG_PCT"],
            "TS_PCT": row["TS_PCT"],
            "PACE": row["PACE"],
            "OREB_PCT": row["OREB_PCT"],
            "DREB_PCT": row["DREB_PCT"],
            "REB_PCT": row["REB_PCT"],
            "TM_TOV_PCT": row["TM_TOV_PCT"],
            "W_PCT": row["W_PCT"],
            "PIE": row["PIE"],
        }

    return lookup


if __name__ == "__main__":
    # 1) Load model + feature_cols
    model, feature_cols = load_model_and_config()

    # 2) Load historical games for state snapshot (Elo + rolling)
    if not os.path.exists(GAMES_CSV):
        raise FileNotFoundError(f"{GAMES_CSV} not found.")
    games_df = pd.read_csv(GAMES_CSV)

    # 3) Decide target date
    target_date = get_target_date_from_args_or_data(games_df)
    target_date_str = target_date.strftime("%Y-%m-%d")
    print(f"Predicting games for {target_date_str}\n")

    # 4) Build team state snapshot as of BEFORE target_date
    snap = TeamStateSnapshot()
    snap.build_from_games(games_df, cutoff_date=target_date)

    # 5) Load prev-season team stats lookup
    prev_stats_lookup = load_prev_team_stats()

    # 6) Fetch slate from BallDontLie
    games_tomorrow = fetch_schedule_from_balldontlie(target_date)
    if not games_tomorrow:
        print("No games returned by BallDontLie for this date.")
        sys.exit(0)

    # 7) Build feature rows for each scheduled game
    rows = []

    def nz(x, default=0.0):
        return x if x is not None else default

    for g in games_tomorrow:
        season = g["season"]
        d = target_date
        home = g["home_team"]
        away = g["away_team"]
        is_postseason = 1 if g.get("postseason", False) else 0

        fh = snap.get_team_features(season, home, d)
        fa = snap.get_team_features(season, away, d)

        # Current-season dynamic features
        elo_diff = fh["elo"] - fa["elo"]
        home_rest_days = fh["rest_days"] if fh["rest_days"] is not None else 3.0
        away_rest_days = fa["rest_days"] if fa["rest_days"] is not None else 3.0
        home_b2b = fh["b2b"]
        away_b2b = fa["b2b"]

        simple_off_diff = nz(fh["simple_off"]) - nz(fa["simple_off"])
        simple_def_diff = nz(fh["simple_def"]) - nz(fa["simple_def"])
        simple_net_diff = nz(fh["simple_net"]) - nz(fa["simple_net"])
        simple_win_pct_diff = nz(fh["simple_win_pct"]) - nz(fa["simple_win_pct"])
        games_played_diff = fh["games_played"] - fa["games_played"]

        lastN_off_diff = nz(fh["lastN_off"]) - nz(fa["lastN_off"])
        lastN_def_diff = nz(fh["lastN_def"]) - nz(fa["lastN_def"])
        lastN_net_diff = nz(fh["lastN_net"]) - nz(fa["lastN_net"])
        lastN_win_pct_diff = nz(fh["lastN_win_pct"]) - nz(fa["lastN_win_pct"])
        lastN_games_diff = fh["lastN_games"] - fa["lastN_games"]

        # Prev-season advanced stats diff (season-1)
        prev_season = season - 1
        home_prev = prev_stats_lookup.get((prev_season, home))
        away_prev = prev_stats_lookup.get((prev_season, away))

        if home_prev is not None and away_prev is not None:
            prev_off_rating_diff = home_prev["OFF_RATING"] - away_prev["OFF_RATING"]
            prev_def_rating_diff = home_prev["DEF_RATING"] - away_prev["DEF_RATING"]
            prev_net_rating_diff = home_prev["NET_RATING"] - away_prev["NET_RATING"]
            prev_efg_pct_diff = home_prev["EFG_PCT"] - away_prev["EFG_PCT"]
            prev_ts_pct_diff = home_prev["TS_PCT"] - away_prev["TS_PCT"]
            prev_pace_diff = home_prev["PACE"] - away_prev["PACE"]
            prev_oreb_pct_diff = home_prev["OREB_PCT"] - away_prev["OREB_PCT"]
            prev_dreb_pct_diff = home_prev["DREB_PCT"] - away_prev["DREB_PCT"]
            prev_reb_pct_diff = home_prev["REB_PCT"] - away_prev["REB_PCT"]
            prev_tm_tov_pct_diff = home_prev["TM_TOV_PCT"] - away_prev["TM_TOV_PCT"]
            prev_w_pct_diff = home_prev["W_PCT"] - away_prev["W_PCT"]
            prev_pie_diff = home_prev["PIE"] - away_prev["PIE"]
        else:
            # Fallback if missing (e.g., very early seasons)
            prev_off_rating_diff = 0.0
            prev_def_rating_diff = 0.0
            prev_net_rating_diff = 0.0
            prev_efg_pct_diff = 0.0
            prev_ts_pct_diff = 0.0
            prev_pace_diff = 0.0
            prev_oreb_pct_diff = 0.0
            prev_dreb_pct_diff = 0.0
            prev_reb_pct_diff = 0.0
            prev_tm_tov_pct_diff = 0.0
            prev_w_pct_diff = 0.0
            prev_pie_diff = 0.0

        feature_row = {
            "elo_diff": elo_diff,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            "is_postseason": is_postseason,
            "simple_off_diff": simple_off_diff,
            "simple_def_diff": simple_def_diff,
            "simple_net_diff": simple_net_diff,
            "simple_win_pct_diff": simple_win_pct_diff,
            "games_played_diff": games_played_diff,
            "lastN_off_diff": lastN_off_diff,
            "lastN_def_diff": lastN_def_diff,
            "lastN_net_diff": lastN_net_diff,
            "lastN_win_pct_diff": lastN_win_pct_diff,
            "lastN_games_diff": lastN_games_diff,
            "prev_off_rating_diff": prev_off_rating_diff,
            "prev_def_rating_diff": prev_def_rating_diff,
            "prev_net_rating_diff": prev_net_rating_diff,
            "prev_efg_pct_diff": prev_efg_pct_diff,
            "prev_ts_pct_diff": prev_ts_pct_diff,
            "prev_pace_diff": prev_pace_diff,
            "prev_oreb_pct_diff": prev_oreb_pct_diff,
            "prev_dreb_pct_diff": prev_dreb_pct_diff,
            "prev_reb_pct_diff": prev_reb_pct_diff,
            "prev_tm_tov_pct_diff": prev_tm_tov_pct_diff,
            "prev_w_pct_diff": prev_w_pct_diff,
            "prev_pie_diff": prev_pie_diff,
            # context
            "season": season,
            "date": d,
            "home_team": home,
            "away_team": away,
        }
        rows.append(feature_row)

    df_features = pd.DataFrame(rows)

    # Check that we have all features expected by the model
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise RuntimeError(f"Missing features in df_features: {missing}")

    # 8) Run model
    X = df_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    df_features["pred_home_win_prob"] = probs

    # 9) Print results
    print(f"Predictions for games on {target_date_str} (from BallDontLie + prev stats):\n")
    for _, row in df_features.sort_values("pred_home_win_prob", ascending=False).iterrows():
        tag = "(playoffs)" if row["is_postseason"] == 1 else ""
        print(
            f"Season {row['season']} | {row['away_team']} @ {row['home_team']} | "
            f"P(home win) = {row['pred_home_win_prob']:.3f} {tag}"
        )
