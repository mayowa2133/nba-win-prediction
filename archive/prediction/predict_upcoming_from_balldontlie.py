import json
import os
import sys
from datetime import datetime, date, timedelta

import joblib
import pandas as pd
import requests

from team_state_snapshot import TeamStateSnapshot

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_config.json")

# Basic games file: season, date, home_team, away_team, home_score, away_score, etc.
GAMES_CSV = "games_all_2015_2025_features_basic.csv"

BALLDONTLIE_URL = "https://api.balldontlie.io/v1/games"


def load_model_and_config():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_final_model.py first.")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}. Run train_final_model.py first.")

    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    feature_cols = config["feature_cols"]
    return model, feature_cols


def get_target_date_from_args_or_data(games_df: pd.DataFrame) -> date:
    """
    If user passes a date (YYYY-MM-DD) as argv[1], use that.
    Otherwise, use (max date in data + 1 day) as a proxy for 'tomorrow'
    in your offline dataset.
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
    Returns a list of dicts: {season, date, home_team, away_team, postseason}
    """
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError("BALLDONTLIE_API_KEY env var not set.")

    target_str = target_date.strftime("%Y-%m-%d")

    headers = {
        "Authorization": api_key
    }
    params = {
        "dates[]": target_str,
        "per_page": 100
    }

    resp = requests.get(BALLDONTLIE_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    games = []
    for g in data:
        season = g["season"]
        d_str = g["date"]  # "YYYY-MM-DD" per docs
        # BallDontLie returns "home_team" and "visitor_team" objects with abbreviation
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


if __name__ == "__main__":
    # 1) Load model
    model, feature_cols = load_model_and_config()

    # 2) Load historical games for state snapshot
    if not os.path.exists(GAMES_CSV):
        raise FileNotFoundError(f"{GAMES_CSV} not found. Make sure your basic games file exists.")

    games_df = pd.read_csv(GAMES_CSV)

    # 3) Decide target date
    target_date = get_target_date_from_args_or_data(games_df)
    target_date_str = target_date.strftime("%Y-%m-%d")
    print(f"Predicting games for {target_date_str}\n")

    # 4) Build team state snapshot as of BEFORE target_date
    snap = TeamStateSnapshot()
    snap.build_from_games(games_df, cutoff_date=target_date)

    # 5) Fetch slate from BallDontLie
    games_tomorrow = fetch_schedule_from_balldontlie(target_date)

    if not games_tomorrow:
        print("No games returned by BallDontLie for this date.")
        sys.exit(0)

    # 6) Build feature rows for each scheduled game
    rows = []
    for g in games_tomorrow:
        season = g["season"]
        d = target_date
        home = g["home_team"]
        away = g["away_team"]
        is_postseason = 1 if g.get("postseason", False) else 0

        fh = snap.get_team_features(season, home, d)
        fa = snap.get_team_features(season, away, d)

        # If a team has not played yet, you'll see None; fall back to 0 or neutral defaults
        def nz(x, default=0.0):
            return x if x is not None else default

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
            # context
            "season": season,
            "date": d,
            "home_team": home,
            "away_team": away,
        }
        rows.append(feature_row)

    df_features = pd.DataFrame(rows)

    # Make sure all feature_cols exist
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise RuntimeError(f"Missing features in df_features: {missing}")

    # 7) Run model
    X = df_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    df_features["pred_home_win_prob"] = probs

    # 8) Print results
    print(f"Predictions for games on {target_date_str} (from BallDontLie):\n")
    for _, row in df_features.sort_values("pred_home_win_prob", ascending=False).iterrows():
        print(
            f"Season {row['season']} | {row['away_team']} @ {row['home_team']} | "
            f"P(home win) = {row['pred_home_win_prob']:.3f} "
            f"{'(playoffs)' if row['is_postseason'] == 1 else ''}"
        )
