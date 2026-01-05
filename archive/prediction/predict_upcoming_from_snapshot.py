import json
import os
from datetime import datetime, date, timedelta

import joblib
import pandas as pd

from team_state_snapshot import TeamStateSnapshot

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_config.json")

# This should be your base games file with scores (no rolling)
GAMES_CSV = "games_all_2015_2025_features_basic.csv"


def load_model_and_config():
    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    feature_cols = config["feature_cols"]
    return model, feature_cols


if __name__ == "__main__":
    model, feature_cols = load_model_and_config()

    # 1) Choose target date (tomorrow) – for now, just pick something
    # In real use, you'd do date.today() + timedelta(days=1)
    target_date_str = "2025-11-23"  # example
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    # 2) Load historical games (with scores)
    games_df = pd.read_csv(GAMES_CSV)

    # 3) Build team state snapshot as of BEFORE target_date
    snap = TeamStateSnapshot()
    snap.build_from_games(games_df, cutoff_date=target_date)

    # 4) Define tomorrow's schedule (for now, manually)
    games_tomorrow = [
        {"season": 2025, "date": target_date_str, "home_team": "MIL", "away_team": "BOS"},
        {"season": 2025, "date": target_date_str, "home_team": "LAL", "away_team": "PHX"},
        # add more...
    ]

    # 5) Build feature rows for each upcoming game
    rows = []
    for g in games_tomorrow:
        season = g["season"]
        d = target_date
        home = g["home_team"]
        away = g["away_team"]

        fh = snap.get_team_features(season, home, d)
        fa = snap.get_team_features(season, away, d)

        # Build the SAME features your model expects
        elo_diff = fh["elo"] - fa["elo"]

        home_rest_days = fh["rest_days"] if fh["rest_days"] is not None else 3.0
        away_rest_days = fa["rest_days"] if fa["rest_days"] is not None else 3.0

        home_b2b = fh["b2b"]
        away_b2b = fa["b2b"]

        is_postseason = 0  # for regular season

        simple_off_diff = (fh["simple_off"] or 0.0) - (fa["simple_off"] or 0.0)
        simple_def_diff = (fh["simple_def"] or 0.0) - (fa["simple_def"] or 0.0)
        simple_net_diff = (fh["simple_net"] or 0.0) - (fa["simple_net"] or 0.0)
        simple_win_pct_diff = (fh["simple_win_pct"] or 0.0) - (fa["simple_win_pct"] or 0.0)
        games_played_diff = fh["games_played"] - fa["games_played"]

        lastN_off_diff = (fh["lastN_off"] or 0.0) - (fa["lastN_off"] or 0.0)
        lastN_def_diff = (fh["lastN_def"] or 0.0) - (fa["lastN_def"] or 0.0)
        lastN_net_diff = (fh["lastN_net"] or 0.0) - (fa["lastN_net"] or 0.0)
        lastN_win_pct_diff = (fh["lastN_win_pct"] or 0.0) - (fa["lastN_win_pct"] or 0.0)
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
            # Plus some context columns:
            "season": season,
            "date": d,
            "home_team": home,
            "away_team": away,
        }
        rows.append(feature_row)

    df_features = pd.DataFrame(rows)

    # 6) Run the model
    X = df_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]

    df_features["pred_home_win_prob"] = probs

    print(f"Predictions for upcoming games on {target_date_str}:\n")
    for _, row in df_features.sort_values("pred_home_win_prob", ascending=False).iterrows():
        print(
            f"Season {row['season']} | {row['away_team']} @ {row['home_team']} | "
            f"P(home win) = {row['pred_home_win_prob']:.3f}"
        )
