import json
import os
import sys
import pandas as pd
import joblib
from datetime import datetime

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "logreg_elo_rolling_last10_config.json")


def load_model_and_config():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_final_model.py first.")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}. Run train_final_model.py first.")

    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    feature_cols = config["feature_cols"]
    data_csv = config["data_csv"]  # games_all_2015_2025_features_rolling_last10.csv
    return model, feature_cols, data_csv


def get_historical_matchups_for_date(df: pd.DataFrame, target_date_str: str) -> pd.DataFrame:
    """
    Historical mode:
    - We already have all features precomputed in df.
    - We just filter by date and use the rows directly.
    """
    df["date"] = pd.to_datetime(df["date"]).dt.date

    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    mask = df["date"] == target_date
    day_games = df.loc[mask].copy()
    return day_games


def get_schedule_from_api_stub(target_date_str: str):
    """
    TEMPLATE for live mode – doesn't do anything yet.

    Later, you can replace this with a real call to BallDontLie / ESPN / NBA API, e.g.:

      - return a list of dicts:
          [
             {"season": 2025, "date": "2025-11-23", "home_team": "MIL", "away_team": "BOS"},
             ...
          ]

    For now, we just return an empty list and rely on historical mode.
    """
    return []


def main():
    # --- Parse date argument ---
    if len(sys.argv) > 1:
        target_date_str = sys.argv[1]
    else:
        # If no date is provided, we’ll default to the LAST date in the dataset (most recent historical day)
        print("No date provided, will default to latest date in data.\n")
        target_date_str = None

    # --- Load model + config + data ---
    model, feature_cols, data_csv = load_model_and_config()
    df = pd.read_csv(data_csv)

    # If no date was passed, compute it from data (max date)
    if target_date_str is None:
        df_dates = pd.to_datetime(df["date"]).dt.date
        latest_date = df_dates.max()
        target_date_str = latest_date.strftime("%Y-%m-%d")
        print(f"Using latest dataset date: {target_date_str}\n")

    # --- HISTORICAL MODE: use rows already in df for that date ---
    day_games = get_historical_matchups_for_date(df, target_date_str)

    if day_games.empty:
        print(f"No historical games found on {target_date_str}.")
        print("In the future, this is where you'd call a schedule API and compute features for upcoming games.\n")

        # Example stub call (currently returns empty):
        schedule = get_schedule_from_api_stub(target_date_str)
        print(f"Schedule from API stub (for {target_date_str}): {schedule}")
        return

    # Ensure numeric on rest columns
    if "home_rest_days" in day_games.columns:
        day_games["home_rest_days"] = day_games["home_rest_days"].astype(float)
    if "away_rest_days" in day_games.columns:
        day_games["away_rest_days"] = day_games["away_rest_days"].astype(float)

    # Build feature matrix exactly as model expects
    X = day_games[feature_cols].values
    probs = model.predict_proba(X)[:, 1]  # P(home win)

    day_games = day_games.assign(pred_home_win_prob=probs)

    print(f"Predictions for {target_date_str} ({len(day_games)} games):\n")
    for _, row in day_games.sort_values("pred_home_win_prob", ascending=False).iterrows():
        home = row["home_team"]
        away = row["away_team"]
        season = row["season"]
        prob = row["pred_home_win_prob"]
        hs = row.get("home_score", None)
        as_ = row.get("away_score", None)

        line = f"Season {season} | {away} @ {home} | P(home win) = {prob:.3f}"

        # If we have final scores (historical), show them
        if pd.notna(hs) and pd.notna(as_):
            actual = "home win" if row["home_win"] == 1 else "home loss"
            line += f" | Final: {home} {hs} - {away} {as_} ({actual})"

        print(line)


if __name__ == "__main__":
    main()
