import json
import os
import sys
import pandas as pd
import joblib

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
    data_csv = config["data_csv"]
    return model, feature_cols, data_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_day_from_history.py YYYY-MM-DD")
        raise SystemExit(1)

    target_date_str = sys.argv[1]

    model, feature_cols, data_csv = load_model_and_config()
    df = pd.read_csv(data_csv)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"]).dt.date

    try:
        target_date = pd.to_datetime(target_date_str).date()
    except Exception:
        raise SystemExit("Date must be in format YYYY-MM-DD")

    # Filter games on that date
    mask = df["date"] == target_date
    day_games = df.loc[mask].copy()

    if day_games.empty:
        print(f"No games found on {target_date}")
        return

    # Ensure numeric rest cols
    if "home_rest_days" in day_games.columns:
        day_games["home_rest_days"] = day_games["home_rest_days"].astype(float)
    if "away_rest_days" in day_games.columns:
        day_games["away_rest_days"] = day_games["away_rest_days"].astype(float)

    X = day_games[feature_cols].values
    probs = model.predict_proba(X)[:, 1]  # P(home win)

    day_games = day_games.assign(pred_home_win_prob=probs)

    print(f"Predictions for {target_date} ({len(day_games)} games):\n")
    for _, row in day_games.sort_values("pred_home_win_prob", ascending=False).iterrows():
        home = row["home_team"]
        away = row["away_team"]
        season = row["season"]
        prob = row["pred_home_win_prob"]
        hs = row["home_score"]
        as_ = row["away_score"]
        actual = "home win" if row["home_win"] == 1 else "home loss"

        print(
            f"Season {season} | {away} @ {home} | "
            f"P(home win) = {prob:.3f} | "
            f"Final: {home} {hs} - {away} {as_} ({actual})"
        )


if __name__ == "__main__":
    main()
