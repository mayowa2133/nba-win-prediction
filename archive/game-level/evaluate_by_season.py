import json
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, log_loss

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
    return model, feature_cols, data_csv, config


if __name__ == "__main__":
    model, feature_cols, data_csv, config = load_model_and_config()

    df = pd.read_csv(data_csv)

    # Ensure numeric for rest columns (just in case)
    if "home_rest_days" in df.columns:
        df["home_rest_days"] = df["home_rest_days"].astype(float)
    if "away_rest_days" in df.columns:
        df["away_rest_days"] = df["away_rest_days"].astype(float)

    seasons = sorted(df["season"].unique().tolist())

    print(f"Evaluating model per season on {data_csv}")
    print(f"Seasons found: {seasons}\n")

    rows = []
    for season in seasons:
        mask = df["season"] == season
        df_season = df.loc[mask]

        X = df_season[feature_cols].values
        y = df_season["home_win"].values

        if len(y) == 0:
            continue

        p = model.predict_proba(X)[:, 1]
        yhat = (p >= 0.5).astype(int)

        acc = accuracy_score(y, yhat)
        ll = log_loss(y, p)

        rows.append((season, len(y), acc, ll))

    # Print a nice table
    print(f"{'Season':<8} {'Games':>6} {'Accuracy':>10} {'LogLoss':>10}")
    print("-" * 38)
    for season, n, acc, ll in rows:
        print(f"{season:<8d} {n:>6d} {acc:>10.3f} {ll:>10.3f}")
