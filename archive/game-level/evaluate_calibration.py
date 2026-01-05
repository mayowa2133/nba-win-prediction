import json
import os
import pandas as pd
import joblib
from sklearn.metrics import log_loss

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
    test_start, test_end = config["test_seasons"]
    return model, feature_cols, data_csv, test_start, test_end


if __name__ == "__main__":
    model, feature_cols, data_csv, test_start, test_end = load_model_and_config()

    df = pd.read_csv(data_csv)

    # Focus on test seasons only
    mask = (df["season"] >= test_start) & (df["season"] <= test_end)
    df_test = df.loc[mask].copy()

    # Ensure numeric
    if "home_rest_days" in df_test.columns:
        df_test["home_rest_days"] = df_test["home_rest_days"].astype(float)
    if "away_rest_days" in df_test.columns:
        df_test["away_rest_days"] = df_test["away_rest_days"].astype(float)

    X = df_test[feature_cols].values
    y = df_test["home_win"].values

    p = model.predict_proba(X)[:, 1]

    overall_ll = log_loss(y, p)
    print(f"Test seasons: {test_start}-{test_end}")
    print(f"Overall test log loss: {overall_ll:.3f}")
    print(f"Total test games: {len(y)}\n")

    df_test["pred_home_win_prob"] = p

    # Define probability buckets
    buckets = [
        (0.0, 0.5),
        (0.5, 0.55),
        (0.55, 0.6),
        (0.6, 0.65),
        (0.65, 0.7),
        (0.7, 0.75),
        (0.75, 0.8),
        (0.8, 1.01),  # include 1.0
    ]

    print(f"{'Bucket':<15} {'Games':>6} {'AvgPred':>10} {'ActualWin%':>11}")
    print("-" * 46)

    for low, high in buckets:
        mask_bucket = (df_test["pred_home_win_prob"] >= low) & (df_test["pred_home_win_prob"] < high)
        df_bucket = df_test.loc[mask_bucket]

        n = len(df_bucket)
        if n == 0:
            avg_pred = float("nan")
            actual = float("nan")
        else:
            avg_pred = df_bucket["pred_home_win_prob"].mean()
            actual = df_bucket["home_win"].mean()

        bucket_label = f"[{low:.2f},{high:.2f})"
        print(f"{bucket_label:<15} {n:>6d} {avg_pred:>10.3f} {actual:>11.3f}")
