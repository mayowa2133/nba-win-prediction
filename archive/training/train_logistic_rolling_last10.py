import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

INPUT_CSV = "games_all_2015_2025_features_rolling_last10.csv"

TRAIN_START, TRAIN_END = 2015, 2020
VAL_START, VAL_END = 2021, 2022
TEST_START, TEST_END = 2023, 2025

FEATURE_COLS = [
    # Core
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "is_postseason",

    # Season-to-date rolling
    "simple_off_diff",
    "simple_def_diff",
    "simple_net_diff",
    "simple_win_pct_diff",
    "games_played_diff",

    # Last-10 rolling
    "lastN_off_diff",
    "lastN_def_diff",
    "lastN_net_diff",
    "lastN_win_pct_diff",
    "lastN_games_diff",
]


def load_data():
    df = pd.read_csv(INPUT_CSV)

    df["home_rest_days"] = df["home_rest_days"].astype(float)
    df["away_rest_days"] = df["away_rest_days"].astype(float)

    X = df[FEATURE_COLS].values
    y = df["home_win"].values
    seasons = df["season"].values

    def mask(start, end):
        return (seasons >= start) & (seasons <= end)

    X_train = X[mask(TRAIN_START, TRAIN_END)]
    y_train = y[mask(TRAIN_START, TRAIN_END)]
    X_val = X[mask(VAL_START, VAL_END)]
    y_val = y[mask(VAL_START, VAL_END)]
    X_test = X[mask(TEST_START, TEST_END)]
    y_test = y[mask(TEST_START, TEST_END)]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # validation
    p_val = model.predict_proba(X_val)[:, 1]
    yhat_val = (p_val >= 0.5).astype(int)
    acc_val = accuracy_score(y_val, yhat_val)
    ll_val = log_loss(y_val, p_val)

    print("\n=== Validation (2021–2022) ===")
    print(f"Accuracy: {acc_val:.3f}")
    print(f"LogLoss:  {ll_val:.3f}")

    # test
    p_test = model.predict_proba(X_test)[:, 1]
    yhat_test = (p_test >= 0.5).astype(int)
    acc_test = accuracy_score(y_test, yhat_test)
    ll_test = log_loss(y_test, p_test)

    print("\n=== Test (2023–2025) ===")
    print(f"Accuracy: {acc_test:.3f}")
    print(f"LogLoss:  {ll_test:.3f}")
