import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

INPUT_CSV = "games_all_2015_2025_features_teamstats.csv"

TRAIN_START, TRAIN_END = 2015, 2020
VAL_START, VAL_END = 2021, 2022
TEST_START, TEST_END = 2023, 2025

# Base features from before
BASE_FEATURES = [
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "is_postseason",
]

# New team stat diff features
TEAM_DIFF_FEATURES = [
    "off_rating_diff",
    "def_rating_diff",
    "net_rating_diff",
    "pace_diff",
    "efg_pct_diff",
    "ts_pct_diff",
    "oreb_pct_diff",
    "dreb_pct_diff",
    "reb_pct_diff",
    "tm_tov_pct_diff",
    "w_pct_diff",
    "pie_diff",
]


def load_data():
    df = pd.read_csv(INPUT_CSV)

    # Ensure numeric where needed
    df["home_rest_days"] = df["home_rest_days"].astype(float)
    df["away_rest_days"] = df["away_rest_days"].astype(float)

    # Only keep team diff features that actually exist
    team_features = [c for c in TEAM_DIFF_FEATURES if c in df.columns]
    feature_cols = BASE_FEATURES + team_features

    print("Using feature columns:")
    for c in feature_cols:
        print(" -", c)

    X = df[feature_cols].values
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

    print(
        f"\nTrain size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}"
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # Validation
    p_val = model.predict_proba(X_val)[:, 1]
    yhat_val = (p_val >= 0.5).astype(int)
    acc_val = accuracy_score(y_val, yhat_val)
    ll_val = log_loss(y_val, p_val)

    print("\n=== Validation (2021–2022) ===")
    print(f"Accuracy: {acc_val:.3f}")
    print(f"LogLoss:  {ll_val:.3f}")

    # Test
    p_test = model.predict_proba(X_test)[:, 1]
    yhat_test = (p_test >= 0.5).astype(int)
    acc_test = accuracy_score(y_test, yhat_test)
    ll_test = log_loss(y_test, p_test)

    print("\n=== Test (2023–2025) ===")
    print(f"Accuracy: {acc_test:.3f}")
    print(f"LogLoss:  {ll_test:.3f}")
