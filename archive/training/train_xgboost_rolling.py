import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

INPUT_CSV = "games_all_2015_2025_features_rolling.csv"

TRAIN_START, TRAIN_END = 2015, 2020
VAL_START, VAL_END = 2021, 2022
TEST_START, TEST_END = 2023, 2025

FEATURE_COLS = [
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "is_postseason",
    "simple_off_diff",
    "simple_def_diff",
    "simple_net_diff",
    "simple_win_pct_diff",
    "games_played_diff",
]


def load_data():
    df = pd.read_csv(INPUT_CSV)

    # ensure numeric
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

    print(
        f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}"
    )

    # Build DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLS)

    # Basic XGBoost params for binary classification
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.0,
    }

    evals = [(dtrain, "train"), (dval, "val")]

    # Train with early stopping on validation log loss
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    print(f"\nBest iteration: {bst.best_iteration}")

    # Validation performance
    p_val = bst.predict(dval)
    yhat_val = (p_val >= 0.5).astype(int)
    acc_val = accuracy_score(y_val, yhat_val)
    ll_val = log_loss(y_val, p_val)

    print("\n=== Validation (2021–2022) ===")
    print(f"Accuracy: {acc_val:.3f}")
    print(f"LogLoss:  {ll_val:.3f}")

    # Test performance
    p_test = bst.predict(dtest)
    yhat_test = (p_test >= 0.5).astype(int)
    acc_test = accuracy_score(y_test, yhat_test)
    ll_test = log_loss(y_test, p_test)

    print("\n=== Test (2023–2025) ===")
    print(f"Accuracy: {acc_test:.3f}")
    print(f"LogLoss:  {ll_test:.3f}")
