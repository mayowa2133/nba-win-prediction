# evaluate_market_baseline.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Use only games where we have market_home_prob and seasons <= 2022
    df = df[(df["season"] <= 2022) & (~df["market_home_prob"].isna())].copy()

    print("Games with odds by season:")
    print(df["season"].value_counts().sort_index())

    # Market-only probabilities
    probs = df["market_home_prob"].values
    preds = (probs >= 0.5).astype(int)
    y = df["home_win"].values

    def eval_split(name, mask):
        y_split = y[mask]
        p_split = probs[mask]
        pred_split = preds[mask]

        acc = accuracy_score(y_split, pred_split)
        ll = log_loss(y_split, p_split)

        print(f"\n=== {name} ===")
        print(f"Games:    {len(y_split)}")
        print(f"Accuracy: {acc:.3f}")
        print(f"LogLoss:  {ll:.3f}")

    # Overall (all seasons with odds)
    mask_all = np.ones(len(df), dtype=bool)
    eval_split("All seasons (market only)", mask_all)

    # Test season (2022)
    mask_2022 = df["season"].values == 2022
    eval_split("2022 (market only)", mask_2022)


if __name__ == "__main__":
    main()
