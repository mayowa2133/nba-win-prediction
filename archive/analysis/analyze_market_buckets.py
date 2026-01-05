# analyze_market_buckets.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

DATA_PATH = Path("games_all_2015_2025_features_rolling_last10_prevstats_odds.csv")


def bucket(row):
    p = row["market_home_prob"]
    if p < 0.55:
        return "[0.50,0.55)"
    elif p < 0.60:
        return "[0.55,0.60)"
    elif p < 0.65:
        return "[0.60,0.65)"
    elif p < 0.70:
        return "[0.65,0.70)"
    elif p < 0.75:
        return "[0.70,0.75)"
    elif p < 0.80:
        return "[0.75,0.80)"
    else:
        return "[0.80,1.00]"


def main():
    df = pd.read_csv(DATA_PATH)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Only games with odds and seasons <= 2022
    df = df[(df["season"] <= 2022) & (~df["market_home_prob"].isna())].copy()

    print("Using seasons with odds (<= 2022). Total games:", len(df))

    # Overall market-only accuracy
    probs = df["market_home_prob"].values
    preds = (probs >= 0.5).astype(int)
    y = df["home_win"].values

    overall_acc = accuracy_score(y, preds)
    print(f"\nOverall market-only accuracy: {overall_acc:.3f}")

    # Bucket by confidence
    df["bucket"] = df.apply(bucket, axis=1)

    print("\nBucket-wise performance (all seasons with odds):")
    print("Bucket         Games   AvgProb   ActualWin%")
    print("-------------------------------------------")

    for b, group in df.groupby("bucket"):
        g = len(group)
        avg_p = group["market_home_prob"].mean()
        win_rate = group["home_win"].mean()
        print(f"{b:12} {g:5d}   {avg_p:7.3f}     {win_rate:7.3f}")

    # Same but just for 2022
    df_2022 = df[df["season"] == 2022].copy()
    print("\nBucket-wise performance (2022 only):")
    print("Bucket         Games   AvgProb   ActualWin%")
    print("-------------------------------------------")
    for b, group in df_2022.groupby("bucket"):
        g = len(group)
        avg_p = group["market_home_prob"].mean()
        win_rate = group["home_win"].mean()
        print(f"{b:12} {g:5d}   {avg_p:7.3f}     {win_rate:7.3f}")


if __name__ == "__main__":
    main()
