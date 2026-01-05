#!/usr/bin/env python
"""
predict_points_over_15_5.py

Usage:
    python predict_points_over_15_5.py --player "LeBron James" --season-min 2023
    python predict_points_over_15_5.py --player "Luka Doncic" --season-min 2023
"""

import argparse
import unicodedata
import re
from pathlib import Path

import joblib
import pandas as pd


DATA_CSV = Path("data/player_points_features.csv")
MODEL_PATH = Path("models/points_over_15_5.pkl")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalize player names so that:
      - accents are removed (Doncic == Dončić)
      - everything is lowercase
      - punctuation is stripped
    """
    if not isinstance(name, str):
        return ""
    # Strip accents
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    # Lowercase
    s = no_accents.lower()
    # Keep only letters, numbers, and spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # Collapse multiple spaces
    s = " ".join(s.split())
    return s


def prob_to_american(p: float):
    """
    Convert probability to American moneyline odds.
    Returns None if p is 0 or 1.
    """
    if p <= 0 or p >= 1:
        return None
    # Favorite (negative odds)
    if p > 0.5:
        return int(round(-100 * p / (1 - p)))
    # Underdog (positive odds)
    else:
        return int(round(100 * (1 - p) / p))


# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player",
        required=True,
        help="Player name, e.g. 'LeBron James' or 'Luka Doncic'",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season (e.g., 2023) to consider when picking the latest game",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DATA_CSV),
        help="Path to player_points_features.csv",
    )
    parser.add_argument(
        "--model-path",
        default=str(MODEL_PATH),
        help="Path to points_over_15_5.pkl",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    model_path = Path(args.model_path)

    # ----------------- Load data -----------------
    if not csv_path.exists():
        raise SystemExit(f"Features CSV not found at {csv_path}")

    print(f"Loading features from {csv_path} ...")
    df = pd.read_csv(csv_path)

    if df.empty:
        raise SystemExit("Feature CSV is empty. Build features first.")

    # Build normalized-name column
    df["name_norm"] = df["player_name"].fillna("").map(normalize_name)

    user_norm = normalize_name(args.player)
    if not user_norm:
        raise SystemExit("Could not normalize player name from input.")

    # Filter by season and normalized name
    season_min = args.season_min
    mask = (df["season"] >= season_min) & (df["name_norm"] == user_norm)
    sub = df.loc[mask].copy()

    if sub.empty:
        print(f"No rows found for player '{args.player}' (normalized as '{user_norm}').")

        # Optional: show some close matches to help debug spelling
        all_unique = sorted(df["player_name"].dropna().unique())
        candidates = [n for n in all_unique if normalize_name(n) == user_norm]
        if candidates:
            print("\nHowever, found these exact normalized matches in the data:")
            for c in candidates:
                print("  -", c)
        else:
            print("\nTip: try searching directly in the CSV or using a shorter substring.")
        return

    # Pick the latest game row (by game_date, then game_id)
    sub["game_date"] = pd.to_datetime(sub["game_date"])
    sub = sub.sort_values(["game_date", "game_id"])
    latest = sub.iloc[-1]

    # Pretty-print which game we're using
    home_away = "HOME" if latest["is_home"] == 1 else "AWAY"
    print("\nUsing latest game row for {}:".format(latest["player_name"]))
    print("  Season:   {}".format(int(latest["season"])))
    print("  Game ID:  {}".format(latest["game_id"]))
    print("  Date:     {}".format(str(latest["game_date"].date())))
    print("  Team:     {} vs {} ({})".format(
        latest["team_abbrev"],
        latest["opp_abbrev"],
        home_away,
    ))

    # ----------------- Load model -----------------
    if not model_path.exists():
        raise SystemExit(f"Model file not found at {model_path}")

    print(f"\nLoading model from {model_path} ...")
    model_dict = joblib.load(model_path)

    model = model_dict["model"]
    feature_cols = model_dict["feature_cols"]
    threshold = model_dict.get("threshold", 0.5)

    # Build feature vector
    X = latest[feature_cols].to_frame().T

    # ----------------- Predict -----------------
    probs = model.predict_proba(X)[0]
    # Assuming class 1 = "OVER 15.5"
    p_under = float(probs[0])
    p_over = float(probs[1])

    over_odds = prob_to_american(p_over)
    under_odds = prob_to_american(p_under)

    print("\n=== Prediction for points OVER 15.5 ===")
    print(f"P(UNDER 15.5): {p_under:.3f}")
    print(f"P(OVER 15.5):  {p_over:.3f}")
    print("\nFair odds (no vig):")
    if over_odds is not None:
        print(f"  OVER 15.5  -> {over_odds:+d}")
    else:
        print("  OVER 15.5  -> (degenerate prob)")

    if under_odds is not None:
        print(f"  UNDER 15.5 -> {under_odds:+d}")
    else:
        print("  UNDER 15.5 -> (degenerate prob)")

    print("\nNotes:")
    print(f"  - Model default threshold during training was {threshold:.2f}")
    print("  - If a book is offering OVER at a better price than these fair odds,")
    print("    that’s a candidate +EV spot to investigate further.")


if __name__ == "__main__":
    main()
