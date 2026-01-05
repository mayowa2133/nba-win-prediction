#!/usr/bin/env python
"""
predict_points_over_line.py

Use the regression model to:
  - Predict a player's expected points for their latest game in the dataset
  - Approximate P(points > LINE) using a normal distribution
  - Convert that into fair (no-vig) moneyline odds

Usage:
  python predict_points_over_line.py --player "Luka Doncic" --line 27.5 --season-min 2023
"""

import argparse
import math
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

DATA_CSV = Path("data/player_points_features.csv")
MODEL_PATH = Path("models/points_regression.pkl")


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Strip accents and lowercase: "Luka Dončić" -> "luka doncic"
    s_norm = unicodedata.normalize("NFKD", s)
    s_ascii = s_norm.encode("ascii", "ignore").decode("ascii")
    return s_ascii.strip().lower()


def load_latest_player_row(player_query: str, season_min: int) -> pd.Series:
    print(f"Loading features from {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)

    if df.empty:
        raise SystemExit("Feature file is empty.")

    df["name_norm"] = df["player_name"].fillna("").apply(normalize_name)
    target_norm = normalize_name(player_query)

    # Filter by season first (to avoid matching retired guys from 2010, etc.)
    df_recent = df[df["season"] >= season_min].copy()
    if df_recent.empty:
        raise SystemExit(f"No rows found with season >= {season_min}.")

    matches = df_recent[df_recent["name_norm"].str.contains(target_norm)]
    if matches.empty:
        # Fallback: search across all seasons
        matches = df[df["name_norm"].str.contains(target_norm)]
        if matches.empty:
            raise SystemExit(f"No rows found for player '{player_query}'. Check spelling.")

    full_names = matches["player_name"].dropna().unique()
    if len(full_names) > 1:
        print("Multiple matching names found:")
        for n in full_names:
            print(f"  - {n}")
        raise SystemExit(
            "Name is ambiguous. Please use a more specific name (e.g., 'Luka Dončić')."
        )

    player_name = full_names[0]
    player_rows = df[df["player_name"] == player_name]

    # Pick the latest game by game_date + season
    player_rows = player_rows.sort_values(["season", "game_date"])
    latest = player_rows.iloc[-1]

    print("\nUsing latest game row for {}:".format(player_name))
    home_away = "(HOME)" if latest["is_home"] == 1 else "(AWAY)"
    print(f"  Season:   {latest['season']}")
    print(f"  Game ID:  {latest['game_id']}")
    print(f"  Date:     {latest['game_date']}")
    print(f"  Team:     {latest['team_abbrev']} vs {latest['opp_abbrev']} {home_away}")

    return latest


def prob_over_normal(mu: float, sigma: float, line: float):
    """Approximate P(points > line) assuming Normal(mu, sigma^2)."""
    if sigma <= 0:
        # Degenerate fallback: if sigma is 0, then always at mu
        return 1.0 if mu > line else 0.0

    z = (line - mu) / sigma
    # Standard normal CDF using erf
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_over = 1.0 - cdf
    return max(0.0, min(1.0, p_over))


def fair_moneyline(p: float) -> str:
    """
    Convert probability to American moneyline odds (no vig).
    p in (0,1)
    """
    p = max(1e-6, min(1 - 1e-6, p))  # avoid 0 or 1 exactly

    if p == 0.5:
        return "+100"
    if p > 0.5:
        # negative odds
        odds = -100 * p / (1 - p)
    else:
        # positive odds
        odds = 100 * (1 - p) / p

    if odds > 0:
        return f"+{odds:.0f}"
    else:
        return f"{odds:.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True, help="Player name (e.g., 'Luka Doncic')")
    parser.add_argument("--line", type=float, required=True, help="Points line (e.g., 27.5)")
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season year to consider (default: 2023)",
    )

    args = parser.parse_args()

    latest = load_latest_player_row(args.player, args.season_min)

    if not MODEL_PATH.exists():
        raise SystemExit(f"Regression model file not found at {MODEL_PATH}. "
                         f"Run build_points_model.py first.")

    print(f"\nLoading regression model from {MODEL_PATH} ...")
    bundle = joblib.load(MODEL_PATH)

    # Backwards compatible: handle both bare model and dict bundle
    if isinstance(bundle, dict):
        model = bundle.get("model", None)
        feature_cols = bundle.get("features", None)
        sigma = float(bundle.get("sigma", 5.0))
        if model is None or feature_cols is None:
            raise SystemExit("Model bundle is missing 'model' or 'features' keys.")
    else:
        # If someone saved just the raw model
        model = bundle
        # You could hard-code feature_cols here, but better to pass via dict
        raise SystemExit(
            "Regression model is not a bundle. Please re-run build_points_model.py "
            "so models/points_regression.pkl includes feature metadata."
        )

    # Build the feature vector for this game
    X = latest[feature_cols].to_frame().T

    # Predict expected points
    mu = float(model.predict(X)[0])

    # Approximate probability of going over the line
    p_over = prob_over_normal(mu, sigma, args.line)
    p_under = 1.0 - p_over

    over_ml = fair_moneyline(p_over)
    under_ml = fair_moneyline(p_under)

    print("\n=== Regression-based prediction for points OVER {:.1f} ===".format(args.line))
    print(f"Expected points (mu): {mu:.2f}")
    print(f"P(UNDER {args.line:.1f}): {p_under:.3f}")
    print(f"P(OVER  {args.line:.1f}): {p_over:.3f}\n")

    print("Fair odds (no vig):")
    print(f"  OVER  {args.line:.1f} -> {over_ml}")
    print(f"  UNDER {args.line:.1f} -> {under_ml}")

    print("\nNotes:")
    print("  - This uses a simple normal approximation around the model's mean prediction.")
    print("  - Next steps would be to calibrate this more carefully via backtesting,")
    print("    but it gives you a first-pass P(OVER line) for any line, not just 15.5.")


if __name__ == "__main__":
    main()
