#!/usr/bin/env python
"""
derive_vegas_from_scores.py

Derives approximate Vegas-style lines from actual NBA game scores.

The idea: Real Vegas lines predict final margins and totals with high accuracy.
By using actual game scores as the "target", we can create training features that
approximate what Vegas lines would have been.

This adds signal because:
1. Game totals correlate with pace and team offensive/defensive strength
2. Spreads reflect the gap between teams

We derive:
- game_total: home_pts + away_pts (actual total scored)
- implied_spread: home_pts - away_pts (actual margin)

These are then normalized and used as features representing the "true" game environment.

Usage:
    python derive_vegas_from_scores.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_CSV = Path("data/player_points_features_with_lineup.csv")
OUTPUT_CSV = Path("data/player_points_features_with_vegas.csv")
GAME_LOGS_CSV = Path("data/player_game_logs.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive Vegas-like features from game scores")
    parser.add_argument("--features-csv", type=str, default=str(FEATURES_CSV))
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV))
    parser.add_argument("--game-logs-csv", type=str, default=str(GAME_LOGS_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    features_path = Path(args.features_csv)
    output_path = Path(args.output_csv)
    game_logs_path = Path(args.game_logs_csv)

    print("=" * 70)
    print("DERIVE VEGAS-LIKE FEATURES FROM GAME SCORES")
    print("=" * 70)

    if not features_path.exists():
        print(f"[ERROR] Features file not found: {features_path}")
        return

    print(f"[INFO] Loading features from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows")

    # First, try to get game-level totals from game logs
    if game_logs_path.exists():
        print(f"[INFO] Loading game logs from {game_logs_path}...")
        logs = pd.read_csv(game_logs_path, low_memory=False)
        
        # Calculate game totals: sum of points for both teams in each game
        # Each game appears twice in logs (once per team)
        logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.strftime("%Y-%m-%d")
        
        # Group by game to get total points
        game_totals = logs.groupby(["game_id", "game_date"]).agg({
            "pts": "sum",  # Total points in game (both teams)
        }).reset_index()
        game_totals = game_totals.rename(columns={"pts": "game_total_pts"})
        
        # Get margin (home - away)
        # Need to identify home vs away
        home_pts = logs[logs["home_away"] == "HOME"].groupby(["game_id", "game_date"])["pts"].sum().reset_index()
        home_pts = home_pts.rename(columns={"pts": "home_pts"})
        
        away_pts = logs[logs["home_away"] == "AWAY"].groupby(["game_id", "game_date"])["pts"].sum().reset_index()
        away_pts = away_pts.rename(columns={"pts": "away_pts"})
        
        game_margins = home_pts.merge(away_pts, on=["game_id", "game_date"], how="inner")
        game_margins["game_margin"] = game_margins["home_pts"] - game_margins["away_pts"]
        
        # Merge totals and margins
        game_data = game_totals.merge(
            game_margins[["game_id", "game_date", "game_margin", "home_pts", "away_pts"]],
            on=["game_id", "game_date"],
            how="inner"
        )
        
        print(f"[INFO] Computed game data for {len(game_data):,} games")
        print(f"[INFO] Game total stats: mean={game_data['game_total_pts'].mean():.1f}, std={game_data['game_total_pts'].std():.1f}")
        print(f"[INFO] Game margin stats: mean={game_data['game_margin'].mean():.1f}, std={game_data['game_margin'].std():.1f}")
        
    else:
        print(f"[WARN] Game logs not found at {game_logs_path}")
        game_data = None

    # Ensure game_date format matches
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

    # If we have game_id in features, merge directly
    if game_data is not None and "game_id" in df.columns:
        print("[INFO] Merging game-level data via game_id...")
        df = df.merge(
            game_data[["game_id", "game_total_pts", "game_margin", "home_pts", "away_pts"]],
            on="game_id",
            how="left"
        )
        
        # For Vegas-like features, we want PRIOR information (what would've been predicted)
        # Using actual scores gives perfect hindsight, which causes leakage
        # Instead, we'll use rolling team averages to approximate pre-game expectations
        
        print("[INFO] Creating rolling approximations (to avoid leakage)...")
        
        # Calculate team-level rolling stats for game totals
        # This approximates what Vegas would have estimated
        team_game_totals = game_data.groupby("game_date")["game_total_pts"].mean()
        
        # For each player row, we need the EXPECTED game environment, not actual
        # Use the player's team's rolling offensive output + opponent's defensive allowance
        
    else:
        print("[WARN] Cannot merge game data - will derive from team features")

    # -------------------------------------------------------------------------
    # Create Vegas-like features from existing rolling stats (no leakage)
    # -------------------------------------------------------------------------
    print("[INFO] Deriving Vegas-like features from rolling team stats...")
    
    # Approximate game total from team pace and opponent pace
    # Pace ~ possessions ~ points opportunity
    if "team_pace_roll15" in df.columns:
        # Estimate expected team points from pace and margin
        team_off = df.get("team_margin_roll15", 0) / 2 + 105  # Rough estimate: 105 is average team score
        opp_off = 105 - df.get("team_margin_roll15", 0) / 2  # Opponent's expected score
        df["vegas_game_total"] = team_off + opp_off
        
        # Adjust by pace factor
        pace_factor = df["team_pace_roll15"] / 100.0  # Normalize around 1.0
        df["vegas_game_total"] = df["vegas_game_total"] * pace_factor
        
        # Clamp to realistic range
        df["vegas_game_total"] = df["vegas_game_total"].clip(190, 260)
    else:
        df["vegas_game_total"] = 220.0  # Default

    # Approximate spread from team margin differential
    if "team_margin_roll15" in df.columns:
        # Spread = expected margin, negative means home favorite
        # Add home court advantage (~3 points)
        home_adv = df["is_home"].fillna(0) * 3.0
        df["vegas_spread"] = -(df["team_margin_roll15"].fillna(0) + home_adv)
        
        # Clamp to realistic range
        df["vegas_spread"] = df["vegas_spread"].clip(-20, 20)
    else:
        df["vegas_spread"] = 0.0

    df["vegas_abs_spread"] = df["vegas_spread"].abs()

    # Game script features
    abs_spread = df["vegas_abs_spread"]
    df["blowout_prob"] = 1.0 / (1.0 + np.exp(-0.15 * (abs_spread - 10)))
    df["is_likely_blowout"] = (abs_spread >= 10).astype(int)
    df["garbage_time_minutes_est"] = (abs_spread / 2.0).clip(upper=10.0)
    df["vegas_spread_abs_normalized"] = abs_spread / 15.0

    # Drop any intermediate columns
    drop_cols = ["game_total_pts", "game_margin", "home_pts", "away_pts"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[INFO] Saved to {output_path}")
    
    # Stats
    print("\n=== DERIVED VEGAS FEATURE STATS ===")
    for col in ["vegas_game_total", "vegas_spread", "vegas_abs_spread", "blowout_prob"]:
        if col in df.columns:
            vals = df[col]
            print(f"{col:30s}: mean={vals.mean():.2f}, std={vals.std():.2f}")

    # Coverage
    coverage = (df["vegas_game_total"] > 0).sum() / len(df)
    print(f"\n[INFO] Vegas feature coverage: {coverage:.1%}")


if __name__ == "__main__":
    main()

