#!/usr/bin/env python
"""
build_player_features_rolling.py

Build rolling features for player points props from player_game_logs.csv.

- Input:  data/player_game_logs.csv
- Output: data/player_points_features.csv

Each row = one player-game with:
  - ID / meta columns
  - rolling stats (last N games)
  - target: pts
"""

import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")
PLAYER_LOGS_CSV = DATA_DIR / "player_game_logs.csv"
OUT_FEATURES_CSV = DATA_DIR / "player_points_features.csv"


def load_player_logs() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_LOGS_CSV)

    # Ensure some basic columns exist
    required_cols = [
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_team_abbrev",
        "is_home",
        "pts",
        "minutes",
        "fga",
        "fg3a",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in player_game_logs.csv: {missing}")

    # Sort so rolling windows make sense
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """
    For each player, compute rolling means/std for key stats over the last N games.
    We exclude the current row from the window (shift by 1).
    """
    df = df.copy()

    stats_cols = ["pts", "minutes", "fga", "fg3a", "reb", "ast", "stl", "blk", "tov"]

    grouped = df.groupby("player_id", group_keys=False)

    for w in windows:
        for col in stats_cols:
            mean_col = f"{col}_rolling_mean_{w}"
            std_col = f"{col}_rolling_std_{w}"

            df[mean_col] = grouped[col].apply(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            df[std_col] = grouped[col].apply(
                lambda s: s.shift(1).rolling(w, min_periods=1).std()
            )

    # Example of simple “form” feature: ratio of last5 mean pts vs season mean so far
    # (You can expand this later.)
    df["season_game_number"] = (
        df.groupby(["season", "player_id"]).cumcount() + 1
    )

    return df


def maybe_join_team_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIONAL:
      If you want to merge team-level game features (pace, elo, rest, etc.)
      from your existing games_all_....csv, you can do it here.

    For now this just returns df unchanged.
    """
    # Example sketch if you want to hook it up later:
    # games_df = pd.read_csv("games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv")
    # games_df = games_df[["game_id", "home_team_abbrev", "away_team_abbrev", "pace", "elo_home_pre", ...]]
    # For each player row, merge on game_id + whether they are home or away.
    return df


def build_features():
    df = load_player_logs()
    df = add_rolling_features(df, windows=(5, 10))
    df = maybe_join_team_context(df)

    # Drop rows where we have no history (optional)
    # e.g., require at least 3 games of history
    df["games_played_so_far"] = (
        df.groupby("player_id").cumcount()
    )
    df = df[df["games_played_so_far"] >= 3].reset_index(drop=True)

    # Save only the columns we care about (ID/meta + features + target)
    id_cols = [
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_team_abbrev",
        "is_home",
    ]
    target_col = ["pts"]

    feature_cols = [c for c in df.columns if c not in id_cols + target_col]

    out_cols = id_cols + feature_cols + target_col

    df[out_cols].to_csv(OUT_FEATURES_CSV, index=False)
    print(f"Saved {len(df)} rows with player rolling features -> {OUT_FEATURES_CSV}")


if __name__ == "__main__":
    build_features()
