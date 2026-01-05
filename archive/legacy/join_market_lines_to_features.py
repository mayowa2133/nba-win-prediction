#!/usr/bin/env python
"""
join_market_lines_to_features.py

Join a single market_lines_YYYY-MM-DD.csv file onto player_points_features.csv,
so you can add features like prop_pts_line / prop_over_odds_best / prop_under_odds_best
into your modeling pipeline.

Typical usage:

  python join_market_lines_to_features.py \
    --features-csv data/player_points_features.csv \
    --market-lines data/props_market/market_lines_2025-12-13.csv \
    --output data/player_points_features_with_props.csv

Notes:
  - Join key is (game_date, player_name) on the features side and
    (game_date, player) on the market side.
  - We standardize game_date to YYYY-MM-DD strings on both sides.
  - If some columns (over_odds_best, under_odds_best, etc.) are missing in the
    market file, they are simply skipped.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join market_lines CSV onto player_points_features CSV."
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default="data/player_points_features.csv",
        help="Path to player_points_features.csv (default: data/player_points_features.csv)",
    )
    parser.add_argument(
        "--market-lines",
        type=str,
        required=True,
        help="Path to a market_lines_YYYY-MM-DD.csv file (from props_to_market_lines.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/player_points_features_with_props.csv",
        help="Where to write the merged features CSV "
             "(default: data/player_points_features_with_props.csv)",
    )
    return parser.parse_args()


def standardize_game_date(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Ensure df has a 'game_date' column in YYYY-MM-DD string format.

    - If 'game_date' exists, parse to datetime and normalize.
    - Else if 'commence_time' exists (from The Odds API), derive game_date from that.
    """
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    elif "commence_time" in df.columns:
        df["game_date"] = pd.to_datetime(df["commence_time"]).dt.date.astype(str)
    else:
        raise ValueError(
            f"{source_name} is missing both 'game_date' and 'commence_time' columns; "
            "cannot align by date."
        )
    return df


def main():
    args = parse_args()

    features_path = Path(args.features_csv)
    market_path = Path(args.market_lines)
    output_path = Path(args.output)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"Market lines file not found: {market_path}")

    print(f"Loading features from: {features_path}")
    df_feat = pd.read_csv(features_path)
    print(f"  -> Loaded {len(df_feat):,} feature rows, columns={len(df_feat.columns)}")

    print(f"Loading market lines from: {market_path}")
    df_mkt = pd.read_csv(market_path)
    print(f"  -> Loaded {len(df_mkt):,} market rows, columns={len(df_mkt.columns)}")

    # Standardize date columns on both sides
    df_feat = standardize_game_date(df_feat, "features")
    df_mkt = standardize_game_date(df_mkt, "market_lines")

    # We expect 'player' in market_lines and 'player_name' in features
    if "player_name" not in df_feat.columns:
        raise ValueError("Expected 'player_name' column in features CSV.")
    if "player" not in df_mkt.columns:
        raise ValueError("Expected 'player' column in market_lines CSV.")

    # Rename market columns to avoid collisions and make intent clear
    rename_map = {
        "player": "player_name",
        "line": "prop_pts_line",
        "over_odds_best": "prop_over_odds_best",
        "under_odds_best": "prop_under_odds_best",
    }
    # Only rename columns that actually exist
    rename_map_effective = {
        src: dst for src, dst in rename_map.items() if src in df_mkt.columns
    }

    df_mkt = df_mkt.rename(columns=rename_map_effective)

    # Select columns to merge from market file
    # Always include join keys
    cols_to_keep = ["game_date", "player_name"]
    for col in [
        "prop_pts_line",
        "prop_over_odds_best",
        "prop_under_odds_best",
        "book",          # if props_to_market_lines kept a primary book
        "book_title",    # optional
        "market_key",    # e.g. player_points
        "sport_key",
        "event_id",
        "home_team",
        "away_team",
    ]:
        if col in df_mkt.columns and col not in cols_to_keep:
            cols_to_keep.append(col)

    df_mkt_small = df_mkt[cols_to_keep].drop_duplicates()

    print(f"\nWill merge the following market columns onto features:")
    print("  " + ", ".join(c for c in cols_to_keep if c not in ("game_date", "player_name")))

    # Do the merge: many feature rows per (game_date, player_name),
    # one market row per (game_date, player_name).
    print("\nMerging on (game_date, player_name) ...")
    before_cols = set(df_feat.columns)

    df_merged = df_feat.merge(
        df_mkt_small,
        on=["game_date", "player_name"],
        how="left",
        suffixes=("", "_mkt"),
    )

    # How many rows got a line?
    n_with_line = (
        df_merged["prop_pts_line"].notna().sum()
        if "prop_pts_line" in df_merged.columns
        else 0
    )
    print(f"  -> After merge: {len(df_merged):,} rows.")
    print(f"  -> Rows with non-null prop_pts_line: {n_with_line:,}")

    # Show new columns added
    new_cols = [c for c in df_merged.columns if c not in before_cols]
    print(f"\nNew columns added to features: {new_cols}")

    # Write out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"\nWrote merged features with props to: {output_path}")


if __name__ == "__main__":
    main()