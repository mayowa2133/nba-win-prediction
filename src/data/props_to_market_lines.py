#!/usr/bin/env python
"""
props_to_market_lines.py

Convert an odds slate (from fetch_props_from_the_odds_api.py) into a
training-friendly "market_lines" CSV: one row per player/game with
the best Over/Under odds for each line.
"""

import argparse
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert odds_slate.csv into market_lines.csv with one row per player/game."
    )
    parser.add_argument("--odds-slate", type=str, default="data/odds_slate.csv")
    parser.add_argument("--games-csv", type=str, default=None)
    parser.add_argument("--players-csv", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/market_lines.csv")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def strip_accents(s: str) -> str:
    """Remove diacritics so Dončić == Doncic, Jokić == Jokic, etc."""
    if not isinstance(s, str):
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def normalize_name(s: str) -> str:
    """Simple name normalizer for player/team strings."""
    if not isinstance(s, str):
        return ""
    s = strip_accents(s)
    s = s.strip().lower()
    for ch in [".", ",", "'", "\"", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def pick_best_row(sub: pd.DataFrame) -> Optional[pd.Series]:
    if sub.empty:
        return None
    idx = sub["odds"].astype(float).idxmax()
    return sub.loc[idx]


def aggregate_props(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "sport_key",
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
        "market_key",
        "player",
        "line",
        "side",
        "odds",
        "book",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Odds slate is missing required columns: {missing}")

    df = df.copy()
    df["line"] = df["line"].astype(float)
    df["odds"] = df["odds"].astype(float)

    df["commence_time_ts"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    df["game_date"] = df["commence_time_ts"].dt.date

    group_cols = [
        "sport_key",
        "event_id",
        "commence_time",
        "game_date",
        "home_team",
        "away_team",
        "market_key",
        "player",
        "line",
    ]

    records = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, sub in grouped:
        key_dict = dict(zip(group_cols, keys))

        over_rows = sub[sub["side"].str.lower() == "over"]
        under_rows = sub[sub["side"].str.lower() == "under"]

        over_best = pick_best_row(over_rows)
        under_best = pick_best_row(under_rows)
        over_odds = float(over_best["odds"]) if over_best is not None else np.nan
        under_odds = float(under_best["odds"]) if under_best is not None else np.nan
        over_book = str(over_best["book"]) if over_best is not None else ""
        under_book = str(under_best["book"]) if under_best is not None else ""

        rec = {
            **key_dict,
            # Keep legacy column name expected by scan_slate_with_model.py
            # even though this can represent points/rebounds/assists/3PM depending on market_key.
            "prop_pts_line": key_dict["line"],
            "over_odds_best": over_odds,
            "under_odds_best": under_odds,
            "best_over_book": over_book,
            "best_under_book": under_book,
            "best_over_source_provider": str(over_best.get("source_provider") or "") if over_best is not None else "",
            "best_under_source_provider": str(under_best.get("source_provider") or "") if under_best is not None else "",
            "best_over_source_mode": str(over_best.get("source_mode") or "") if over_best is not None else "",
            "best_under_source_mode": str(under_best.get("source_mode") or "") if under_best is not None else "",
            "best_over_source_page_url": str(over_best.get("source_page_url") or "") if over_best is not None else "",
            "best_under_source_page_url": str(under_best.get("source_page_url") or "") if under_best is not None else "",
            "best_over_source_book": str(over_best.get("source_book") or over_book) if over_best is not None else "",
            "best_under_source_book": str(under_best.get("source_book") or under_book) if under_best is not None else "",
            "best_over_page_snapshot_at": str(over_best.get("page_snapshot_at") or "") if over_best is not None else "",
            "best_under_page_snapshot_at": str(under_best.get("page_snapshot_at") or "") if under_best is not None else "",
        }
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    out = out.drop(columns=["line"], errors="ignore")

    front_cols = [
        "sport_key",
        "event_id",
        "commence_time",
        "game_date",
        "home_team",
        "away_team",
        "market_key",
        "player",
        "prop_pts_line",
        "over_odds_best",
        "under_odds_best",
        "best_over_book",
        "best_under_book",
    ]
    other_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + other_cols]

    return out


def maybe_attach_game_id(df: pd.DataFrame, games_csv: Path) -> pd.DataFrame:
    if not games_csv.exists():
        print(f"[GAMES] games CSV not found at {games_csv}, skipping game_id mapping.")
        return df

    print(f"[GAMES] Loading games from {games_csv} ...")
    games = pd.read_csv(games_csv)

    required_cols = ["game_id", "game_date", "home_team", "away_team"]
    missing = [c for c in required_cols if c not in games.columns]
    if missing:
        print(f"[GAMES] Missing expected columns in games.csv: {missing}. Skipping game_id mapping.")
        return df

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    games["home_team_norm"] = games["home_team"].map(normalize_name)
    games["away_team_norm"] = games["away_team"].map(normalize_name)

    df = df.copy()
    df["home_team_norm"] = df["home_team"].map(normalize_name)
    df["away_team_norm"] = df["away_team"].map(normalize_name)

    merged = df.merge(
        games[["game_id", "game_date", "home_team_norm", "away_team_norm"]],
        left_on=["game_date", "home_team_norm", "away_team_norm"],
        right_on=["game_date", "home_team_norm", "away_team_norm"],
        how="left",
    )

    matched = merged["game_id"].notna().sum()
    total = len(merged)
    print(f"[GAMES] Mapped game_id for {matched} / {total} rows "
          f"({matched / max(total, 1):.1%}).")

    merged = merged.drop(columns=["home_team_norm", "away_team_norm"])
    return merged


def maybe_attach_player_id(df: pd.DataFrame, players_csv: Path) -> pd.DataFrame:
    if not players_csv.exists():
        print(f"[PLAYERS] players CSV not found at {players_csv}, skipping player_id mapping.")
        return df

    print(f"[PLAYERS] Loading players from {players_csv} ...")
    players = pd.read_csv(players_csv)

    required_cols = ["player_id", "player_name"]
    missing = [c for c in required_cols if c not in players.columns]
    if missing:
        print(f"[PLAYERS] Missing expected columns in players.csv: {missing}. "
              f"Skipping player_id mapping.")
        return df

    players = players.copy()
    players["player_name_norm"] = players["player_name"].map(normalize_name)

    df = df.copy()
    df["player_norm"] = df["player"].map(normalize_name)

    merged = df.merge(
        players[["player_id", "player_name_norm"]],
        left_on="player_norm",
        right_on="player_name_norm",
        how="left",
    )

    matched = merged["player_id"].notna().sum()
    total = len(merged)
    print(f"[PLAYERS] Mapped player_id for {matched} / {total} rows "
          f"({matched / max(total, 1):.1%}).")

    merged = merged.drop(columns=["player_norm", "player_name_norm"])
    return merged


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    args = parse_args()

    odds_path = Path(args.odds_slate)
    output_path = Path(args.output)

    if not odds_path.exists():
        raise FileNotFoundError(f"Odds slate file not found: {odds_path}")

    print(f"Loading odds slate from {odds_path} ...")
    df_odds = pd.read_csv(odds_path)
    print(f"  -> Loaded {len(df_odds):,} rows")

    print("\nAggregating odds into one row per player/game/line ...")
    df_agg = aggregate_props(df_odds)
    print(f"  -> Aggregated to {len(df_agg):,} rows "
          f"({df_agg['player'].nunique()} unique players)")

    if args.games_csv:
        df_agg = maybe_attach_game_id(df_agg, Path(args.games_csv))

    if args.players_csv:
        df_agg = maybe_attach_player_id(df_agg, Path(args.players_csv))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_path, index=False)
    print(f"\nWrote market lines to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
