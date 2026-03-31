"""Job wrapper for building starter history from official box scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.build_starter_history import build_starter_history_frame, persist_starter_history
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build historical starter records from NBA box scores.")
    parser.add_argument("--logs-csv", default="data/player_game_logs.csv", help="Player game logs CSV.")
    parser.add_argument("--output", default="data/starter_history.csv", help="Starter history CSV output.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    parser.add_argument("--max-games", type=int, default=None, help="Optional max number of games to fetch.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs_df = pd.read_csv(args.logs_csv)
    output_path = Path(args.output)
    existing_game_ids = []
    if output_path.exists():
        existing_game_ids = pd.read_csv(output_path)["game_id"].astype(str).dropna().unique().tolist()

    frame = build_starter_history_frame(
        logs_df,
        existing_game_ids=existing_game_ids,
        max_games=args.max_games,
    )
    count = persist_starter_history(frame, output_path=output_path, database_url=get_database_url(args.database_url))
    print(f"[INFO] Persisted {count} starter history row(s)")


if __name__ == "__main__":
    main()
