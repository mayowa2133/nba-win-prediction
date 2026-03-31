"""Job wrapper for deterministic projected starting lineups."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.build_lineup_projections import build_lineup_projection_frame, persist_lineup_projections
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build projected NBA starting lineups.")
    parser.add_argument("--target-date", default=date.today().isoformat(), help="Target slate date in YYYY-MM-DD.")
    parser.add_argument("--starter-history-csv", default="data/starter_history.csv", help="Starter history CSV.")
    parser.add_argument("--logs-csv", default="data/player_game_logs.csv", help="Player game logs CSV.")
    parser.add_argument("--injuries-csv", default="data/official_injuries.csv", help="Official injuries CSV.")
    parser.add_argument("--player-positions-csv", default="data/player_positions.csv", help="Player positions CSV.")
    parser.add_argument("--consensus-csv", default=None, help="Optional consensus lineup CSV for QA.")
    parser.add_argument("--output", default="data/lineup_projections.csv", help="Lineup projection CSV output.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    starter_history_df = pd.read_csv(args.starter_history_csv) if Path(args.starter_history_csv).exists() else pd.DataFrame()
    logs_df = pd.read_csv(args.logs_csv)
    injuries_df = pd.read_csv(args.injuries_csv) if Path(args.injuries_csv).exists() else pd.DataFrame()
    player_positions_df = pd.read_csv(args.player_positions_csv) if Path(args.player_positions_csv).exists() else pd.DataFrame()
    consensus_df = pd.read_csv(args.consensus_csv) if args.consensus_csv else pd.DataFrame()

    frame = build_lineup_projection_frame(
        target_date=date.fromisoformat(args.target_date),
        starter_history_df=starter_history_df,
        logs_df=logs_df,
        injuries_df=injuries_df,
        player_positions_df=player_positions_df,
        consensus_df=consensus_df,
    )
    count = persist_lineup_projections(frame, output_path=Path(args.output), database_url=get_database_url(args.database_url))
    print(f"[INFO] Persisted {count} lineup projection row(s)")


if __name__ == "__main__":
    main()
