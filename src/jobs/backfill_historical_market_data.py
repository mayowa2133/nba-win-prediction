"""Backfill local logs and synthetic historical snapshots from canonical historical odds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.historical_game_odds import (
    DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV,
    backfill_player_logs,
    build_historical_snapshot_frame,
    export_game_lines_history,
)
from src.data.oddspapi_game_odds import persist_game_odds
from src.warehouse.db import get_database_url


PLAYER_LOGS_CSV = Path("data/player_game_logs.csv")
GAME_ODDS_SNAPSHOTS_CSV = Path("data/game_odds_snapshots.csv")
CLOSING_LINES_CSV = Path("data/closing_lines.csv")
GAME_LINES_HISTORY_DIR = Path("data/game_lines_history")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill logs and synthetic historical odds artifacts from canonical historical odds.")
    parser.add_argument("--canonical-odds-csv", default=str(DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV))
    parser.add_argument("--logs-csv", default=str(PLAYER_LOGS_CSV))
    parser.add_argument("--output-logs-csv", default=str(PLAYER_LOGS_CSV))
    parser.add_argument("--snapshots-output", default=str(GAME_ODDS_SNAPSHOTS_CSV))
    parser.add_argument("--closing-output", default=str(CLOSING_LINES_CSV))
    parser.add_argument("--game-lines-dir", default=str(GAME_LINES_HISTORY_DIR))
    parser.add_argument("--database-url", default=None)
    return parser


def run_backfill(args: argparse.Namespace) -> dict:
    canonical_df = pd.read_csv(args.canonical_odds_csv) if Path(args.canonical_odds_csv).exists() else pd.DataFrame()
    if canonical_df.empty:
        raise RuntimeError(f"No canonical historical odds rows found at {args.canonical_odds_csv}")

    logs_df = pd.read_csv(args.logs_csv)
    backfilled_logs, coverage = backfill_player_logs(logs_df, canonical_df)
    Path(args.output_logs_csv).parent.mkdir(parents=True, exist_ok=True)
    backfilled_logs.to_csv(args.output_logs_csv, index=False)

    snapshot_df = build_historical_snapshot_frame(canonical_df)
    snapshot_count, closing_count = persist_game_odds(
        snapshot_df,
        snapshots_output_path=Path(args.snapshots_output),
        closing_output_path=Path(args.closing_output),
        database_url=get_database_url(args.database_url),
    )
    game_lines_count = export_game_lines_history(canonical_df, output_dir=Path(args.game_lines_dir))

    return {
        "spread_coverage_rate": coverage["spread_coverage_rate"],
        "total_coverage_rate": coverage["total_coverage_rate"],
        "moneyline_coverage_rate": coverage["moneyline_coverage_rate"],
        "snapshot_count": snapshot_count,
        "closing_count": closing_count,
        "game_lines_count": game_lines_count,
        "output_logs_csv": str(args.output_logs_csv),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_backfill(args)
    print(
        f"[INFO] Backfilled logs to {result['output_logs_csv']}; "
        f"spread_coverage={result['spread_coverage_rate']:.4f}, "
        f"total_coverage={result['total_coverage_rate']:.4f}, "
        f"moneyline_coverage={result['moneyline_coverage_rate']:.4f}"
    )
    print(
        f"[INFO] Persisted {result['snapshot_count']} synthetic historical snapshot row(s), "
        f"{result['closing_count']} closing row(s), and {result['game_lines_count']} dated game-line row(s)"
    )


if __name__ == "__main__":
    main()
