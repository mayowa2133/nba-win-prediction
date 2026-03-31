"""Job wrapper for historical NBA game odds backfill."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.data.oddspapi_game_odds import (
    DEFAULT_BOOKMAKERS,
    fetch_historical_game_odds_snapshots,
    get_odds_papi_api_key,
    persist_game_odds,
)
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical NBA game odds from OddsPapi.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD.")
    parser.add_argument("--api-key", default=None, help="OddsPapi API key.")
    parser.add_argument(
        "--bookmakers",
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker slugs.",
    )
    parser.add_argument("--output", default="data/game_odds_snapshots.csv", help="Snapshots CSV output.")
    parser.add_argument("--closing-output", default="data/closing_lines.csv", help="Closing lines CSV output.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    parser.add_argument("--max-fixtures", type=int, default=None, help="Optional fixture cap for one run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    snapshots = fetch_historical_game_odds_snapshots(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        api_key=get_odds_papi_api_key(args.api_key),
        bookmakers=[item.strip() for item in args.bookmakers.split(",") if item.strip()],
        max_fixtures=args.max_fixtures,
    )
    snapshot_count, closing_count = persist_game_odds(
        snapshots,
        snapshots_output_path=Path(args.output),
        closing_output_path=Path(args.closing_output),
        database_url=get_database_url(args.database_url),
    )
    print(f"[INFO] Persisted {snapshot_count} historical game odds snapshot row(s) and {closing_count} closing row(s)")


if __name__ == "__main__":
    main()
