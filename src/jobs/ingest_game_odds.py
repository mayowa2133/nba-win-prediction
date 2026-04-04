"""Job wrapper for current-day NBA game odds ingestion from The Odds API."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.data.oddspapi_game_odds import persist_game_odds
from src.data.the_odds_api_game_odds import (
    DEFAULT_BOOKMAKERS,
    fetch_current_game_odds_snapshots,
    get_the_odds_api_key,
)
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch current-day NBA game odds from The Odds API.")
    parser.add_argument("--report-date", default=date.today().isoformat(), help="Report date in YYYY-MM-DD.")
    parser.add_argument("--api-key", default=None, help="The Odds API key.")
    parser.add_argument(
        "--bookmakers",
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker slugs.",
    )
    parser.add_argument("--output", default="data/game_odds_snapshots.csv", help="Snapshots CSV output.")
    parser.add_argument("--closing-output", default="data/closing_lines.csv", help="Closing lines CSV output.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    snapshots = fetch_current_game_odds_snapshots(
        report_date=date.fromisoformat(args.report_date),
        api_key=get_the_odds_api_key(args.api_key),
        bookmakers=[item.strip() for item in args.bookmakers.split(",") if item.strip()],
    )
    snapshot_count, closing_count = persist_game_odds(
        snapshots,
        snapshots_output_path=Path(args.output),
        closing_output_path=Path(args.closing_output),
        database_url=get_database_url(args.database_url),
    )
    print(f"[INFO] Persisted {snapshot_count} game odds snapshot row(s) and {closing_count} closing row(s)")


if __name__ == "__main__":
    main()
