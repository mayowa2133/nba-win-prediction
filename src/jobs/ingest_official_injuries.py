"""Job wrapper for fetching official NBA injury reports."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.data.official_injuries import fetch_official_injury_reports, persist_official_injury_reports
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch official NBA injury reports and persist them.")
    parser.add_argument("--report-date", default=date.today().isoformat(), help="Report date in YYYY-MM-DD.")
    parser.add_argument("--all-snapshots", action="store_true", help="Persist every report PDF for the date.")
    parser.add_argument("--output", default="data/official_injuries.csv", help="Output CSV path.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_df = fetch_official_injury_reports(
        report_date=date.fromisoformat(args.report_date),
        latest_only=not args.all_snapshots,
    )
    count = persist_official_injury_reports(
        report_df,
        output_path=Path(args.output),
        database_url=get_database_url(args.database_url),
    )
    print(f"[INFO] Persisted {count} official injury report row(s)")


if __name__ == "__main__":
    main()
