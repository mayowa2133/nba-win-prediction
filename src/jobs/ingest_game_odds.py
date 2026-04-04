"""Job wrapper for current-day NBA game odds ingestion from selected live sources."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.data.oddspapi_game_odds import persist_game_odds
from src.data.public_page_game_odds import fetch_espn_game_frames, fetch_scoresandodds_game_frames
from src.data.sportsgameodds import fetch_sportsgameodds_game_frames, get_sportsgameodds_api_key
from src.data.the_odds_api_game_odds import (
    DEFAULT_BOOKMAKERS,
    fetch_current_game_odds_snapshots,
    get_the_odds_api_key,
)
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch current-day NBA game odds from a selected live source.")
    parser.add_argument("--report-date", default=date.today().isoformat(), help="Report date in YYYY-MM-DD.")
    parser.add_argument("--source", choices=["scoresandodds", "espn", "sportsgameodds", "the-odds-api"], default="scoresandodds")
    parser.add_argument("--api-key", default=None, help="The Odds API key.")
    parser.add_argument("--sportsgameodds-api-key", default=None, help="SportsGameOdds API key.")
    parser.add_argument(
        "--bookmakers",
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker slugs.",
    )
    parser.add_argument("--output", default="data/game_odds_snapshots.csv", help="Snapshots CSV output.")
    parser.add_argument("--closing-output", default="data/closing_lines.csv", help="Closing lines CSV output.")
    parser.add_argument("--game-lines-output", default="data/game_lines.csv", help="Current game-lines CSV output.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_date = date.fromisoformat(args.report_date)
    game_lines = None
    if args.source == "scoresandodds":
        snapshots, game_lines = fetch_scoresandodds_game_frames(report_date=report_date)
    elif args.source == "espn":
        snapshots, game_lines = fetch_espn_game_frames(report_date=report_date)
    elif args.source == "sportsgameodds":
        snapshots, game_lines = fetch_sportsgameodds_game_frames(
            report_date=report_date,
            api_key=get_sportsgameodds_api_key(args.sportsgameodds_api_key),
            bookmakers=[item.strip() for item in args.bookmakers.split(",") if item.strip()],
        )
    else:
        snapshots = fetch_current_game_odds_snapshots(
            report_date=report_date,
            api_key=get_the_odds_api_key(args.api_key),
            bookmakers=[item.strip() for item in args.bookmakers.split(",") if item.strip()],
        )
    snapshot_count, closing_count = persist_game_odds(
        snapshots,
        snapshots_output_path=Path(args.output),
        closing_output_path=Path(args.closing_output),
        database_url=get_database_url(args.database_url),
    )
    if game_lines is not None and not game_lines.empty:
        output_path = Path(args.game_lines_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        game_lines.to_csv(output_path, index=False)
        print(f"[INFO] Wrote {len(game_lines)} current game-line row(s) to {output_path}")
    print(f"[INFO] Persisted {snapshot_count} game odds snapshot row(s) and {closing_count} closing row(s) from {args.source}")


if __name__ == "__main__":
    main()
