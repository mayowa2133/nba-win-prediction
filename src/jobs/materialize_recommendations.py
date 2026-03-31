"""Persist precomputed recommendation CSV output into the warehouse."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.warehouse.db import get_database_url
from src.warehouse.materialize import materialize_edges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize scored recommendation artifacts into the warehouse.")
    parser.add_argument(
        "--edges-path",
        default="data/edges_with_market.csv",
        help="Path to a scored edges CSV produced by scan_slate_with_model.py.",
    )
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    parser.add_argument(
        "--recommendation-origin",
        default="live_daily",
        help="Origin tag to persist on recommendation rows (default: live_daily).",
    )
    parser.add_argument(
        "--readiness-path",
        default=None,
        help="Optional CSV with market readiness overrides.",
    )
    parser.add_argument(
        "--skip-readiness",
        action="store_true",
        help="Skip market-readiness snapshot persistence when materializing recommendations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database_url = get_database_url(args.database_url)
    scored_count, readiness_count = materialize_edges(
        Path(args.edges_path),
        database_url=database_url,
        readiness_path=Path(args.readiness_path) if args.readiness_path else None,
        recommendation_origin=str(args.recommendation_origin),
        persist_readiness=not args.skip_readiness,
    )
    print(
        f"[INFO] Materialized {scored_count} recommendation row(s) and "
        f"{readiness_count} market-readiness snapshot row(s) into {database_url}"
    )


if __name__ == "__main__":
    main()
