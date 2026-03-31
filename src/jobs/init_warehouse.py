"""Initialize the local analytics warehouse schema."""

from __future__ import annotations

import argparse

from src.warehouse.db import get_database_url, init_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize the NBA betting beta warehouse schema.")
    parser.add_argument("--database-url", default=None, help="SQLAlchemy database URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database_url = get_database_url(args.database_url)
    init_database(database_url)
    print(f"[INFO] Warehouse schema ready at {database_url}")


if __name__ == "__main__":
    main()
