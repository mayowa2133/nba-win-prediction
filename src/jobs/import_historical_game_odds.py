"""Import and reconcile local historical NBA game odds sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.historical_game_odds import (
    DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV,
    DEFAULT_HISTORICAL_ODDS_CONFLICTS_CSV,
    DEFAULT_HISTORICAL_ODDS_MANIFEST,
    import_historical_odds_sources,
    persist_historical_odds,
    reconcile_historical_odds,
    write_historical_odds_artifacts,
)
from src.warehouse.db import get_database_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import and reconcile local historical NBA odds sources.")
    parser.add_argument("--manifest", default=str(DEFAULT_HISTORICAL_ODDS_MANIFEST))
    parser.add_argument("--canonical-output", default=str(DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV))
    parser.add_argument("--conflicts-output", default=str(DEFAULT_HISTORICAL_ODDS_CONFLICTS_CSV))
    parser.add_argument("--database-url", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_rows = import_historical_odds_sources(manifest_path=Path(args.manifest))
    canonical_df, conflicts_df = reconcile_historical_odds(source_rows)
    write_historical_odds_artifacts(
        canonical_df,
        conflicts_df,
        canonical_output_path=Path(args.canonical_output),
        conflicts_output_path=Path(args.conflicts_output),
    )
    odds_count, conflict_count = persist_historical_odds(
        canonical_df,
        conflicts_df,
        database_url=get_database_url(args.database_url),
    )
    print(
        f"[INFO] Wrote {len(canonical_df)} canonical historical odds row(s), "
        f"{len(conflicts_df)} conflict row(s), "
        f"persisted {odds_count} odds row(s) and {conflict_count} conflict row(s)"
    )


if __name__ == "__main__":
    main()
