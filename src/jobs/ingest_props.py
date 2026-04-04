"""Job wrapper for current-day NBA prop ingestion from selected live sources."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys

from src.data.public_page_props import (
    PROPS_SOURCE_FALLBACKS,
    SUPPORTED_PROP_MARKETS,
    fetch_covers_prop_rows,
    merge_prop_source_rows,
    fetch_scoresandodds_prop_rows,
    write_prop_rows,
)
from src.data.sportsgameodds import fetch_sportsgameodds_prop_rows, get_sportsgameodds_api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch current-day NBA props from a selected live source.")
    parser.add_argument("--report-date", default=date.today().isoformat(), help="Report date in YYYY-MM-DD.")
    parser.add_argument("--source", choices=["scoresandodds", "covers", "sportsgameodds", "the-odds-api"], default="scoresandodds")
    parser.add_argument("--sportsgameodds-api-key", default=None, help="SportsGameOdds API key.")
    parser.add_argument(
        "--markets",
        default=",".join(sorted(SUPPORTED_PROP_MARKETS)),
        help="Comma-separated market keys. Supported public-page markets: player_points,player_rebounds,player_assists,player_threes.",
    )
    parser.add_argument("--output", default="data/odds_slate.csv", help="Raw prop slate CSV output.")
    parser.add_argument("--market-lines-output", default="data/market_lines.csv", help="Aggregated market-lines CSV output.")
    parser.add_argument(
        "--allow-one-sided",
        action="store_true",
        help="Keep one-sided market rows instead of requiring both over and under odds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_date = date.fromisoformat(args.report_date)
    markets = [item.strip() for item in args.markets.split(",") if item.strip()]
    output_path = Path(args.output)
    market_lines_output = Path(args.market_lines_output)

    if args.source == "the-odds-api":
        subprocess.run(
            [
                sys.executable,
                "src/data/fetch_props_from_the_odds_api.py",
                "--markets",
                ",".join(markets),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        rows_written = int(len(Path(output_path).read_text(encoding="utf-8").splitlines())) - 1 if output_path.exists() else 0
        print(f"[INFO] Wrote raw prop rows from the-odds-api to {output_path}")
    else:
        rows_by_source: dict[str, list[dict]] = {}
        failures: list[str] = []
        for candidate in PROPS_SOURCE_FALLBACKS[args.source]:
            try:
                if candidate == "scoresandodds":
                    rows_by_source[candidate] = fetch_scoresandodds_prop_rows(report_date=report_date, allowed_markets=markets)
                elif candidate == "covers":
                    rows_by_source[candidate] = fetch_covers_prop_rows(report_date=report_date, allowed_markets=markets)
                elif candidate == "sportsgameodds":
                    rows_by_source[candidate] = fetch_sportsgameodds_prop_rows(
                        report_date=report_date,
                        allowed_markets=markets,
                        api_key=get_sportsgameodds_api_key(args.sportsgameodds_api_key),
                    )
                else:
                    continue
                if not rows_by_source[candidate]:
                    failures.append(f"{candidate}: no rows")
                    rows_by_source.pop(candidate, None)
            except Exception as exc:  # pragma: no cover - exercised in pipeline
                failures.append(f"{candidate}: {exc}")

        if not rows_by_source:
            raise RuntimeError("No live props source succeeded: " + " | ".join(failures))

        rows, sources_used = merge_prop_source_rows(
            rows_by_source,
            source_priority=PROPS_SOURCE_FALLBACKS[args.source],
        )
        write_prop_rows(rows, output_path)
        print(f"[INFO] Wrote {len(rows)} raw prop row(s) from {', '.join(sources_used)} to {output_path}")
        if failures:
            print(f"[WARN] Some prop sources failed: {' | '.join(failures)}")

    command = [
        sys.executable,
        "src/data/props_to_market_lines.py",
        "--odds-slate",
        str(output_path),
        "--output",
        str(market_lines_output),
    ]
    if not args.allow_one_sided:
        command.append("--require-two-sided")
    subprocess.run(command, check=True)
    print(f"[INFO] Wrote market lines to {market_lines_output}")


if __name__ == "__main__":
    main()
