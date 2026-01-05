#!/usr/bin/env python
"""
log_today_props.py

1. Fetch today's NBA player props from The Odds API into a dated raw file.
2. Convert them into one-row-per-player/game/line market lines into another dated file.

Requires:
  - ODDS_API_KEY environment variable set
  - fetch_props_from_the_odds_api.py in the same repo
  - props_to_market_lines.py in the same repo
"""

import subprocess
from pathlib import Path
from datetime import datetime
import sys

# Adjust these paths if your repo layout is different
FETCH_SCRIPT = "fetch_props_from_the_odds_api.py"
PROPS_TO_MARKET_SCRIPT = "props_to_market_lines.py"

RAW_DIR = Path("data/props_raw")
MARKET_DIR = Path("data/props_market")


def main():
    # Use today's date in YYYY-MM-DD (or change format if you prefer)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"odds_{today}.csv"
    market_path = MARKET_DIR / f"market_lines_{today}.csv"

    print("=== Logging today’s props ===")
    print(f"Date (UTC): {today}")
    print(f"Raw odds file:    {raw_path}")
    print(f"Market lines file:{market_path}")
    print("================================\n")

    # 1) Fetch raw odds into dated file
    try:
        subprocess.run(
            [
                sys.executable,
                FETCH_SCRIPT,
                "--markets",
                "player_points",
                "--output",
                str(raw_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: fetch_props_from_the_odds_api.py failed: {e}")
        sys.exit(1)

    # 2) Convert to market lines into dated file
    try:
        subprocess.run(
            [
                sys.executable,
                PROPS_TO_MARKET_SCRIPT,
                "--odds-slate",
                str(raw_path),
                "--output",
                str(market_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: props_to_market_lines.py failed: {e}")
        sys.exit(1)

    print("\nDone logging today’s props.")


if __name__ == "__main__":
    main()