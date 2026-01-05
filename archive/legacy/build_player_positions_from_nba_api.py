#!/usr/bin/env python
"""
build_player_positions_from_nba_api.py

Build (or update) a small lookup table of player positions from nba_api.

Reads:
  - data/player_game_logs.csv

Optionally reads:
  - data/player_positions.csv (if it already exists) and only fetches
    positions for NEW player_ids.

Outputs:
  - data/player_positions.csv with columns:
      * player_id (int)
      * player_name (str, from game logs)
      * raw_position (str, as returned by nba_api CommonPlayerInfo)
      * position (str, normalized bucket: 'G', 'F', 'C' or None)

Usage:
    python build_player_positions_from_nba_api.py
"""

import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from nba_api.stats.endpoints import CommonPlayerInfo

from build_player_game_logs_from_nba_api import OUT_CSV as PLAYER_LOGS_CSV

OUT_DIR = Path("data")
OUT_POSITIONS_CSV = OUT_DIR / "player_positions.csv"


def normalize_position(raw_pos: Optional[str]) -> Optional[str]:
    """
    Normalize raw position strings like:
      - "G", "F", "C"
      - "G-F", "F-G", "F-C", "C-F", etc.

    into coarse buckets: 'G', 'F', 'C'.

    If nothing sensible can be inferred, return None.
    """
    if raw_pos is None:
        return None

    s = str(raw_pos).strip().upper()
    if not s:
        return None

    # Treat hybrid positions by a simple priority:
    #   G > F > C
    has_g = "G" in s
    has_f = "F" in s
    has_c = "C" in s

    if has_g:
        return "G"
    if has_f:
        return "F"
    if has_c:
        return "C"
    return None


def fetch_player_position(player_id: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Call CommonPlayerInfo for a single player_id and return:
        (raw_position, normalized_position)

    If the call fails for any reason, returns (None, None).
    """
    try:
        info = CommonPlayerInfo(player_id=player_id)
        df_info = info.get_data_frames()[0]
        if "POSITION" not in df_info.columns:
            return None, None
        raw_pos = df_info["POSITION"].iloc[0]
        norm_pos = normalize_position(raw_pos)
        return str(raw_pos) if raw_pos is not None else None, norm_pos
    except Exception as e:
        print(f"WARNING: failed to fetch CommonPlayerInfo for player_id={player_id}: {e}")
        return None, None


def main():
    if not PLAYER_LOGS_CSV.exists():
        print(
            f"Player logs CSV not found at {PLAYER_LOGS_CSV}.\n"
            "Run build_player_game_logs_from_nba_api.py first."
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading player logs from {PLAYER_LOGS_CSV} ...")
    df_logs = pd.read_csv(PLAYER_LOGS_CSV)

    if "player_id" not in df_logs.columns or "player_name" not in df_logs.columns:
        print("Expected 'player_id' and 'player_name' in player logs; aborting.")
        return

    # Unique player_id / name combos from logs
    players = (
        df_logs[["player_id", "player_name"]]
        .drop_duplicates()
        .sort_values("player_id")
        .reset_index(drop=True)
    )

    print(f"Found {len(players)} unique players in logs.")

    # Load existing positions (if any)
    if OUT_POSITIONS_CSV.exists():
        print(f"Loading existing positions from {OUT_POSITIONS_CSV} ...")
        df_pos_existing = pd.read_csv(OUT_POSITIONS_CSV)
        if "player_id" not in df_pos_existing.columns:
            print("Existing player_positions.csv missing 'player_id' column; ignoring it.")
            df_pos_existing = pd.DataFrame(
                columns=["player_id", "player_name", "raw_position", "position"]
            )
    else:
        df_pos_existing = pd.DataFrame(
            columns=["player_id", "player_name", "raw_position", "position"]
        )

    # Treat only rows with a non-null 'position' as "existing"
    if not df_pos_existing.empty:
        valid_mask = df_pos_existing["position"].notna()
        existing_ids = set(df_pos_existing.loc[valid_mask, "player_id"].unique())
    else:
        existing_ids = set()

    all_ids = set(players["player_id"].unique())

    missing_ids = sorted(all_ids - existing_ids)
    print(f"{len(existing_ids)} players already have positions.")
    print(f"{len(missing_ids)} players are missing positions and will be fetched.")

    new_rows: List[Dict] = []

    # Fail-safe against long stretches of timeouts / rate limits
    fail_streak = 0
    max_fail_streak = 10  # stop if we see this many failures in a row

    for idx, pid in enumerate(missing_ids, start=1):
        name = players.loc[players["player_id"] == pid, "player_name"].iloc[0]
        print(f"[{idx}/{len(missing_ids)}] Fetching position for {name} (player_id={pid}) ...", end="")
        raw_pos, norm_pos = fetch_player_position(int(pid))

        if raw_pos is None and norm_pos is None:
            fail_streak += 1
            print(" -> FAILED (timeout / error); will retry in a future run.")
            if fail_streak >= max_fail_streak:
                print(
                    f"Too many consecutive failures ({fail_streak}); "
                    "stopping early to avoid more rate-limiting. "
                    "Re-run this script later to continue."
                )
                break
        else:
            fail_streak = 0
            print(f" -> raw='{raw_pos}', norm='{norm_pos}'")
            new_rows.append(
                {
                    "player_id": int(pid),
                    "player_name": name,
                    "raw_position": raw_pos,
                    "position": norm_pos,
                }
            )

        # Be kind to stats.nba.com; slightly longer delay helps avoid rate limiting.
        time.sleep(1.0)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_positions = pd.concat([df_pos_existing, df_new], ignore_index=True)
    else:
        df_positions = df_pos_existing

    # Drop duplicates in case something weird happened; keep last
    if not df_positions.empty:
        df_positions = (
            df_positions.sort_values(["player_id"])
            .drop_duplicates(subset=["player_id"], keep="last")
            .reset_index(drop=True)
        )

    df_positions.to_csv(OUT_POSITIONS_CSV, index=False)
    print(f"\nSaved {len(df_positions)} player positions to {OUT_POSITIONS_CSV}")

    print("\nSample rows:")
    print(
        df_positions.head(15)[
            ["player_id", "player_name", "raw_position", "position"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()