#!/usr/bin/env python
"""End-to-end pipeline for training, scoring, and materializing beta recommendations."""

import argparse
import json
import subprocess
import sys
import shutil
import re
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import numpy as np

from src.data.build_lineup_projections import build_lineup_projection_frame, persist_lineup_projections
from src.data.build_starter_history import build_starter_history_frame, persist_starter_history
from src.data.fetch_props_from_the_odds_api import write_csv as write_prop_slate_csv
from src.data.historical_game_odds import (
    DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV,
    DEFAULT_HISTORICAL_ODDS_CONFLICTS_CSV,
    DEFAULT_HISTORICAL_ODDS_MANIFEST,
    backfill_player_logs,
    build_historical_snapshot_frame,
    export_game_lines_history,
    import_historical_odds_sources,
    persist_historical_odds,
    reconcile_historical_odds,
    write_historical_odds_artifacts,
)
from src.data.oddspapi_game_odds import persist_game_odds
from src.data.official_injuries import fetch_official_injury_reports, persist_official_injury_reports
from src.data.public_page_game_odds import fetch_espn_game_frames, fetch_scoresandodds_game_frames
from src.data.public_page_props import (
    PROPS_SOURCE_FALLBACKS,
    SUPPORTED_PROP_MARKETS,
    fetch_covers_prop_rows,
    fetch_scoresandodds_prop_rows,
    merge_prop_source_rows,
)
from src.data.sportsgameodds import (
    fetch_sportsgameodds_game_frames,
    fetch_sportsgameodds_prop_rows,
    get_sportsgameodds_api_key,
)
from src.data.the_odds_api_game_odds import (
    DEFAULT_BOOKMAKERS,
    fetch_current_game_odds_snapshots,
    get_the_odds_api_key,
)
from src.evaluation.build_market_readiness_snapshot import load_training_metrics
from src.evaluation.build_market_readiness_snapshot import main as build_market_readiness_main
from src.inference.score_game_markets import build_game_market_recommendations
from src.warehouse.db import get_database_url, init_database
from src.warehouse.materialize import materialize_edges

# You can tweak these defaults if you want different behavior
TRAIN_MAX_SEASON = "2024"
VAL_MIN_SEASON = "2025"

# New: model + tuning defaults for the regression step
MODEL_TYPE = "xgboost"       # or "histgb"
N_TUNE_ITER = "40"           # how many hyperparam configs to sample

SEASON_MIN_FOR_FORM = "2023"
MIN_LINE_FOR_STARS = "15.0"
MAX_STARS = "35"
BOOKS = "Bet365,FanDuel,DraftKings,Bovada,BetMGM,BetRivers"
TOP_K = "10"
MIN_EDGE_DISPLAY = "5.0"
LADDER_THRESHOLDS = "10,15,20,25,30,35,40"
TARGET_PROB = "0.50"
DEFAULT_GAME_ODDS_SOURCE = "scoresandodds"
DEFAULT_PROPS_SOURCE = "scoresandodds"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_BIN = sys.executable

# NEW: logging locations
PROPS_RAW_DIR = Path("data/props_raw")
PROPS_MARKET_DIR = Path("data/props_market")
EDGES_DIR = Path("data/edges")
RUN_LOG_DIR = Path("data/run_logs")
GAME_LINES_DIR = Path("data/game_lines_history")  # NEW: for Vegas lines

LATEST_ODDS_SLATE = Path("data/odds_slate.csv")
LATEST_MARKET_LINES = Path("data/market_lines.csv")
LATEST_EDGES = Path("data/edges_with_market.csv")
LATEST_GAME_LINES = Path("data/game_lines.csv")  # NEW: for Vegas lines
LATEST_GAME_MARKET_RECOMMENDATIONS = Path("data/game_market_recommendations.csv")
LATEST_MARKET_READINESS = Path("data/market_readiness.csv")
OFFICIAL_INJURIES_CSV = Path("data/official_injuries.csv")
STARTER_HISTORY_CSV = Path("data/starter_history.csv")
LINEUP_PROJECTIONS_CSV = Path("data/lineup_projections.csv")
GAME_ODDS_SNAPSHOTS_CSV = Path("data/game_odds_snapshots.csv")
CLOSING_LINES_CSV = Path("data/closing_lines.csv")
PLAYER_POSITIONS_CSV = Path("data/player_positions.csv")
HISTORICAL_ODDS_DIR = Path("data/historical_odds")
HISTORICAL_ODDS_MANIFEST = DEFAULT_HISTORICAL_ODDS_MANIFEST
CANONICAL_HISTORICAL_ODDS_CSV = DEFAULT_CANONICAL_HISTORICAL_ODDS_CSV
HISTORICAL_ODDS_CONFLICTS_CSV = DEFAULT_HISTORICAL_ODDS_CONFLICTS_CSV
PIPELINE_STATE_DIR = Path("data/pipeline_state")
INJURY_BACKFILL_CURSOR = PIPELINE_STATE_DIR / "injury_backfill_cursor.json"
HISTORICAL_REPLAY_CURSOR = PIPELINE_STATE_DIR / "historical_replay_cursor.json"

FEATURES_CSV = Path("data/player_points_features.csv")
FEATURES_WITH_PROPS_CSV = Path("data/player_points_features_with_props.csv")
FEATURES_WITH_VEGAS_CSV = Path("data/player_points_features_with_vegas.csv")
FEATURES_WITH_INJURIES_CSV = Path("data/player_points_features_with_injuries.csv")  # NEW: final features with all enhancements
FEATURES_WITH_LINEUP_CSV = Path("data/player_points_features_with_lineup.csv")
INJURY_DATA_CSV = Path("data/injury_data.csv")

GAME_ODDS_SOURCE_FALLBACKS = {
    "scoresandodds": ("scoresandodds", "espn"),
    "espn": ("espn",),
    "sportsgameodds": ("sportsgameodds",),
    "the-odds-api": ("the-odds-api",),
}

def run(cmd, desc=None, stdout_path: Path | None = None):
    """
    Helper to run a shell command and print nice separators.
    If stdout_path is provided, prints to terminal AND writes to that file.
    """
    if desc:
        print("\n" + "=" * 80)
        print(desc)
        print("=" * 80)

    print(f"\n$ {' '.join(cmd)}\n")

    try:
        if stdout_path is None:
            subprocess.run(cmd, check=True)
            return

        stdout_path.parent.mkdir(parents=True, exist_ok=True)

        # Tee output: terminal + file
        with open(stdout_path, "w", encoding="utf-8") as f:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end="")     # terminal
                f.write(line)           # file
            rc = p.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        if stdout_path is not None:
            print(f"[ERROR] See log: {stdout_path}")
        sys.exit(e.returncode)


def python_cmd(*args: str) -> list[str]:
    return [PYTHON_BIN, *args]


def normalize_player_name(s: str) -> str:
    """
    A light normalizer to reduce name mismatches between props feeds and features.
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(".", "")
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s'-]", "", s)  # keep letters/numbers/spaces/'/-
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_features_with_vegas_lines(
    features_csv: Path,
    game_lines_dir: Path,
    out_csv: Path
) -> tuple[int, float]:
    """
    Join ALL logged game_lines_YYYY-MM-DD.csv files into the full features table.
    
    Adds vegas_game_total, vegas_home_spread, vegas_away_spread, vegas_abs_spread
    as features for model training.
    
    Returns:
      (non_null_vegas_count, vegas_share)
    """
    from nba_api.stats.static import teams as nba_teams
    
    if not features_csv.exists():
        print(f"[WARN] Features file not found: {features_csv}; skipping Vegas lines join.")
        return 0, 0.0
    
    game_lines_files = sorted(game_lines_dir.glob("game_lines_*.csv"))
    if not game_lines_files:
        print(f"[INFO] No game lines files in {game_lines_dir}. Skipping Vegas lines join.")
        # Write a copy so downstream doesn't break
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0
    
    print(f"[INFO] Building Vegas lines history from {len(game_lines_files)} files...")
    
    # Build team name to abbreviation mapping
    all_teams = nba_teams.get_teams()
    team_name_to_abbrev = {}
    for t in all_teams:
        abbr = t.get("abbreviation", "").upper()
        full_name = t.get("full_name", "").lower()
        nickname = t.get("nickname", "").lower()
        city = t.get("city", "").lower()
        if abbr:
            if full_name:
                team_name_to_abbrev[full_name] = abbr
            if nickname:
                team_name_to_abbrev[nickname] = abbr
            if city and nickname:
                team_name_to_abbrev[f"{city} {nickname}"] = abbr
    
    def normalize_team_name(name):
        if not name:
            return ""
        return str(name).strip().lower()
    
    def get_team_abbrev(name):
        return team_name_to_abbrev.get(normalize_team_name(name), "")
    
    # Load all game lines files
    lines_list = []
    for fp in game_lines_files:
        try:
            df_lines = pd.read_csv(fp)
            if "game_date" not in df_lines.columns:
                print(f"[WARN] Skipping {fp} (missing game_date column)")
                continue
            lines_list.append(df_lines)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    
    if not lines_list:
        print("[WARN] No valid game lines loaded; writing features copy only.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0
    
    lines_df = pd.concat(lines_list, ignore_index=True)
    
    # Parse and normalize
    lines_df["game_date"] = pd.to_datetime(lines_df["game_date"]).dt.date
    lines_df["home_abbrev"] = lines_df["home_team"].apply(get_team_abbrev)
    lines_df["away_abbrev"] = lines_df["away_team"].apply(get_team_abbrev)
    
    # Dedupe: keep last entry per game_date + home_team + away_team
    lines_df = lines_df.drop_duplicates(
        subset=["game_date", "home_abbrev", "away_abbrev"],
        keep="last"
    )
    
    # Load features
    features = pd.read_csv(features_csv)
    if "game_date" not in features.columns or "team_abbrev" not in features.columns:
        print("[WARN] Features missing game_date/team_abbrev; cannot join Vegas lines.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0
    
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
    
    # Create Vegas columns in features (for home team rows)
    features_home = features.merge(
        lines_df[["game_date", "home_abbrev", "vegas_game_total", "vegas_home_spread", "vegas_abs_spread"]],
        how="left",
        left_on=["game_date", "team_abbrev"],
        right_on=["game_date", "home_abbrev"],
    )
    features_home = features_home.drop(columns=["home_abbrev"], errors="ignore")
    features_home["is_home_for_vegas"] = features_home["vegas_game_total"].notna()
    
    # For away team rows
    lines_df_away = lines_df[["game_date", "away_abbrev", "vegas_game_total", "vegas_away_spread", "vegas_abs_spread"]].copy()
    lines_df_away = lines_df_away.rename(columns={
        "vegas_game_total": "vegas_game_total_away",
        "vegas_away_spread": "vegas_home_spread_away",  # away spread becomes "their" spread
        "vegas_abs_spread": "vegas_abs_spread_away",
    })
    
    features_merged = features_home.merge(
        lines_df_away,
        how="left",
        left_on=["game_date", "team_abbrev"],
        right_on=["game_date", "away_abbrev"],
    )
    features_merged = features_merged.drop(columns=["away_abbrev"], errors="ignore")
    
    # Coalesce home and away values
    features_merged["vegas_game_total"] = features_merged["vegas_game_total"].fillna(
        features_merged.get("vegas_game_total_away", pd.NA)
    )
    features_merged["vegas_spread"] = features_merged["vegas_home_spread"].fillna(
        features_merged.get("vegas_home_spread_away", pd.NA)
    )
    features_merged["vegas_abs_spread"] = features_merged["vegas_abs_spread"].fillna(
        features_merged.get("vegas_abs_spread_away", pd.NA)
    )
    
    # Drop intermediate columns
    drop_cols = ["vegas_game_total_away", "vegas_home_spread", "vegas_home_spread_away", 
                 "vegas_abs_spread_away", "is_home_for_vegas"]
    features_merged = features_merged.drop(columns=[c for c in drop_cols if c in features_merged.columns])
    
    # PHASE 4A: Add game script features (blowout probability)
    if "vegas_abs_spread" in features_merged.columns and "team_margin_roll5" in features_merged.columns:
        # Blowout probability: higher abs_spread = more likely blowout
        # Normalize abs_spread to 0-1 scale (assuming max spread ~20)
        features_merged["vegas_spread_abs_normalized"] = (
            features_merged["vegas_abs_spread"].fillna(0) / 20.0
        ).clip(0, 1)
        
        # Binary: is likely blowout? (spread > 12 points)
        features_merged["is_likely_blowout"] = (
            features_merged["vegas_abs_spread"].fillna(0) > 12
        ).astype(int)
        
        # Blowout probability (simple heuristic: spread/20, capped at 1.0)
        features_merged["blowout_prob"] = features_merged["vegas_spread_abs_normalized"]
        
        # Estimated garbage time minutes (more blowout = more garbage time)
        # Assume ~5-10 minutes of garbage time in blowouts
        features_merged["garbage_time_minutes_est"] = (
            features_merged["blowout_prob"] * 7.5  # 7.5 minutes average in blowouts
        )
        
        print("[INFO] Added game script features: blowout_prob, is_likely_blowout, garbage_time_minutes_est")
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features_merged.to_csv(out_csv, index=False)
    
    non_null = int(features_merged["vegas_game_total"].notna().sum()) if "vegas_game_total" in features_merged.columns else 0
    share = float(features_merged["vegas_game_total"].notna().mean()) if "vegas_game_total" in features_merged.columns else 0.0
    
    print(f"[INFO] Wrote features-with-vegas: {out_csv}")
    print(f"[INFO] vegas_game_total non-null: {non_null:,} ({share:.4%})")
    return non_null, share


def build_features_with_injuries(features_csv: Path, injury_csv: Path, out_csv: Path) -> tuple[int, float]:
    """
    Join injury/availability data into the features table.
    
    Adds:
    - is_injured (binary: 1 if injured/DNP, 0 otherwise)
    - injury_status (probable/questionable/out/healthy)
    - days_since_last_dnp
    - dnp_count_last_10
    
    Returns:
      (non_null_injury_count, injury_share)
    """
    if not features_csv.exists():
        print(f"[WARN] Features file not found: {features_csv}; skipping injury join.")
        return 0, 0.0
    
    if not injury_csv.exists():
        print(f"[INFO] Injury data file not found: {injury_csv}. Skipping injury join.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0
    
    print(f"[INFO] Loading injury data from {injury_csv} ...")
    injuries = pd.read_csv(injury_csv)
    
    # Normalize dates
    injuries["game_date"] = pd.to_datetime(injuries["game_date"]).dt.date
    
    # Load features
    features = pd.read_csv(features_csv)
    if "game_date" not in features.columns or "player_id" not in features.columns:
        print("[WARN] Features missing game_date/player_id; cannot join injuries.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0
    
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
    
    # Select injury columns to merge
    injury_cols = [
        "player_id",
        "game_date",
        "is_injured",
        "injury_status",
        "days_since_last_dnp",
        "dnp_count_last_10",
    ]
    
    # Merge (left join: keep all feature rows)
    merged = features.merge(
        injuries[injury_cols],
        on=["player_id", "game_date"],
        how="left",
        suffixes=("", "_injury"),
    )
    
    # Fill missing injury data (for games before injury tracking started)
    merged["is_injured"] = merged["is_injured"].fillna(0).astype(int)
    merged["injury_status"] = merged["injury_status"].fillna("healthy")
    merged["days_since_last_dnp"] = merged["days_since_last_dnp"].fillna(999).astype(int)
    merged["dnp_count_last_10"] = merged["dnp_count_last_10"].fillna(0).astype(int)
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    
    non_null = int(merged["is_injured"].sum()) if "is_injured" in merged.columns else 0
    share = float(merged["is_injured"].mean()) if "is_injured" in merged.columns else 0.0
    
    print(f"[INFO] Wrote features-with-injuries: {out_csv}")
    print(f"[INFO] is_injured=1 count: {non_null:,} ({share:.4%})")
    return non_null, share


def build_features_with_lineup_context(features_csv: Path, out_csv: Path) -> tuple[int, float]:
    """
    Add lineup context / rotation depth features from the per-player feature table.

    Requires columns:
      - season, game_date, team_abbrev, is_injured
    Uses (if available) as proxies for "missing usage/minutes":
      - usg_events_roll15, minutes_roll15, star_tier_pts

    Adds:
      - teammate_out_count
      - teammate_out_star_count
      - teammate_out_usg15_sum
      - teammate_out_min15_sum
      - team_available_players
      - is_team_shorthanded

    Returns:
      (non_null_rows, share_non_zero_teammate_out)
    """
    if not features_csv.exists():
        print(f"[WARN] Features file not found: {features_csv}; skipping lineup context.")
        return 0, 0.0

    df = pd.read_csv(features_csv)
    if "game_date" not in df.columns or "team_abbrev" not in df.columns:
        print("[WARN] Features missing game_date/team_abbrev; cannot build lineup context. Writing copy only.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Ensure required columns exist (safe defaults)
    if "season" not in df.columns:
        df["season"] = 0
    if "is_injured" not in df.columns:
        df["is_injured"] = 0

    if "usg_events_roll15" not in df.columns:
        df["usg_events_roll15"] = 0.0
    if "minutes_roll15" not in df.columns:
        df["minutes_roll15"] = 0.0
    if "star_tier_pts" not in df.columns:
        df["star_tier_pts"] = 0

    # Clean types
    df["is_injured"] = df["is_injured"].fillna(0).astype(int)
    df["usg_events_roll15"] = pd.to_numeric(df["usg_events_roll15"], errors="coerce").fillna(0.0)
    df["minutes_roll15"] = pd.to_numeric(df["minutes_roll15"], errors="coerce").fillna(0.0)
    df["star_tier_pts"] = pd.to_numeric(df["star_tier_pts"], errors="coerce").fillna(0).astype(int)

    group_keys = ["season", "game_date", "team_abbrev"]

    g = df.groupby(group_keys, dropna=False)

    # Compute weighted sums via vectorization (avoid slow groupby apply)
    df["_inj_usg15"] = df["is_injured"] * df["usg_events_roll15"]
    df["_inj_min15"] = df["is_injured"] * df["minutes_roll15"]
    df["_inj_star"] = (df["is_injured"] * (df["star_tier_pts"] >= 2).astype(int)).astype(int)

    df_team = g.agg(
        team_injured_count=("is_injured", "sum"),
        team_player_rows=("is_injured", "count"),
        team_injured_usg15_sum=("_inj_usg15", "sum"),
        team_injured_min15_sum=("_inj_min15", "sum"),
        team_injured_star_count=("_inj_star", "sum"),
    ).reset_index()

    merged = df.merge(df_team, on=group_keys, how="left")

    # Per-player teammate features (exclude self)
    merged["teammate_out_count"] = (merged["team_injured_count"] - merged["is_injured"]).clip(lower=0).astype(int)
    merged["teammate_out_star_count"] = (
        merged["team_injured_star_count"] - merged["_inj_star"]
    ).clip(lower=0).astype(int)
    merged["teammate_out_usg15_sum"] = (merged["team_injured_usg15_sum"] - merged["_inj_usg15"]).clip(lower=0.0)
    merged["teammate_out_min15_sum"] = (merged["team_injured_min15_sum"] - merged["_inj_min15"]).clip(lower=0.0)

    merged["team_available_players"] = (merged["team_player_rows"] - merged["team_injured_count"]).clip(lower=0).astype(int)
    merged["is_team_shorthanded"] = (merged["teammate_out_count"] >= 2).astype(int)

    # Cleanup temp cols
    merged = merged.drop(columns=["_inj_usg15", "_inj_min15", "_inj_star"], errors="ignore")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    non_zero = int((merged["teammate_out_count"] > 0).sum()) if "teammate_out_count" in merged.columns else 0
    share = float((merged["teammate_out_count"] > 0).mean()) if "teammate_out_count" in merged.columns else 0.0
    print(f"[INFO] Wrote features-with-lineup: {out_csv}")
    print(f"[INFO] teammate_out_count > 0 rows: {non_zero:,} ({share:.4%})")
    return non_zero, share


def build_features_with_props(features_csv: Path, market_dir: Path, out_csv: Path) -> tuple[int, float]:
    """
    Join ALL logged market_lines_YYYY-MM-DD.csv files into the full features table.

    Returns:
      (non_null_prop_count, prop_share)
    """
    if not features_csv.exists():
        print(f"[WARN] Features file not found: {features_csv}; skipping props join.")
        return 0, 0.0

    market_files = sorted(market_dir.glob("market_lines_*.csv"))
    if not market_files:
        print(f"[INFO] No market lines files in {market_dir}. Skipping props join.")
        # still write a copy so downstream doesn't break if they point to it
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0

    print(f"[INFO] Building props history from {len(market_files)} files...")
    props_list = []
    for fp in market_files:
        try:
            dfp = pd.read_csv(fp)
            # expected cols: player, game_date, prop_pts_line, prop_over_odds_best, prop_under_odds_best, ...
            if "player" not in dfp.columns or "game_date" not in dfp.columns:
                print(f"[WARN] Skipping {fp} (missing player/game_date columns)")
                continue
            props_list.append(dfp)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")

    if not props_list:
        print("[WARN] No valid market lines loaded; writing features copy only.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0

    props = pd.concat(props_list, ignore_index=True)

    # Rename odds columns to match expected names (prop_over_odds_best, prop_under_odds_best)
    rename_map = {
        "over_odds_best": "prop_over_odds_best",
        "under_odds_best": "prop_under_odds_best",
    }
    # Only rename columns that actually exist
    rename_map_effective = {
        src: dst for src, dst in rename_map.items() if src in props.columns
    }
    if rename_map_effective:
        props = props.rename(columns=rename_map_effective)

    # Normalize dates + names
    props["game_date"] = pd.to_datetime(props["game_date"]).dt.date
    props["player_key"] = props["player"].map(normalize_player_name)

    # If duplicates exist (many books/lines), keep “best row” per player/date.
    # market_lines should already be collapsed per player/date/line, but we’ll dedupe anyway.
    dedupe_cols = ["game_date", "player_key", "prop_pts_line"]
    keep_cols = [c for c in props.columns if c not in ["player"]]  # keep original player column optional
    props = props[keep_cols].dropna(subset=["game_date", "player_key"])
    props = props.drop_duplicates(subset=dedupe_cols, keep="last")

    # Load features
    features = pd.read_csv(features_csv)
    if "game_date" not in features.columns or "player_name" not in features.columns:
        print("[WARN] Features missing game_date/player_name; cannot join props. Writing features copy only.")
        shutil.copyfile(features_csv, out_csv)
        return 0, 0.0

    features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
    features["player_key"] = features["player_name"].map(normalize_player_name)

    # Merge (left join: keep all feature rows)
    merged = features.merge(
        props,
        how="left",
        on=["game_date", "player_key"],
        suffixes=("", "_prop"),
    )

    # Drop join key helper
    merged.drop(columns=["player_key"], inplace=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    non_null = int(merged["prop_pts_line"].notna().sum()) if "prop_pts_line" in merged.columns else 0
    share = float(merged["prop_pts_line"].notna().mean()) if "prop_pts_line" in merged.columns else 0.0

    print(f"[INFO] Wrote features-with-props: {out_csv}")
    print(f"[INFO] prop_pts_line non-null: {non_null:,} ({share:.4%})")
    return non_null, share


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NBA betting pipeline in daily, bootstrap, or backfill-only mode.")
    parser.add_argument("--mode", choices=["daily", "bootstrap", "backfill-only"], default="daily")
    parser.add_argument("--report-date", default=datetime.now().date().isoformat())
    parser.add_argument("--backfill-start-date", default=None)
    parser.add_argument("--backfill-end-date", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--historical-odds-manifest", default=str(HISTORICAL_ODDS_MANIFEST))
    parser.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
    parser.add_argument("--game-odds-source", choices=sorted(GAME_ODDS_SOURCE_FALLBACKS), default=DEFAULT_GAME_ODDS_SOURCE)
    parser.add_argument("--props-source", choices=sorted(PROPS_SOURCE_FALLBACKS), default=DEFAULT_PROPS_SOURCE)
    parser.add_argument("--skip-star-screener", action="store_true")
    parser.add_argument("--skip-prop-training", action="store_true")
    parser.add_argument("--skip-game-training", action="store_true")
    parser.add_argument("--publish-current-day-at-end", action="store_true")
    parser.add_argument("--reset-cursors", action="store_true")
    return parser


def ensure_runtime_dirs() -> None:
    PROPS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROPS_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    GAME_LINES_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICAL_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)


def run_command(
    cmd: list[str],
    *,
    stdout_path: Optional[Path] = None,
) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    if stdout_path is None:
        subprocess.run(cmd, check=True)
        return

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def execute_step(
    run_log: dict,
    name: str,
    func: Callable[[], Optional[dict]],
    *,
    strict: bool,
) -> Optional[dict]:
    started_at = utc_now_iso()
    try:
        result = func() or {}
        run_log["steps"].append(
            {
                "name": name,
                "status": "completed",
                "strict": strict,
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                **result,
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - failure path exercised in pipeline tests
        run_log["steps"].append(
            {
                "name": name,
                "status": "failed",
                "strict": strict,
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "error": str(exc),
            }
        )
        if strict:
            raise
        run_log.setdefault("warnings", []).append(f"{name}: {exc}")
        return None


def default_backfill_window(report_day: date) -> tuple[date, date]:
    current_season_start = report_day.year if report_day.month >= 10 else report_day.year - 1
    start = date(current_season_start - 1, 10, 1)
    end = report_day - timedelta(days=1)
    return start, end


def rows_for_date(path: Path, date_value: str, *date_columns: str) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    for column in date_columns:
        if column not in df.columns:
            continue
        return int((df[column].astype(str) == date_value).sum())
    return 0


def require_rows(path: Path, date_value: str, *date_columns: str) -> int:
    count = rows_for_date(path, date_value, *date_columns)
    if count <= 0:
        raise RuntimeError(f"Expected at least one row in {path} for {date_value}")
    return count


def next_day(value: date) -> date:
    return value + timedelta(days=1)


def current_log_paths(report_day: date) -> dict[str, Path]:
    stamp = report_day.isoformat()
    return {
        "raw_props": PROPS_RAW_DIR / f"odds_slate_{stamp}.csv",
        "market_lines": PROPS_MARKET_DIR / f"market_lines_{stamp}.csv",
        "edges": EDGES_DIR / f"edges_with_market_{stamp}.csv",
        "screener_log": RUN_LOG_DIR / f"star_best_bets_{stamp}.txt",
        "game_lines": GAME_LINES_DIR / f"game_lines_{stamp}.csv",
        "run_log": RUN_LOG_DIR / f"pipeline_{stamp}.json",
    }


def write_current_game_lines(frame: pd.DataFrame, *, output_path: Path) -> int:
    if frame.empty:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return int(len(frame))


def backfill_official_injuries(
    *,
    start_date: date,
    end_date: date,
    output_path: Path,
    database_url: str,
    cursor_path: Path,
    reset_cursor: bool,
) -> dict:
    cursor = {} if reset_cursor else read_json(cursor_path)
    current = start_date
    if cursor.get("last_completed_date"):
        current = max(current, next_day(date.fromisoformat(cursor["last_completed_date"])))

    days_completed = 0
    rows_persisted = 0
    while current <= end_date:
        report_df = fetch_official_injury_reports(report_date=current, latest_only=True)
        rows_persisted += persist_official_injury_reports(report_df, output_path=output_path, database_url=database_url)
        days_completed += 1
        write_json(
            cursor_path,
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "last_completed_date": current.isoformat(),
                "updated_at": utc_now_iso(),
            },
        )
        current = next_day(current)
    return {"days_completed": days_completed, "rows_persisted": rows_persisted, "cursor_path": str(cursor_path)}


def import_historical_game_odds(
    *,
    manifest_path: Path,
    database_url: str,
) -> dict:
    source_rows = import_historical_odds_sources(manifest_path=manifest_path)
    canonical_df, conflicts_df = reconcile_historical_odds(source_rows)
    write_historical_odds_artifacts(
        canonical_df,
        conflicts_df,
        canonical_output_path=CANONICAL_HISTORICAL_ODDS_CSV,
        conflicts_output_path=HISTORICAL_ODDS_CONFLICTS_CSV,
    )
    odds_count, conflict_count = persist_historical_odds(
        canonical_df,
        conflicts_df,
        database_url=database_url,
    )
    return {
        "source_rows": int(len(source_rows)),
        "canonical_rows": int(len(canonical_df)),
        "conflict_rows": int(len(conflicts_df)),
        "persisted_odds_rows": odds_count,
        "persisted_conflict_rows": conflict_count,
        "canonical_output_path": str(CANONICAL_HISTORICAL_ODDS_CSV),
    }


def backfill_historical_market_data(*, database_url: str) -> dict:
    canonical_df = pd.read_csv(CANONICAL_HISTORICAL_ODDS_CSV) if CANONICAL_HISTORICAL_ODDS_CSV.exists() else pd.DataFrame()
    if canonical_df.empty:
        raise RuntimeError(f"No canonical historical odds rows found at {CANONICAL_HISTORICAL_ODDS_CSV}")

    logs_df = pd.read_csv("data/player_game_logs.csv")
    backfilled_logs, coverage = backfill_player_logs(logs_df, canonical_df)
    backfilled_logs.to_csv("data/player_game_logs.csv", index=False)

    snapshots = build_historical_snapshot_frame(canonical_df)
    snapshot_rows, closing_rows = persist_game_odds(
        snapshots,
        snapshots_output_path=GAME_ODDS_SNAPSHOTS_CSV,
        closing_output_path=CLOSING_LINES_CSV,
        database_url=database_url,
    )
    game_line_rows = export_game_lines_history(canonical_df, output_dir=GAME_LINES_DIR)
    return {
        "snapshot_rows": snapshot_rows,
        "closing_rows": closing_rows,
        "game_line_rows": game_line_rows,
        "spread_coverage_rate": coverage["spread_coverage_rate"],
        "total_coverage_rate": coverage["total_coverage_rate"],
        "moneyline_coverage_rate": coverage["moneyline_coverage_rate"],
    }


def build_prop_feature_stack() -> dict:
    run_command(python_cmd("src/pipeline/update_player_game_logs_incremental.py"))
    run_command(python_cmd("src/data/build_player_points_features.py"))
    run_command(python_cmd("src/data/fetch_injury_data.py"))
    injury_non_null, injury_share = build_features_with_injuries(
        features_csv=FEATURES_CSV,
        injury_csv=INJURY_DATA_CSV,
        out_csv=FEATURES_WITH_INJURIES_CSV,
    )
    lineup_non_zero, lineup_share = build_features_with_lineup_context(
        features_csv=FEATURES_WITH_INJURIES_CSV,
        out_csv=FEATURES_WITH_LINEUP_CSV,
    )
    prop_non_null, prop_share = build_features_with_props(
        features_csv=FEATURES_WITH_LINEUP_CSV,
        market_dir=PROPS_MARKET_DIR,
        out_csv=FEATURES_WITH_PROPS_CSV,
    )
    vegas_non_null, vegas_share = build_features_with_vegas_lines(
        features_csv=FEATURES_WITH_PROPS_CSV,
        game_lines_dir=GAME_LINES_DIR,
        out_csv=FEATURES_WITH_VEGAS_CSV,
    )
    return {
        "injury_non_null": injury_non_null,
        "injury_share": injury_share,
        "lineup_non_zero": lineup_non_zero,
        "lineup_share": lineup_share,
        "prop_non_null": prop_non_null,
        "prop_share": prop_share,
        "vegas_non_null": vegas_non_null,
        "vegas_share": vegas_share,
    }


def train_prop_models() -> None:
    run_command(
        [
            *python_cmd("src/models/build_minutes_regression.py"),
            "--model-type",
            MODEL_TYPE,
            "--train-max-season",
            TRAIN_MAX_SEASON,
            "--val-min-season",
            VAL_MIN_SEASON,
        ]
    )
    run_command(
        [
            *python_cmd("src/models/build_points_regression.py"),
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
            "--model-path",
            "models/points_regression_no_props.pkl",
            "--val-preds-out",
            "data/points_regression_no_props_val_preds.csv",
            "--model-type",
            MODEL_TYPE,
            "--train-max-season",
            TRAIN_MAX_SEASON,
            "--val-min-season",
            VAL_MIN_SEASON,
            "--tune-hyperparams",
            "--n-tune-iter",
            N_TUNE_ITER,
            "--use-minutes-pred",
        ]
    )
    run_command(
        [
            *python_cmd("src/models/build_points_regression.py"),
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
            "--model-type",
            MODEL_TYPE,
            "--train-max-season",
            TRAIN_MAX_SEASON,
            "--val-min-season",
            VAL_MIN_SEASON,
            "--tune-hyperparams",
            "--n-tune-iter",
            N_TUNE_ITER,
            "--use-prop-features",
            "--use-prop-derived-features",
            "--use-minutes-pred",
            "--model-baseline-path",
            "models/points_regression_no_props.pkl",
        ]
    )
    for target_col, model_out, val_out in [
        ("target_reb", "models/rebounds_regression.pkl", "data/rebounds_regression_val_preds.csv"),
        ("target_ast", "models/assists_regression.pkl", "data/assists_regression_val_preds.csv"),
        ("target_fg3m", "models/threes_regression.pkl", "data/threes_regression_val_preds.csv"),
    ]:
        run_command(
            [
                *python_cmd("src/models/build_points_regression.py"),
                "--features-csv",
                str(FEATURES_WITH_VEGAS_CSV),
                "--target-col",
                target_col,
                "--model-path",
                model_out,
                "--val-preds-out",
                val_out,
                "--model-type",
                MODEL_TYPE,
                "--train-max-season",
                TRAIN_MAX_SEASON,
                "--val-min-season",
                VAL_MIN_SEASON,
            ]
        )


def train_game_models(database_url: str) -> None:
    run_command(
        [
            *python_cmd("src/jobs/train_game_market_models.py"),
            "--logs-csv",
            str(Path("data/player_game_logs.csv")),
            "--injuries-csv",
            str(OFFICIAL_INJURIES_CSV),
            "--starters-csv",
            str(STARTER_HISTORY_CSV),
            "--models-dir",
            "models",
            "--metrics-out",
            str(Path("data/game_market_model_metrics.csv")),
        ]
    )


def active_teams_for_report_day(report_day: date, *, injuries_csv: Optional[Path] = None) -> list[str]:
    injuries_csv = injuries_csv or OFFICIAL_INJURIES_CSV
    if not injuries_csv.exists():
        return []
    try:
        injuries = pd.read_csv(injuries_csv)
    except Exception:
        return []
    if injuries.empty or "team_abbrev" not in injuries.columns:
        return []

    report_day_iso = report_day.isoformat()
    mask = pd.Series(False, index=injuries.index)
    for column in ("report_date", "game_date"):
        if column in injuries.columns:
            mask = mask | (injuries[column].astype(str) == report_day_iso)

    teams = (
        injuries.loc[mask, "team_abbrev"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    return sorted({team for team in teams if team and team.lower() != "nan"})


def limit_logs_to_recent_team_games(
    logs_df: pd.DataFrame,
    *,
    report_day: date,
    team_abbrevs: list[str],
    recent_window_days: int,
    recent_games_per_team: int,
) -> pd.DataFrame:
    if logs_df.empty:
        return logs_df.copy()

    logs = logs_df.copy()
    if "game_date" in logs.columns:
        logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
        recent_cutoff = pd.Timestamp(report_day - timedelta(days=recent_window_days))
        logs = logs[logs["game_date"] >= recent_cutoff].copy()

    if team_abbrevs and "team_abbrev" in logs.columns:
        team_set = {str(team).upper() for team in team_abbrevs}
        logs["team_abbrev"] = logs["team_abbrev"].astype(str).str.upper()
        logs = logs[logs["team_abbrev"].isin(team_set)].copy()

    if logs.empty:
        return logs

    required_columns = {"team_abbrev", "game_id", "game_date"}
    if not required_columns.issubset(logs.columns):
        return logs

    recent_team_games = (
        logs[["team_abbrev", "game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["team_abbrev", "game_date", "game_id"], ascending=[True, False, False])
        .groupby("team_abbrev")
        .head(recent_games_per_team)
    )
    limited = logs.merge(
        recent_team_games[["team_abbrev", "game_id"]],
        on=["team_abbrev", "game_id"],
        how="inner",
    )
    return limited.sort_values(["game_date", "game_id"], ascending=[False, False]).copy()


def refresh_starter_history(
    database_url: str,
    *,
    report_day: Optional[date] = None,
    recent_window_days: int = 21,
    recent_games_per_team: int = 10,
    max_games_if_missing: int = 30,
) -> dict:
    logs_df = pd.read_csv("data/player_game_logs.csv")
    max_games = None
    if report_day is not None and not logs_df.empty and "game_date" in logs_df.columns:
        active_teams = active_teams_for_report_day(report_day)
        logs_df = limit_logs_to_recent_team_games(
            logs_df,
            report_day=report_day,
            team_abbrevs=active_teams,
            recent_window_days=recent_window_days,
            recent_games_per_team=recent_games_per_team,
        )
    existing_game_ids = []
    if STARTER_HISTORY_CSV.exists():
        existing_game_ids = pd.read_csv(STARTER_HISTORY_CSV)["game_id"].astype(str).dropna().unique().tolist()
    elif report_day is not None:
        max_games = max_games_if_missing
    frame = build_starter_history_frame(logs_df, existing_game_ids=existing_game_ids, max_games=max_games)
    count = persist_starter_history(frame, output_path=STARTER_HISTORY_CSV, database_url=database_url)
    return {"rows_persisted": count}


def ingest_official_injuries_for_day(report_day: date, database_url: str) -> dict:
    report_df = fetch_official_injury_reports(report_date=report_day, latest_only=True)
    count = persist_official_injury_reports(report_df, output_path=OFFICIAL_INJURIES_CSV, database_url=database_url)
    return {"rows_persisted": count}


def build_lineups_for_day(report_day: date, database_url: str) -> dict:
    starter_history_df = pd.read_csv(STARTER_HISTORY_CSV) if STARTER_HISTORY_CSV.exists() else pd.DataFrame()
    logs_df = pd.read_csv("data/player_game_logs.csv")
    injuries_df = pd.read_csv(OFFICIAL_INJURIES_CSV) if OFFICIAL_INJURIES_CSV.exists() else pd.DataFrame()
    positions_df = pd.read_csv(PLAYER_POSITIONS_CSV) if PLAYER_POSITIONS_CSV.exists() else pd.DataFrame()
    frame = build_lineup_projection_frame(
        target_date=report_day,
        starter_history_df=starter_history_df,
        logs_df=logs_df,
        injuries_df=injuries_df,
        player_positions_df=positions_df,
    )
    count = persist_lineup_projections(frame, output_path=LINEUP_PROJECTIONS_CSV, database_url=database_url)
    return {"rows_persisted": count}


def ingest_current_game_odds(
    report_day: date,
    database_url: str,
    bookmakers: list[str],
    source: str,
) -> dict:
    errors: list[str] = []
    for candidate in GAME_ODDS_SOURCE_FALLBACKS[source]:
        try:
            game_lines_df = pd.DataFrame()
            if candidate == "scoresandodds":
                snapshots, game_lines_df = fetch_scoresandodds_game_frames(report_date=report_day)
            elif candidate == "espn":
                snapshots, game_lines_df = fetch_espn_game_frames(report_date=report_day)
            elif candidate == "sportsgameodds":
                snapshots, game_lines_df = fetch_sportsgameodds_game_frames(
                    report_date=report_day,
                    api_key=get_sportsgameodds_api_key(None),
                    bookmakers=bookmakers,
                )
            elif candidate == "the-odds-api":
                snapshots = fetch_current_game_odds_snapshots(
                    report_date=report_day,
                    api_key=get_the_odds_api_key(None),
                    bookmakers=bookmakers,
                )
            else:  # pragma: no cover - guarded by argparse choices
                raise RuntimeError(f"Unsupported game odds source: {candidate}")

            if snapshots.empty:
                raise RuntimeError(f"{candidate} returned no game odds snapshots for {report_day.isoformat()}")

            snapshot_count, closing_count = persist_game_odds(
                snapshots,
                snapshots_output_path=GAME_ODDS_SNAPSHOTS_CSV,
                closing_output_path=CLOSING_LINES_CSV,
                database_url=database_url,
            )
            game_line_rows = write_current_game_lines(game_lines_df, output_path=LATEST_GAME_LINES) if not game_lines_df.empty else 0
            return {
                "snapshot_rows": snapshot_count,
                "closing_rows": closing_count,
                "game_line_rows": game_line_rows,
                "source_used": candidate,
            }
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(" | ".join(errors) if errors else "No live game odds source succeeded")


def fetch_current_props_and_market_lines(
    report_day: date,
    dated_paths: dict[str, Path],
    source: str,
) -> dict:
    errors: list[str] = []
    rows_by_source: dict[str, list[dict]] = {}

    for path in (LATEST_ODDS_SLATE, LATEST_MARKET_LINES, dated_paths["raw_props"], dated_paths["market_lines"]):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    for candidate in PROPS_SOURCE_FALLBACKS[source]:
        try:
            if candidate == "scoresandodds":
                rows_by_source[candidate] = fetch_scoresandodds_prop_rows(
                    report_date=report_day,
                    allowed_markets=SUPPORTED_PROP_MARKETS,
                )
                if not rows_by_source[candidate]:
                    raise RuntimeError(f"{candidate} returned no prop rows for {report_day.isoformat()}")
                continue
            if candidate == "covers":
                rows_by_source[candidate] = fetch_covers_prop_rows(
                    report_date=report_day,
                    allowed_markets=SUPPORTED_PROP_MARKETS,
                )
                if not rows_by_source[candidate]:
                    raise RuntimeError(f"{candidate} returned no prop rows for {report_day.isoformat()}")
                continue
            if candidate == "sportsgameodds":
                rows_by_source[candidate] = fetch_sportsgameodds_prop_rows(
                    report_date=report_day,
                    allowed_markets=SUPPORTED_PROP_MARKETS,
                    api_key=get_sportsgameodds_api_key(None),
                    bookmakers=DEFAULT_BOOKMAKERS,
                )
                if not rows_by_source[candidate]:
                    raise RuntimeError(f"{candidate} returned no prop rows for {report_day.isoformat()}")
                continue
            if candidate == "the-odds-api":
                run_command(
                    [
                        *python_cmd("src/data/fetch_props_from_the_odds_api.py"),
                        "--markets",
                        "player_points,player_rebounds,player_assists,player_threes",
                    ]
                )
                rows_by_source[candidate] = pd.read_csv(LATEST_ODDS_SLATE).to_dict("records") if LATEST_ODDS_SLATE.exists() else []
                if not rows_by_source[candidate]:
                    raise RuntimeError(f"{candidate} returned no prop rows for {report_day.isoformat()}")
                continue
            raise RuntimeError(f"Unsupported props source: {candidate}")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    if not rows_by_source:
        raise RuntimeError(" | ".join(errors) if errors else "No live props source succeeded")

    merged_rows, sources_used = merge_prop_source_rows(
        rows_by_source,
        source_priority=PROPS_SOURCE_FALLBACKS[source],
    )
    write_prop_slate_csv(merged_rows, LATEST_ODDS_SLATE)
    rows_written = len(merged_rows)

    if LATEST_ODDS_SLATE.exists():
        dated_paths["raw_props"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LATEST_ODDS_SLATE, dated_paths["raw_props"])
    if LATEST_GAME_LINES.exists():
        dated_paths["game_lines"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LATEST_GAME_LINES, dated_paths["game_lines"])
    run_command(
        [
            *python_cmd("src/data/props_to_market_lines.py"),
            "--odds-slate",
            str(LATEST_ODDS_SLATE),
            "--output",
            str(dated_paths["market_lines"]),
            "--require-two-sided",
        ]
    )
    market_line_rows = 0
    if dated_paths["market_lines"].exists():
        market_lines_df = pd.read_csv(dated_paths["market_lines"])
        market_line_rows = int(len(market_lines_df))
    if market_line_rows > 0:
        shutil.copyfile(dated_paths["market_lines"], LATEST_MARKET_LINES)
    else:
        try:
            LATEST_MARKET_LINES.unlink()
        except FileNotFoundError:
            pass
    return {
        "raw_props_path": str(dated_paths["raw_props"]),
        "market_lines_path": str(dated_paths["market_lines"]),
        "rows_written": rows_written,
        "market_line_rows": market_line_rows,
        "source_used": sources_used[0],
        "sources_used": sources_used,
        "source_failures": errors,
    }


def score_and_materialize_live_props(report_day: date, database_url: str, dated_paths: dict[str, Path]) -> dict:
    market_lines_path = dated_paths.get("market_lines", LATEST_MARKET_LINES)

    for path in (dated_paths["edges"], LATEST_EDGES):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    if not market_lines_path.exists():
        return {"rows_materialized": 0, "status": "skipped_no_two_sided_prop_markets"}
    try:
        market_lines_df = pd.read_csv(market_lines_path)
    except pd.errors.EmptyDataError:
        return {"rows_materialized": 0, "status": "skipped_no_two_sided_prop_markets"}
    if market_lines_df.empty:
        return {"rows_materialized": 0, "status": "skipped_no_two_sided_prop_markets"}

    run_command(
        [
            *python_cmd("src/inference/scan_slate_with_model.py"),
            "--model-paths",
            "player_points=models/points_regression.pkl,player_rebounds=models/rebounds_regression.pkl,player_assists=models/assists_regression.pkl,player_threes=models/threes_regression.pkl",
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
            "--market-lines",
            str(LATEST_MARKET_LINES),
            "--output",
            str(dated_paths["edges"]),
            "--min-edge",
            "0.03",
        ]
    )

    if not dated_paths["edges"].exists():
        return {"rows_materialized": 0, "status": "skipped_no_live_prop_recommendations"}

    shutil.copyfile(dated_paths["edges"], LATEST_EDGES)
    scored_count, _ = materialize_edges(
        LATEST_EDGES,
        database_url=database_url,
        recommendation_origin="live_daily",
        persist_readiness=False,
    )
    return {"rows_materialized": scored_count}


def score_and_materialize_live_game_markets(report_day: date, database_url: str, bookmakers: list[str]) -> dict:
    models_dir = Path("models")
    available_models = [
        market
        for market in ("game_moneyline", "game_spread", "game_total")
        if (models_dir / f"{market}_model.pkl").exists()
    ]
    if not available_models:
        return {"rows_materialized": 0, "status": "skipped_no_game_market_models"}

    logs_df = pd.read_csv("data/player_game_logs.csv")
    odds_df = pd.read_csv(GAME_ODDS_SNAPSHOTS_CSV)
    injuries_df = pd.read_csv(OFFICIAL_INJURIES_CSV) if OFFICIAL_INJURIES_CSV.exists() else pd.DataFrame()
    starter_history_df = pd.read_csv(STARTER_HISTORY_CSV) if STARTER_HISTORY_CSV.exists() else pd.DataFrame()
    lineup_df = pd.read_csv(LINEUP_PROJECTIONS_CSV) if LINEUP_PROJECTIONS_CSV.exists() else pd.DataFrame()
    sportsbook = bookmakers[0] if bookmakers else None
    frame = build_game_market_recommendations(
        logs_df=logs_df,
        odds_snapshots_df=odds_df,
        models_dir=models_dir,
        sportsbook=sportsbook,
        target_date=report_day.isoformat(),
        min_edge=0.03,
        injuries_df=injuries_df,
        lineup_df=lineup_df,
        starter_history_df=starter_history_df,
    )
    if frame.empty:
        return {"rows_materialized": 0, "status": "skipped_no_game_market_recommendations"}
    frame["recommendation_origin"] = "live_daily"
    LATEST_GAME_MARKET_RECOMMENDATIONS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(LATEST_GAME_MARKET_RECOMMENDATIONS, index=False)
    scored_count, _ = materialize_edges(
        LATEST_GAME_MARKET_RECOMMENDATIONS,
        database_url=database_url,
        recommendation_origin="live_daily",
        persist_readiness=False,
    )
    return {"rows_materialized": scored_count}


def settle_and_refresh_readiness(database_url: str) -> dict:
    run_command(
        [
            *python_cmd("src/jobs/settle_recommendations.py"),
            "--logs-csv",
            "data/player_game_logs.csv",
            "--database-url",
            database_url,
        ]
    )
    run_command(
        [
            *python_cmd("src/jobs/build_market_readiness_snapshot.py"),
            "--database-url",
            database_url,
            "--models-dir",
            "models",
            "--output",
            str(LATEST_MARKET_READINESS),
        ]
    )
    return {"readiness_path": str(LATEST_MARKET_READINESS)}


def run_star_screener(dated_paths: dict[str, Path]) -> dict:
    run_command(
        [
            *python_cmd("src/inference/star_best_bets_screener.py"),
            "--auto-stars",
            "--odds-file",
            "data/odds_slate.csv",
            "--season-min",
            SEASON_MIN_FOR_FORM,
            "--min-line",
            MIN_LINE_FOR_STARS,
            "--max-stars",
            str(MAX_STARS),
            "--books",
            BOOKS,
            "--top-k",
            str(TOP_K),
            "--min-edge",
            MIN_EDGE_DISPLAY,
            "--ladder-thresholds",
            LADDER_THRESHOLDS,
            "--target-prob",
            TARGET_PROB,
        ],
        stdout_path=dated_paths["screener_log"],
    )
    return {"screener_log": str(dated_paths["screener_log"])}


def replay_historical_range(
    *,
    start_date: date,
    end_date: date,
    database_url: str,
    reset_cursor: bool,
) -> dict:
    command = [
        *python_cmd("src/jobs/replay_historical_recommendations.py"),
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        end_date.isoformat(),
        "--database-url",
        database_url,
        "--cursor-path",
        str(HISTORICAL_REPLAY_CURSOR),
    ]
    if reset_cursor:
        command.append("--reset-cursor")
    run_command(command)
    return {"cursor_path": str(HISTORICAL_REPLAY_CURSOR)}


def run_daily_mode(
    report_day: date,
    database_url: str,
    bookmakers: list[str],
    run_log: dict,
    *,
    game_odds_source: str,
    props_source: str,
    skip_star_screener: bool,
) -> None:
    dated_paths = current_log_paths(report_day)
    execute_step(run_log, "build_prop_feature_stack", build_prop_feature_stack, strict=True)
    execute_step(run_log, "ingest_official_injuries", lambda: ingest_official_injuries_for_day(report_day, database_url), strict=True)
    require_rows(OFFICIAL_INJURIES_CSV, report_day.isoformat(), "report_date", "game_date")
    execute_step(
        run_log,
        "refresh_starter_history",
        lambda: refresh_starter_history(database_url, report_day=report_day),
        strict=True,
    )
    execute_step(run_log, "build_lineup_projections", lambda: build_lineups_for_day(report_day, database_url), strict=True)
    require_rows(LINEUP_PROJECTIONS_CSV, report_day.isoformat(), "game_date")
    execute_step(
        run_log,
        "ingest_current_game_odds",
        lambda: ingest_current_game_odds(report_day, database_url, bookmakers, game_odds_source),
        strict=True,
    )
    require_rows(GAME_ODDS_SNAPSHOTS_CSV, report_day.isoformat(), "game_date")
    execute_step(
        run_log,
        "fetch_current_props",
        lambda: fetch_current_props_and_market_lines(report_day, dated_paths, props_source),
        strict=True,
    )
    execute_step(run_log, "score_materialize_live_props", lambda: score_and_materialize_live_props(report_day, database_url, dated_paths), strict=True)
    execute_step(run_log, "score_materialize_live_game_markets", lambda: score_and_materialize_live_game_markets(report_day, database_url, bookmakers), strict=True)
    execute_step(run_log, "settle_refresh_readiness", lambda: settle_and_refresh_readiness(database_url), strict=True)
    if not skip_star_screener:
        execute_step(run_log, "star_best_bets_screener", lambda: run_star_screener(dated_paths), strict=False)


def run_bootstrap_or_backfill_mode(
    *,
    mode: str,
    report_day: date,
    backfill_start: date,
    backfill_end: date,
    database_url: str,
    historical_manifest_path: Path,
    bookmakers: list[str],
    game_odds_source: str,
    props_source: str,
    run_log: dict,
    skip_prop_training: bool,
    skip_game_training: bool,
    publish_current_day_at_end: bool,
    reset_cursors: bool,
    skip_star_screener: bool,
) -> None:
    if mode == "bootstrap":
        execute_step(run_log, "init_database", lambda: {"database_url": str(init_database(database_url).url)}, strict=True)
    execute_step(
        run_log,
        "backfill_official_injuries",
        lambda: backfill_official_injuries(
            start_date=backfill_start,
            end_date=backfill_end,
            output_path=OFFICIAL_INJURIES_CSV,
            database_url=database_url,
            cursor_path=INJURY_BACKFILL_CURSOR,
            reset_cursor=reset_cursors,
        ),
        strict=False,
    )
    execute_step(run_log, "refresh_starter_history", lambda: refresh_starter_history(database_url), strict=False)
    execute_step(
        run_log,
        "import_historical_game_odds",
        lambda: import_historical_game_odds(
            manifest_path=historical_manifest_path,
            database_url=database_url,
        ),
        strict=False,
    )
    execute_step(
        run_log,
        "backfill_historical_market_data",
        lambda: backfill_historical_market_data(database_url=database_url),
        strict=False,
    )
    execute_step(run_log, "build_prop_feature_stack", build_prop_feature_stack, strict=False)
    if not skip_prop_training:
        execute_step(run_log, "train_prop_models", lambda: (train_prop_models(), {"trained": "prop_models"})[1], strict=False)
    if not skip_game_training:
        execute_step(run_log, "train_game_models", lambda: (train_game_models(database_url), {"trained": "game_models"})[1], strict=False)
    execute_step(
        run_log,
        "replay_historical_recommendations",
        lambda: replay_historical_range(
            start_date=backfill_start,
            end_date=backfill_end,
            database_url=database_url,
            reset_cursor=reset_cursors,
        ),
        strict=False,
    )
    execute_step(run_log, "settle_refresh_readiness", lambda: settle_and_refresh_readiness(database_url), strict=False)
    if mode == "bootstrap" and publish_current_day_at_end:
        run_daily_mode(
            report_day,
            database_url,
            bookmakers,
            run_log,
            game_odds_source=game_odds_source,
            props_source=props_source,
            skip_star_screener=skip_star_screener,
        )


def main() -> None:
    args = build_parser().parse_args()
    print(f"Running pipeline from: {PROJECT_ROOT}")
    if PROJECT_ROOT != Path.cwd():
        print(f"Changing working directory to: {PROJECT_ROOT}")
        os.chdir(PROJECT_ROOT)

    ensure_runtime_dirs()
    report_day = date.fromisoformat(args.report_date)
    default_start, default_end = default_backfill_window(report_day)
    backfill_start = date.fromisoformat(args.backfill_start_date) if args.backfill_start_date else default_start
    backfill_end = date.fromisoformat(args.backfill_end_date) if args.backfill_end_date else default_end
    database_url = get_database_url(args.database_url)
    historical_manifest_path = Path(args.historical_odds_manifest)
    bookmakers = [item.strip() for item in args.bookmakers.split(",") if item.strip()]

    run_log = {
        "mode": args.mode,
        "report_date": report_day.isoformat(),
        "backfill_start_date": backfill_start.isoformat(),
        "backfill_end_date": backfill_end.isoformat(),
        "database_url": database_url,
        "historical_odds_manifest": str(historical_manifest_path),
        "bookmakers": bookmakers,
        "game_odds_source": args.game_odds_source,
        "props_source": args.props_source,
        "started_at": utc_now_iso(),
        "steps": [],
        "warnings": [],
    }
    log_path = current_log_paths(report_day)["run_log"]

    try:
        if args.mode == "daily":
            run_daily_mode(
                report_day,
                database_url,
                bookmakers,
                run_log,
                game_odds_source=args.game_odds_source,
                props_source=args.props_source,
                skip_star_screener=bool(args.skip_star_screener),
            )
        else:
            run_bootstrap_or_backfill_mode(
                mode=args.mode,
                report_day=report_day,
                backfill_start=backfill_start,
                backfill_end=backfill_end,
                database_url=database_url,
                historical_manifest_path=historical_manifest_path,
                bookmakers=bookmakers,
                game_odds_source=args.game_odds_source,
                props_source=args.props_source,
                run_log=run_log,
                skip_prop_training=bool(args.skip_prop_training),
                skip_game_training=bool(args.skip_game_training),
                publish_current_day_at_end=bool(args.publish_current_day_at_end),
                reset_cursors=bool(args.reset_cursors),
                skip_star_screener=bool(args.skip_star_screener),
            )
        run_log["status"] = "completed"
    except Exception as exc:
        run_log["status"] = "failed"
        run_log["error"] = str(exc)
        raise
    finally:
        run_log["finished_at"] = utc_now_iso()
        write_json(log_path, run_log)
        print(f"[INFO] Wrote structured pipeline log: {log_path}")


if __name__ == "__main__":
    main()
