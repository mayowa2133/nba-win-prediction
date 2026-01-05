#!/usr/bin/env python
"""
run_full_slate_pipeline.py

One-click pipeline to:

  1) Update player game logs incrementally
  2) Rebuild player points features
  3) Train minutes regression model
  4) Retrain points regression model (default: XGBoost) with a fixed season split
  5) Train sigma (variance) model for points
  6) Refit over-probability calibrator
  7) Fetch fresh props from The Odds API into data/odds_slate.csv
  7b) Log raw props + build market lines (dated files)
  7c) Scan slate with model and write edges (dated file)
  7d) Join ALL logged market lines into features to create features_with_props
  8) Run star_best_bets_screener on the fresh slate (and log output)

Assumes:
  - You run this from the project root (where these scripts live)
  - Your virtualenv is already activated (so `python` is the right one)

Usage:
  python run_full_slate_pipeline.py
"""

import subprocess
import sys
import shutil
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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

FEATURES_CSV = Path("data/player_points_features.csv")
FEATURES_WITH_PROPS_CSV = Path("data/player_points_features_with_props.csv")
FEATURES_WITH_VEGAS_CSV = Path("data/player_points_features_with_vegas.csv")
FEATURES_WITH_INJURIES_CSV = Path("data/player_points_features_with_injuries.csv")  # NEW: final features with all enhancements
FEATURES_WITH_LINEUP_CSV = Path("data/player_points_features_with_lineup.csv")
INJURY_DATA_CSV = Path("data/injury_data.csv")


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


def main():
    # Just to be safe, run everything from the project root
    print(f"Running pipeline from: {PROJECT_ROOT}")
    if PROJECT_ROOT != Path.cwd():
        print(f"Changing working directory to: {PROJECT_ROOT}")
        try:
            Path.chdir(PROJECT_ROOT)  # type: ignore[attr-defined]
        except AttributeError:
            import os
            os.chdir(PROJECT_ROOT)

    # NEW: ensure dirs exist
    PROPS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROPS_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    GAME_LINES_DIR.mkdir(parents=True, exist_ok=True)

    # Use today's date stamp for logging (local machine time)
    today = datetime.now().date().isoformat()

    raw_props_dated = PROPS_RAW_DIR / f"odds_slate_{today}.csv"
    market_lines_dated = PROPS_MARKET_DIR / f"market_lines_{today}.csv"
    edges_dated = EDGES_DIR / f"edges_with_market_{today}.csv"
    screener_log = RUN_LOG_DIR / f"star_best_bets_{today}.txt"
    game_lines_dated = GAME_LINES_DIR / f"game_lines_{today}.csv"

    # 1) Update logs incrementally
    run(
        ["python", "src/pipeline/update_player_game_logs_incremental.py"],
        desc="Step 1/9: Updating player game logs incrementally",
    )

    # 2) Rebuild features
    run(
        ["python", "src/data/build_player_points_features.py"],
        desc="Step 2/9: Rebuilding player points features",
    )

    # 2a) Fetch/update injury data from game logs
    run(
        ["python", "src/data/fetch_injury_data.py"],
        desc="Step 2a/9: Fetching injury/availability data from game logs",
    )

    # 2b) Join injury data into features
    build_features_with_injuries(
        features_csv=FEATURES_CSV,
        injury_csv=INJURY_DATA_CSV,
        out_csv=FEATURES_WITH_INJURIES_CSV,
    )

    # 2c) Add lineup context features (teammates out / shorthandedness)
    build_features_with_lineup_context(
        features_csv=FEATURES_WITH_INJURIES_CSV,
        out_csv=FEATURES_WITH_LINEUP_CSV,
    )

    # 2d) Join ALL logged market_lines into the features table so props accumulate historically
    # Use FEATURES_WITH_LINEUP_CSV as input so we have injuries + lineup context + props
    build_features_with_props(
        features_csv=FEATURES_WITH_LINEUP_CSV,
        market_dir=PROPS_MARKET_DIR,
        out_csv=FEATURES_WITH_PROPS_CSV,
    )

    # 2d) Join ALL logged game_lines into the features table for Vegas lines features
    # Use FEATURES_WITH_PROPS_CSV as input so final file has injuries + props + Vegas
    build_features_with_vegas_lines(
        features_csv=FEATURES_WITH_PROPS_CSV,
        game_lines_dir=GAME_LINES_DIR,
        out_csv=FEATURES_WITH_VEGAS_CSV,
    )

    # 3) Train minutes regression model
    run(
        [
            "python",
            "src/models/build_minutes_regression.py",
            "--model-type",
            MODEL_TYPE,
            "--train-max-season",
            TRAIN_MAX_SEASON,
            "--val-min-season",
            VAL_MIN_SEASON,
        ],
        desc=(
            f"Step 3/9: Training minutes regression model "
            f"(model_type={MODEL_TYPE}, train <= {TRAIN_MAX_SEASON}, val >= {VAL_MIN_SEASON})"
        ),
    )

    # 4) Retrain points regression model WITH prop features and minutes prediction enabled
    # Use FEATURES_WITH_VEGAS_CSV which contains both props and Vegas lines
    run(
        [
            "python",
            "src/models/build_points_regression.py",
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
        ],
        desc=(
            f"Step 4/9: Training points regression model with prop features + minutes prediction "
            f"(model_type={MODEL_TYPE}, train <= {TRAIN_MAX_SEASON}, val >= {VAL_MIN_SEASON})"
        ),
    )

    # 4b) Train quantile regression models (Phase 4B: 10th, 50th, 90th percentiles)
    run(
        [
            "python",
            "src/models/build_points_regression_quantile.py",
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
            "--train-max-season",
            TRAIN_MAX_SEASON,
            "--val-min-season",
            VAL_MIN_SEASON,
        ],
        desc=(
            f"Step 4b/9: Training quantile regression models (10th, 50th, 90th percentiles) "
            f"(train <= {TRAIN_MAX_SEASON}, val >= {VAL_MIN_SEASON})"
        ),
    )

    # 4c) Train tiered models (separate models for each player tier)
    run(
        [
            "python",
            "src/models/build_points_regression_tiered.py",
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
            str(int(N_TUNE_ITER) // 2),  # Fewer iterations per tier
        ],
        desc=(
            f"Step 4c/9: Training tiered models (tiers 0-3) "
            f"(model_type={MODEL_TYPE}, train <= {TRAIN_MAX_SEASON}, val >= {VAL_MIN_SEASON})"
        ),
    )

    # 5) Train sigma model (heteroscedastic variance)
    # Use FEATURES_WITH_VEGAS_CSV to match the features used in main model
    run(
        [
            "python",
            "src/models/build_points_sigma_model.py",
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
            "--train-max-season",
            TRAIN_MAX_SEASON,
        ],
        desc="Step 5/9: Training sigma (variance) model for player points",
    )

    # 6) Refit over/under probability calibrator
    # Use FEATURES_WITH_VEGAS_CSV to match the features used in main model
    run(
        [
            "python",
            "src/models/build_over_prob_calibrator.py",
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
        ],
        desc="Step 6/9: Fitting over-probability calibrator on recent seasons",
    )

    # 7) Fetch fresh props (points+reb+ast+3PM) AND game lines (keeps your existing default output: data/odds_slate.csv)
    run(
        [
            "python",
            "src/data/fetch_props_from_the_odds_api.py",
            "--markets",
            "player_points,player_rebounds,player_assists,player_threes",
        ],
        desc="Step 7/9: Fetching fresh props and game lines into data/odds_slate.csv and data/game_lines.csv",
    )

    # 7b) Log raw odds slate to dated file
    if LATEST_ODDS_SLATE.exists():
        shutil.copyfile(LATEST_ODDS_SLATE, raw_props_dated)
        print(f"[INFO] Saved dated raw props: {raw_props_dated}")
    else:
        print(f"[WARN] Expected {LATEST_ODDS_SLATE} but it does not exist. Skipping dated raw props save.")

    # 7b2) Log game lines to dated file (NEW: for Vegas lines features)
    if LATEST_GAME_LINES.exists():
        shutil.copyfile(LATEST_GAME_LINES, game_lines_dated)
        print(f"[INFO] Saved dated game lines: {game_lines_dated}")
    else:
        print(f"[WARN] Expected {LATEST_GAME_LINES} but it does not exist. Skipping dated game lines save.")

    # 7c) Build market lines (ALL markets) from the latest odds slate
    run(
        [
            "python",
            "src/data/props_to_market_lines.py",
            "--odds-slate",
            str(LATEST_ODDS_SLATE),
            "--output",
            str(market_lines_dated),
        ],
        desc="Step 7b/9: Aggregating raw props into market lines (dated file)",
    )

    # Keep a “latest” market_lines.csv for other scripts
    if market_lines_dated.exists():
        shutil.copyfile(market_lines_dated, LATEST_MARKET_LINES)
        print(f"[INFO] Updated latest market lines: {LATEST_MARKET_LINES}")

    # 7d) Scan slate with per-market models vs market to produce edges (dated + latest)
    # Uses:
    #  - points: models/points_regression.pkl (and optional points-only sigma/calibrator if enabled)
    #  - rebounds: models/rebounds_regression.pkl
    #  - assists: models/assists_regression.pkl
    #  - threes: models/threes_regression.pkl
    run(
        [
            "python",
            "src/inference/scan_slate_with_model.py",
            "--model-paths",
            "player_points=models/points_regression.pkl,player_rebounds=models/rebounds_regression.pkl,player_assists=models/assists_regression.pkl,player_threes=models/threes_regression.pkl",
            "--features-csv",
            str(FEATURES_WITH_VEGAS_CSV),
            "--market-lines",
            str(LATEST_MARKET_LINES),
            "--output",
            str(edges_dated),
            "--min-edge",
            "0.03",
        ],
        desc="Step 7c/9: Scanning slate with per-market models to compute edges (dated file)",
    )

    if edges_dated.exists():
        shutil.copyfile(edges_dated, LATEST_EDGES)
        print(f"[INFO] Updated latest edges file: {LATEST_EDGES}")

    # Note: Props and Vegas lines are now joined BEFORE model training (steps 2a-2b)
    # This ensures they're available for training. The joins above are kept for
    # historical accumulation but the main training uses the files created earlier.

    # 8) Run star best bets screener (and log output)
    run(
        [
            "python",
            "src/inference/star_best_bets_screener.py",
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
        desc="Step 8/9: Running star_best_bets_screener on the latest slate",
        stdout_path=screener_log,
    )

    print(f"\n[INFO] Screener output saved to: {screener_log}")
    print("\n✅ Pipeline completed successfully.\n")


if __name__ == "__main__":
    main()