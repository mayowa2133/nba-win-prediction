#!/usr/bin/env python
"""
scan_slate_with_model.py

UPDATED:
- Recomputes matchup-dependent features for the UPCOMING game:
    is_home
    days_since_last_game
    is_b2b
    is_long_rest
    opp_pts_allowed_roll5/15
    opp_dvp_pos_pts_roll5/15   (requires player positions mapping)

Everything else is carried forward from the player's most recent prior game row,
which is correct for "as-of last game" inference.

FIXES:
- Properly skips a prop row if required matchup features can't be recomputed.
- Adds a staleness guard so you don't infer off a player's last game from months ago.
- Guards against negative rest_days (bad date ordering / mismatch).

QUALITY-OF-LIFE:
- Optional --debug-skips to show why rows were skipped.
- Adds extra debug columns (opp_abbrev_used, pos_bucket_used, days_stale_used, last_game_date_used).
"""

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

MINUTES_MODEL_PATH = Path("models/minutes_regression.pkl")
SIGMA_MODEL_PATH = Path("models/points_sigma_model.pkl")
CALIBRATOR_PATH = Path("models/over_prob_calibrator.pkl")

# Default per-market model bundles (mean models)
DEFAULT_MODEL_PATHS = {
    "player_points": "models/points_regression.pkl",
    "player_rebounds": "models/rebounds_regression.pkl",
    "player_assists": "models/assists_regression.pkl",
    "player_threes": "models/threes_regression.pkl",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    for ch in [".", ",", "'", "\"", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def american_to_prob(odds: float) -> float:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return float("nan")
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and math.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def load_sigma_model_bundle(path: Path):
    """
    Load heteroscedastic sigma model bundle (if present).

    Expected:
      {
        "model": <regressor>,
        "feature_cols": [...],
        "config": { "use_log_target": bool, "eps": float, "sigma_scale": float }
      }
    """
    if not path.exists():
        return None, None, {}
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        # Old style: just the model
        return bundle, None, {"use_log_target": False, "eps": 1e-3, "sigma_scale": 1.0}

    sigma_model = bundle.get("model") or bundle.get("sigma_model") or bundle.get("regressor")
    sigma_cols = bundle.get("feature_cols")
    cfg = bundle.get("config") if isinstance(bundle.get("config"), dict) else {}
    cfg_out = {
        "use_log_target": bool(cfg.get("use_log_target", bundle.get("use_log_target", False))),
        "eps": float(cfg.get("eps", bundle.get("eps", 1e-3))),
        "sigma_scale": float(cfg.get("sigma_scale", bundle.get("sigma_scale", 1.0))),
    }
    return sigma_model, sigma_cols, cfg_out


def load_over_prob_calibrator(path: Path):
    """
    Load isotonic regression calibrator bundle (if present).
    Expected: {"calibrator": <IsotonicRegression>, "info": {...}}
    """
    if not path.exists():
        return None, {}
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict) and "calibrator" in bundle:
        return bundle["calibrator"], bundle.get("info", {})
    # Backwards compat: bundle itself is calibrator
    return bundle, {}


# ---------------------------------------------------------------------
# NBA team mapping helpers (Odds API uses full names)
# ---------------------------------------------------------------------

NBA_FULL_TO_ABBR = {
    # East
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "detroit pistons": "DET",
    "indiana pacers": "IND",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "new york knicks": "NYK",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "toronto raptors": "TOR",
    "washington wizards": "WAS",
    # West
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "oklahoma city thunder": "OKC",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "utah jazz": "UTA",
    # Common alternate strings (just in case)
    "okc thunder": "OKC",
    "la clippers ": "LAC",
    "la lakers ": "LAL",
}


def full_team_to_abbr(team_full: str) -> Optional[str]:
    key = normalize_name(team_full)
    return NBA_FULL_TO_ABBR.get(key)


# ---------------------------------------------------------------------
# Minutes helper
# ---------------------------------------------------------------------

def add_minutes_pred_feature(df: pd.DataFrame, minutes_model_path: Path) -> bool:
    if not minutes_model_path.exists():
        print(f"[WARN] Minutes model not found at {minutes_model_path}; cannot add minutes_pred.")
        return False

    try:
        with open(minutes_model_path, "rb") as f:
            minutes_bundle = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load minutes model from {minutes_model_path}: {e}")
        return False

    minutes_model = minutes_bundle.get("model")
    minutes_feature_cols = minutes_bundle.get("feature_cols")

    if minutes_model is None or minutes_feature_cols is None:
        print("[WARN] Minutes model bundle missing 'model' or 'feature_cols'; cannot add minutes_pred.")
        return False

    missing = [c for c in minutes_feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Some minutes feature columns missing in df: {missing}; cannot add minutes_pred.")
        return False

    X_min = df[minutes_feature_cols].to_numpy()
    print(f"[INFO] Computing minutes_pred for {len(df):,} rows using minutes model...")
    df["minutes_pred"] = minutes_model.predict(X_min)
    print("[INFO] Added 'minutes_pred' column to features DataFrame.")
    return True


# ---------------------------------------------------------------------
# Build "as-of" lookup tables for matchup features
# ---------------------------------------------------------------------

def normalize_position(pos: str) -> Optional[str]:
    """
    Map common position encodings to a small set used for DvP.
    Adjust if your training used different buckets.
    """
    if not isinstance(pos, str) or not pos.strip():
        return None
    p = pos.strip().upper()

    # Examples: PG, SG, SF, PF, C, G, F, G-F, F-C, etc.
    if "PG" in p or "SG" in p or p.startswith("G"):
        return "G"
    if "SF" in p or "PF" in p or p.startswith("F"):
        return "F"
    if "C" in p:
        return "C"

    # fallback: first letter
    if p[0] in ("G", "F", "C"):
        return p[0]
    return None


def load_player_positions(player_positions_csv: Path) -> Dict[int, str]:
    """
    Expected columns (any of these will work):
      - player_id + position
      - player_id + pos
      - player_id + primary_position
    """
    if not player_positions_csv.exists():
        print(f"[WARN] player_positions file not found: {player_positions_csv}")
        return {}

    df = pd.read_csv(player_positions_csv)
    if "player_id" not in df.columns:
        print("[WARN] player_positions.csv missing player_id; cannot use it.")
        return {}

    pos_col = None
    for c in ["position", "pos", "primary_position"]:
        if c in df.columns:
            pos_col = c
            break
    if pos_col is None:
        print("[WARN] player_positions.csv missing a position column; expected one of: position/pos/primary_position")
        return {}

    out: Dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            pid = int(r["player_id"])
        except Exception:
            continue
        p = normalize_position(r[pos_col])
        if p:
            out[pid] = p
    print(f"[INFO] Loaded positions for {len(out):,} players from {player_positions_csv}")
    return out


def build_team_game_log(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (game_id, team_abbrev) with team points scored.
    Then attach points allowed by looking up opponent points scored.
    """
    needed = ["game_id", "game_date_ts", "team_abbrev", "opp_abbrev", "target_pts"]
    missing = [c for c in needed if c not in df_features.columns]
    if missing:
        raise ValueError(f"Features missing required columns for team log: {missing}")

    team_pts = (
        df_features.groupby(["game_id", "game_date_ts", "team_abbrev", "opp_abbrev"], as_index=False)["target_pts"]
        .sum()
        .rename(columns={"target_pts": "team_pts_scored"})
    )

    opp_pts = team_pts.rename(
        columns={
            "team_abbrev": "opp_team",
            "opp_abbrev": "team_abbrev",
            "team_pts_scored": "opp_pts_scored",
        }
    )[["game_id", "team_abbrev", "opp_team", "opp_pts_scored"]]

    merged = team_pts.merge(
        opp_pts,
        left_on=["game_id", "team_abbrev", "opp_abbrev"],
        right_on=["game_id", "team_abbrev", "opp_team"],
        how="left",
    )

    merged = merged.drop(columns=["opp_team"])
    merged = merged.rename(columns={"opp_abbrev": "opp_abbrev_in_game"})
    merged["team_pts_allowed"] = merged["opp_pts_scored"]
    merged = merged.drop(columns=["opp_pts_scored"])

    return merged.sort_values(["team_abbrev", "game_date_ts"])


def build_team_allowed_roll_asof(team_game_log: pd.DataFrame, windows=(5, 15)) -> pd.DataFrame:
    """
    For each team, compute rolling mean of points allowed over last N games
    (excluding current game via shift(1)).
    """
    df = team_game_log.copy()
    df = df.sort_values(["team_abbrev", "game_date_ts"])
    g = df.groupby("team_abbrev", group_keys=False)

    for w in windows:
        col = f"team_pts_allowed_roll{w}"
        df[col] = g["team_pts_allowed"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    return df[["team_abbrev", "game_date_ts"] + [f"team_pts_allowed_roll{w}" for w in windows]]


def build_dvp_pos_roll_asof(
    df_features: pd.DataFrame,
    player_id_to_pos: Dict[int, str],
    windows=(5, 15),
) -> pd.DataFrame:
    """
    DvP-by-position:
      For each defense team and position, build per-game points allowed to that position,
      then rolling mean over last N games (excluding current game).
    """
    if not player_id_to_pos:
        return pd.DataFrame()

    needed = ["game_id", "game_date_ts", "opp_abbrev", "player_id", "target_pts"]
    missing = [c for c in needed if c not in df_features.columns]
    if missing:
        raise ValueError(f"Features missing required columns for DvP: {missing}")

    tmp = df_features[needed].copy()
    tmp["pos_bucket"] = tmp["player_id"].map(player_id_to_pos)
    tmp = tmp.dropna(subset=["pos_bucket"])

    dvp_game = (
        tmp.groupby(["opp_abbrev", "pos_bucket", "game_id", "game_date_ts"], as_index=False)["target_pts"]
        .sum()
        .rename(columns={"opp_abbrev": "def_team", "target_pts": "pts_allowed_to_pos"})
        .sort_values(["def_team", "pos_bucket", "game_date_ts"])
    )

    g = dvp_game.groupby(["def_team", "pos_bucket"], group_keys=False)
    for w in windows:
        col = f"dvp_pos_pts_roll{w}"
        dvp_game[col] = g["pts_allowed_to_pos"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    keep_cols = ["def_team", "pos_bucket", "game_date_ts"] + [f"dvp_pos_pts_roll{w}" for w in windows]
    return dvp_game[keep_cols]


def asof_lookup(
    df: pd.DataFrame,
    key_cols: List[str],
    date_col: str,
    value_cols: List[str],
) -> Dict[Tuple, Tuple[np.ndarray, np.ndarray]]:
    """
    Build a dict for fast "as-of" lookup:
      key -> (dates_sorted, values_matrix)
    """
    out: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
    if df.empty:
        return out

    df = df.sort_values(key_cols + [date_col])
    for key, sub in df.groupby(key_cols):
        dates = sub[date_col].to_numpy()
        vals = sub[value_cols].to_numpy()
        out[(key,) if not isinstance(key, tuple) else key] = (dates, vals)
    return out


def asof_get(
    table: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    key: Tuple,
    when: pd.Timestamp,
) -> Optional[np.ndarray]:
    if key not in table:
        return None
    dates, vals = table[key]
    idx = np.searchsorted(dates, np.datetime64(when), side="left") - 1
    if idx < 0:
        return None
    return vals[idx]


# ---------------------------------------------------------------------
# Model/data loading
# ---------------------------------------------------------------------

def load_model_bundle(model_path: Path) -> Dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if "model" not in bundle or "feature_cols" not in bundle or "sigma" not in bundle:
        raise ValueError("Model bundle must contain 'model', 'feature_cols', and 'sigma'")
    return bundle


def load_features(features_csv: Path) -> pd.DataFrame:
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    df = pd.read_csv(features_csv)
    if "player_name" not in df.columns or "game_date" not in df.columns:
        raise ValueError("Expected 'player_name' and 'game_date' columns in features CSV.")

    df = df.copy()
    df["player_name_norm"] = df["player_name"].map(normalize_name)
    df["game_date_ts"] = pd.to_datetime(df["game_date"], errors="coerce").dt.tz_localize(None)
    return df


def load_market_lines(market_lines_csv: Path) -> pd.DataFrame:
    if not market_lines_csv.exists():
        raise FileNotFoundError(f"Market lines CSV not found: {market_lines_csv}")

    df = pd.read_csv(market_lines_csv)
    required = ["player", "prop_pts_line", "over_odds_best", "under_odds_best", "game_date", "home_team", "away_team", "market_key"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"market_lines.csv is missing required columns: {missing}")

    df = df.copy()
    df["player_norm"] = df["player"].map(normalize_name)
    df["game_date_ts"] = pd.to_datetime(df["game_date"], errors="coerce").dt.tz_localize(None)
    return df


def find_latest_feature_row(
    player_features: pd.DataFrame,
    player_norm: str,
    game_date_ts: pd.Timestamp
) -> Optional[pd.Series]:
    sub = player_features[player_features["player_name_norm"] == player_norm]
    if sub.empty:
        return None
    sub = sub[sub["game_date_ts"] < game_date_ts]
    if sub.empty:
        return None
    sub = sub.sort_values("game_date_ts", ascending=False)
    return sub.iloc[0]


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def compute_matchup_overrides(
    feat_row: pd.Series,
    market_row: pd.Series,
    allowed_asof: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    dvp_asof: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    player_id_to_pos: Dict[int, str],
    game_lines_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """
    Returns:
      (overrides, debug_meta)

    debug_meta includes:
      - opp_abbr_used
      - pos_bucket_used
    """
    overrides: Dict[str, float] = {}
    meta: Dict[str, object] = {"opp_abbr_used": None, "pos_bucket_used": None}

    home_abbr = full_team_to_abbr(market_row["home_team"])
    away_abbr = full_team_to_abbr(market_row["away_team"])
    if home_abbr is None or away_abbr is None:
        return {}, meta

    if "team_abbrev" not in feat_row.index:
        return {}, meta

    player_team = str(feat_row["team_abbrev"])
    if player_team == home_abbr:
        is_home = 1.0
        opp_abbr = away_abbr
    elif player_team == away_abbr:
        is_home = 0.0
        opp_abbr = home_abbr
    else:
        return {}, meta

    overrides["is_home"] = is_home
    meta["opp_abbr_used"] = opp_abbr

    last_game_date = feat_row["game_date_ts"]
    game_date = market_row["game_date_ts"]
    if pd.isna(last_game_date) or pd.isna(game_date):
        return {}, meta

    rest_days = int((game_date - last_game_date).days)
    if rest_days < 0:
        return {}, meta

    overrides["days_since_last_game"] = float(rest_days)
    overrides["is_b2b"] = 1.0 if rest_days == 1 else 0.0
    overrides["is_long_rest"] = 1.0 if rest_days >= 3 else 0.0

    v = asof_get(allowed_asof, (opp_abbr,), game_date)
    if v is not None and len(v) >= 2:
        overrides["opp_pts_allowed_roll5"] = float(v[0])
        overrides["opp_pts_allowed_roll15"] = float(v[1])

    pid = None
    if "player_id" in feat_row.index:
        try:
            pid = int(feat_row["player_id"])
        except Exception:
            pid = None

    pos_bucket = player_id_to_pos.get(pid) if pid is not None else None
    meta["pos_bucket_used"] = pos_bucket

    if pos_bucket and dvp_asof:
        vv = asof_get(dvp_asof, (opp_abbr, pos_bucket), game_date)
        if vv is not None and len(vv) >= 2:
            overrides["opp_dvp_pos_pts_roll5"] = float(vv[0])
            overrides["opp_dvp_pos_pts_roll15"] = float(vv[1])

    # Add Vegas lines features if available
    if game_lines_df is not None and not game_lines_df.empty:
        game_date_only = game_date.date() if hasattr(game_date, 'date') else game_date
        if isinstance(game_date_only, pd.Timestamp):
            game_date_only = game_date_only.date()
        
        # Match by game_date and team (home or away)
        match = game_lines_df[
            (game_lines_df["game_date"] == game_date_only) &
            (
                (game_lines_df["home_team"].str.contains(home_abbr, case=False, na=False)) |
                (game_lines_df["away_team"].str.contains(away_abbr, case=False, na=False))
            )
        ]
        
        if not match.empty:
            gl_row = match.iloc[0]
            if pd.notna(gl_row.get("vegas_game_total")):
                overrides["vegas_game_total"] = float(gl_row["vegas_game_total"])
            if pd.notna(gl_row.get("vegas_abs_spread")):
                overrides["vegas_abs_spread"] = float(gl_row["vegas_abs_spread"])
            
            # Spread: use home spread if player is home, away spread if away
            if is_home == 1.0 and pd.notna(gl_row.get("vegas_home_spread")):
                overrides["vegas_spread"] = float(gl_row["vegas_home_spread"])
            elif is_home == 0.0 and pd.notna(gl_row.get("vegas_away_spread")):
                overrides["vegas_spread"] = float(gl_row["vegas_away_spread"])

    return overrides, meta


def evaluate_slate(
    df_market: pd.DataFrame,
    df_features: pd.DataFrame,
    model,
    feature_cols: List[str],
    sigma: float,
    min_edge: float,
    allowed_asof: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    dvp_asof: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    player_id_to_pos: Dict[int, str],
    max_stale_days: int,
    debug_skips: bool,
    game_lines_df: Optional[pd.DataFrame] = None,
    use_tiered_models: bool = False,
    tier_models: Optional[Dict[int, Dict]] = None,
) -> pd.DataFrame:
    # Initialize optional model toggles if not set by main()
    if not hasattr(evaluate_slate, "_use_quantile"):
        evaluate_slate._use_quantile = False
    if not hasattr(evaluate_slate, "_use_sigma_model"):
        evaluate_slate._use_sigma_model = False
    if not hasattr(evaluate_slate, "_use_calibrator"):
        evaluate_slate._use_calibrator = False
    if not hasattr(evaluate_slate, "_use_tiered_ensemble"):
        evaluate_slate._use_tiered_ensemble = False

    records = []

    # Which matchup features does the model actually require?
    # (If a model doesn't use some of these, we won't force recompute.)
    must_keys = [
        "is_home",
        "days_since_last_game",
        "is_b2b",
        "is_long_rest",
        "opp_pts_allowed_roll5",
        "opp_pts_allowed_roll15",
        "opp_dvp_pos_pts_roll5",
        "opp_dvp_pos_pts_roll15",
    ]
    skipped_counts: Dict[str, int] = {}

    def skip(reason: str):
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

    for _, row in df_market.iterrows():
        market_key = str(row.get("market_key", "player_points"))
        per_market = getattr(evaluate_slate, "_per_market_models", None) or {}
        if market_key not in per_market:
            skip(f"no_model_for_{market_key}")
            continue

        model_bundle = per_market[market_key]
        model = model_bundle["model"]
        feature_cols = model_bundle["feature_cols"]
        sigma = float(model_bundle.get("sigma", 6.0))

        # Recompute required matchup features only if this market's model expects them
        required = [k for k in must_keys if k in feature_cols]

        player_norm = row["player_norm"]
        game_date_ts = row["game_date_ts"]

        feat_row = find_latest_feature_row(df_features, player_norm, game_date_ts)
        if feat_row is None:
            skip("no_feature_row")
            continue

        last_game_ts = feat_row["game_date_ts"]
        if pd.isna(last_game_ts) or pd.isna(game_date_ts):
            skip("bad_dates")
            continue

        days_stale = int((game_date_ts - last_game_ts).days)
        if days_stale < 0:
            skip("negative_stale_days")
            continue
        if days_stale > int(max_stale_days):
            skip("too_stale")
            continue

        overrides, meta = compute_matchup_overrides(
            feat_row=feat_row,
            market_row=row,
            allowed_asof=allowed_asof,
            dvp_asof=dvp_asof,
            player_id_to_pos=player_id_to_pos,
            game_lines_df=game_lines_df,
        )

        # Require recomputed matchup features if model expects them.
        # Allow missing DvP if positions are not loaded OR if this player has no pos bucket.
        skip_row = False
        for k in required:
            if k in overrides:
                continue

            if k.startswith("opp_dvp_pos"):
                if not player_id_to_pos:
                    continue
                if meta.get("pos_bucket_used") is None:
                    # If you’d rather *skip* when pos missing, flip this to skip_row=True.
                    continue

            # anything else: can't safely compute tonight's matchup value
            skip_row = True
            skip(f"missing_{k}")
            break

        if skip_row:
            continue

        # Create feature vector from last row + overrides
        x_vals: List[float] = []
        for c in feature_cols:
            if c in overrides:
                x_vals.append(float(overrides[c]))
            else:
                if c not in feat_row.index:
                    # Handle missing Vegas features gracefully (for backward compatibility)
                    if c in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
                        x_vals.append(0.0)
                    # Handle missing injury features gracefully
                    elif c == "is_injured":
                        x_vals.append(0)  # Assume healthy if not available
                    elif c == "days_since_last_dnp":
                        x_vals.append(999)  # Never injured if not available
                    elif c == "dnp_count_last_10":
                        x_vals.append(0)  # No DNPs if not available
                    # Handle missing prop features gracefully
                    elif c in ["prop_pts_line", "prop_over_odds_best", "prop_under_odds_best", "has_prop_line",
                               "prop_minus_pts_roll5", "prop_minus_pts_roll15", "prop_minus_season_mean",
                               "prop_minus_career_mean", "prop_minus_model_baseline"]:
                        if c == "has_prop_line":
                            x_vals.append(0.0)
                        else:
                            x_vals.append(0.0)
                    # Handle missing minutes_pred (will be computed if needed)
                    elif c == "minutes_pred":
                        # This should be computed earlier, but if missing, use 0
                        x_vals.append(0.0)
                    # Handle missing Phase 4A features gracefully
                    elif c in ["blowout_prob", "is_likely_blowout", "garbage_time_minutes_est", "vegas_spread_abs_normalized"]:
                        x_vals.append(0.0)
                    elif c in ["player_vs_opp_pts_avg_career", "player_vs_opp_pts_avg_last_5",
                               "player_vs_opp_minutes_avg_career", "player_vs_opp_minutes_avg_last_5",
                               "player_vs_opp_games_count"]:
                        x_vals.append(0.0)
                    elif c in ["opp_fg_pct_allowed_vs_pos_roll5", "opp_fg_pct_allowed_vs_pos_roll15",
                               "opp_3pt_pct_allowed_vs_pos_roll5", "opp_3pt_pct_allowed_vs_pos_roll15"]:
                        # Use league averages
                        if "fg_pct" in c:
                            x_vals.append(0.45)
                        elif "3pt_pct" in c:
                            x_vals.append(0.35)
                        else:
                            x_vals.append(0.0)
                    # Phase 4B: Lineup context features
                    elif c in [
                        "teammate_out_count",
                        "teammate_out_star_count",
                        "teammate_out_usg15_sum",
                        "teammate_out_min15_sum",
                        "team_available_players",
                        "is_team_shorthanded",
                    ]:
                        x_vals.append(0.0)
                    else:
                        raise ValueError(f"Feature row missing expected column: {c}")
                else:
                    x_vals.append(float(feat_row[c]))

        x = np.array(x_vals, dtype=float).reshape(1, -1)
        
        # Use quantile ensemble if enabled (check via closure variable)
        use_quantile = getattr(evaluate_slate, '_use_quantile', False)
        use_tier_ens = getattr(evaluate_slate, "_use_tiered_ensemble", False)
        
        is_points = market_key == "player_points"

        # Points-only advanced routing (quantiles / tiering / ensembles).
        # For other markets, always use the per-market unified model.
        if (not is_points):
            mu = float(model.predict(x)[0])
        elif use_quantile:
            from src.utils.load_quantile_model import predict_with_quantile_ensemble
            
            mu, quantile_used = predict_with_quantile_ensemble(
                X=x,
                feature_cols=feature_cols,
                quantile_models=evaluate_slate._quantile_models,
                base_model=evaluate_slate._base_model,
            )
            quantile_used = str(quantile_used)
        elif use_tier_ens:
            # Blend tiered + unified predictions
            w = float(getattr(evaluate_slate, "_ensemble_weight", 0.45))
            w = max(0.0, min(1.0, w))
            tier_models_local = getattr(evaluate_slate, "_tier_models", None) or {}

            # Unified mu
            mu_u = float(model.predict(x)[0])

            # Tier selection from latest feature row (as-of last game)
            player_tier = int(feat_row.get("star_tier_pts", 1)) if "star_tier_pts" in feat_row.index else 1
            player_tier = max(0, min(3, player_tier))

            mu_t = mu_u
            tb = tier_models_local.get(player_tier)
            if tb is not None:
                cols_t = tb.get("feature_cols", [])
                # cols_t are unified feature cols minus star_tier_pts (by construction)
                feat_map = {c: float(v) for c, v in zip(feature_cols, x_vals)}
                x_t_vals: List[float] = []
                for c in cols_t:
                    if c in feat_map:
                        x_t_vals.append(_safe_float(feat_map[c], 0.0))
                    else:
                        # defaults
                        if c in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
                            x_t_vals.append(0.0)
                        elif c == "is_injured":
                            x_t_vals.append(0.0)
                        elif c == "days_since_last_dnp":
                            x_t_vals.append(999.0)
                        elif c == "dnp_count_last_10":
                            x_t_vals.append(0.0)
                        elif c == "has_prop_line":
                            x_t_vals.append(0.0)
                        elif "fg_pct" in c:
                            x_t_vals.append(0.45)
                        elif "3pt_pct" in c:
                            x_t_vals.append(0.35)
                        else:
                            x_t_vals.append(0.0)
                try:
                    mu_t = float(tb["model"].predict(np.array(x_t_vals, dtype=float).reshape(1, -1))[0])
                except Exception:
                    mu_t = mu_u

            mu = w * mu_t + (1.0 - w) * mu_u
        # Use tiered model if enabled and available
        elif use_tiered_models and tier_models is not None:
            # Get player's tier from feature row
            player_tier = int(feat_row.get("star_tier_pts", 1)) if "star_tier_pts" in feat_row.index else 1
            player_tier = max(0, min(3, player_tier))  # Clamp to valid range
            
            tier_bundle = tier_models.get(player_tier)
            if tier_bundle is not None:
                tier_model = tier_bundle["model"]
                tier_sigma = float(tier_bundle.get("sigma", sigma))
                tier_feature_cols = tier_bundle.get("feature_cols", feature_cols)
                
                # Ensure feature vector matches tier model's features
                if len(tier_feature_cols) == len(x_vals):
                    mu = float(tier_model.predict(x)[0])
                else:
                    # Fallback to unified model if feature mismatch
                    mu = float(model.predict(x)[0])
            else:
                # Fallback to unified model
                mu = float(model.predict(x)[0])
        else:
            # Use unified model
            mu = float(model.predict(x)[0])

        # ------------------------------------------------------------------
        # Sigma: global vs heteroscedastic sigma model (optional, points only)
        # ------------------------------------------------------------------
        sigma_eff = sigma if sigma > 0 else 1e-6
        sigma_raw_pred = float("nan")
        use_sigma_model = getattr(evaluate_slate, "_use_sigma_model", False)
        if is_points and use_sigma_model and getattr(evaluate_slate, "_sigma_model", None) is not None:
            sm = evaluate_slate._sigma_model
            sigma_cols = getattr(evaluate_slate, "_sigma_feature_cols", None)
            sigma_cfg = getattr(evaluate_slate, "_sigma_cfg", {}) or {}
            eps = float(sigma_cfg.get("eps", 1e-3))
            scale = float(sigma_cfg.get("sigma_scale", 1.0))
            use_log = bool(sigma_cfg.get("use_log_target", False))

            if sigma_cols is None:
                sigma_cols = feature_cols + ["mu_hat"]

            # Fast path: expected bundle shape from build_points_sigma_model.py
            if sigma_cols == feature_cols + ["mu_hat"]:
                x_sigma = np.concatenate([x, np.array([[mu]], dtype=float)], axis=1)
            else:
                feat_map = {c: float(v) for c, v in zip(feature_cols, x_vals)}
                feat_map["mu_hat"] = float(mu)
                # If sigma model expects derived sigma features, compute them.
                try:
                    from src.utils.sigma_features import add_sigma_derived_features_map
                    add_sigma_derived_features_map(feat_map)
                except Exception:
                    pass
                x_sigma_vals: List[float] = []
                for c in sigma_cols:
                    if c in feat_map:
                        x_sigma_vals.append(_safe_float(feat_map[c], 0.0))
                    else:
                        # fall back to defaults similar to training scripts
                        if c in ["vegas_game_total", "vegas_spread", "vegas_abs_spread"]:
                            x_sigma_vals.append(0.0)
                        elif c == "is_injured":
                            x_sigma_vals.append(0.0)
                        elif c == "days_since_last_dnp":
                            x_sigma_vals.append(999.0)
                        elif c == "dnp_count_last_10":
                            x_sigma_vals.append(0.0)
                        elif c == "has_prop_line":
                            x_sigma_vals.append(0.0)
                        elif "fg_pct" in c:
                            x_sigma_vals.append(0.45)
                        elif "3pt_pct" in c:
                            x_sigma_vals.append(0.35)
                        else:
                            x_sigma_vals.append(0.0)
                x_sigma = np.array(x_sigma_vals, dtype=float).reshape(1, -1)

            try:
                sigma_raw_pred = float(sm.predict(x_sigma)[0])
                sigma_pred = float(np.exp(sigma_raw_pred)) if use_log else sigma_raw_pred
                sigma_eff = max(sigma_pred * scale, eps)
                sigma_eff = float(np.clip(sigma_eff, 1.0, 20.0))
            except Exception:
                sigma_eff = sigma if sigma > 0 else 1e-6

        line = float(row["prop_pts_line"])

        z = (line - mu) / sigma_eff
        p_over_raw = 1.0 - norm_cdf(z)
        p_over_raw = float(min(1.0, max(0.0, p_over_raw)))

        # Optional calibration (points only)
        p_over = p_over_raw
        use_cal = getattr(evaluate_slate, "_use_calibrator", False)
        if is_points and use_cal and getattr(evaluate_slate, "_calibrator", None) is not None:
            try:
                p_over = float(evaluate_slate._calibrator.predict([p_over_raw])[0])
                p_over = float(min(1.0, max(0.0, p_over)))
            except Exception:
                p_over = p_over_raw

        p_under = 1.0 - p_over

        over_odds = row.get("over_odds_best", np.nan)
        under_odds = row.get("under_odds_best", np.nan)

        implied_over = american_to_prob(over_odds) if not pd.isna(over_odds) else float("nan")
        implied_under = american_to_prob(under_odds) if not pd.isna(under_odds) else float("nan")

        edge_over = p_over - implied_over if not math.isnan(implied_over) else float("nan")
        edge_under = p_under - implied_under if not math.isnan(implied_under) else float("nan")

        rec = {
            "player": row["player"],
            "game_date": row["game_date"],
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "market_key": row.get("market_key"),
            "prop_pts_line": line,
            "model_mean_pts": mu,
            "sigma": float(sigma_eff),
            "over_odds_best": over_odds,
            "under_odds_best": under_odds,
            "model_p_over": p_over,
            "model_p_under": p_under,
            "model_p_over_raw": float(p_over_raw),
            "model_sigma_raw": sigma_raw_pred,
            "implied_p_over": implied_over,
            "implied_p_under": implied_under,
            "edge_over": edge_over,
            "edge_under": edge_under,
            # debug columns
            "is_home_used": overrides.get("is_home", feat_row.get("is_home", np.nan)),
            "days_rest_used": overrides.get("days_since_last_game", feat_row.get("days_since_last_game", np.nan)),
            "opp_allowed5_used": overrides.get("opp_pts_allowed_roll5", feat_row.get("opp_pts_allowed_roll5", np.nan)),
            "opp_allowed15_used": overrides.get("opp_pts_allowed_roll15", feat_row.get("opp_pts_allowed_roll15", np.nan)),
            "opp_abbrev_used": meta.get("opp_abbr_used"),
            "pos_bucket_used": meta.get("pos_bucket_used"),
            "days_stale_used": float(days_stale),
            "last_game_date_used": str(last_game_ts.date()) if not pd.isna(last_game_ts) else None,
        }
        records.append(rec)

    if debug_skips and skipped_counts:
        total_skipped = sum(skipped_counts.values())
        print(f"\n[DEBUG] Skipped {total_skipped:,} rows. Top reasons:")
        for k, v in sorted(skipped_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]:
            print(f"  - {k}: {v:,}")

    df_out = pd.DataFrame.from_records(records)
    if df_out.empty:
        return df_out

    def pick_side(r):
        eo, eu = r["edge_over"], r["edge_under"]
        if math.isnan(eo) and math.isnan(eu):
            return pd.Series({"best_side": None, "best_edge": float("nan")})
        if math.isnan(eo):
            return pd.Series({"best_side": "under", "best_edge": eu})
        if math.isnan(eu):
            return pd.Series({"best_side": "over", "best_edge": eo})
        if eo >= eu:
            return pd.Series({"best_side": "over", "best_edge": eo})
        else:
            return pd.Series({"best_side": "under", "best_edge": eu})

    side_info = df_out.apply(pick_side, axis=1)
    df_out = pd.concat([df_out, side_info], axis=1)

    df_out = df_out[df_out["best_edge"] >= min_edge].copy()
    df_out = df_out.sort_values("best_edge", ascending=False)
    return df_out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a market_lines slate with per-market models and compute edges."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/points_regression.pkl",
        help="(Legacy) Default model bundle path used for player_points if per-market paths not provided.",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        default=None,
        help="Optional per-market model bundle paths, comma-separated like "
             "'player_points=models/points_regression.pkl,player_rebounds=models/rebounds_regression.pkl'. "
             "If omitted, uses built-in defaults for points/rebounds/assists/threes.",
    )
    parser.add_argument("--features-csv", type=str, default="data/player_points_features.csv")
    parser.add_argument("--market-lines", type=str, default="data/market_lines.csv")
    parser.add_argument("--output", type=str, default="data/edges_with_market.csv")
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument(
        "--player-positions-csv",
        type=str,
        default="data/player_positions.csv",
        help="Optional mapping needed to recompute opp_dvp_pos_* features.",
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=60,
        help="Skip a prop if the player's latest feature row is older than this many days before the prop game_date.",
    )
    parser.add_argument(
        "--debug-skips",
        action="store_true",
        help="Print counts of why props were skipped (useful when wiring up new slates).",
    )
    parser.add_argument(
        "--use-tiered-models",
        action="store_true",
        help="If set, use tier-specific models (tier 0-3) based on player's star_tier_pts.",
    )
    parser.add_argument(
        "--use-tiered-ensemble",
        action="store_true",
        help="If set, blend tiered + unified predictions: mu = w*tiered + (1-w)*unified.",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.45,
        help="Weight w for tiered prediction in tiered ensemble (default: 0.45).",
    )
    parser.add_argument(
        "--use-quantile-models",
        action="store_true",
        help="If set, use quantile regression ensemble (10th/50th/90th percentiles) to address systematic bias.",
    )
    parser.add_argument(
        "--use-sigma-model",
        action="store_true",
        help="If set, use heteroscedastic sigma model (models/points_sigma_model.pkl) instead of global sigma.",
    )
    parser.add_argument(
        "--sigma-model-path",
        type=str,
        default=str(SIGMA_MODEL_PATH),
        help="Path to sigma model bundle (default: models/points_sigma_model.pkl).",
    )
    parser.add_argument(
        "--use-calibrator",
        action="store_true",
        help="If set, apply isotonic calibrator (models/over_prob_calibrator.pkl) to model_p_over.",
    )
    parser.add_argument(
        "--calibrator-path",
        type=str,
        default=str(CALIBRATOR_PATH),
        help="Path to over-prob calibrator bundle (default: models/over_prob_calibrator.pkl).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    features_csv = Path(args.features_csv)
    market_lines_csv = Path(args.market_lines)
    output_path = Path(args.output)
    min_edge = float(args.min_edge)

    # Per-market model mapping
    model_paths: Dict[str, Path] = {}
    if args.model_paths:
        # Format: key=path,key=path
        for part in str(args.model_paths).split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                model_paths[k] = Path(v)
    else:
        model_paths = {k: Path(v) for k, v in DEFAULT_MODEL_PATHS.items()}
        # Allow legacy --model-path override for points
        model_paths["player_points"] = model_path

    # For multi-market support, we load models per market_key and route each row
    # to the appropriate model bundle. Tiered/quantile/sigma/calibration currently
    # apply to player_points only (you can extend later per market).
    use_quantile = args.use_quantile_models
    use_ensemble = args.use_tiered_ensemble and not use_quantile
    use_tiered = args.use_tiered_models and (not use_quantile) and (not use_ensemble)  # quantile/ensemble take precedence
    tier_models = None
    
    if use_quantile:
        print("Loading quantile regression models...")
        from src.utils.load_quantile_model import load_quantile_models, load_base_model
        
        quantile_models = load_quantile_models()
        base_model = load_base_model()
        
        if not quantile_models:
            print("[WARN] No quantile models found, falling back to unified model")
            use_quantile = False
        else:
            print(f"  -> Loaded {len(quantile_models)} quantile models (10th, 50th, 90th percentiles)")
            # Store in evaluate_slate function for access
            evaluate_slate._use_quantile = True
            evaluate_slate._quantile_models = quantile_models
            evaluate_slate._base_model = base_model
            # Use base model's feature cols and sigma
            feature_cols = base_model["feature_cols"]
            sigma = float(base_model.get("sigma", 6.0))
            model = base_model["model"]  # For fallback
            print(f"  -> Using quantile ensemble with base model features")
    elif use_ensemble:
        print("Loading unified + tiered models for ensemble...")
        from src.utils.load_tiered_model import load_tier_model

        # Unified model
        bundle_u = load_model_bundle(model_path)
        model_u = bundle_u["model"]
        feature_cols_u = bundle_u["feature_cols"]
        sigma_u = float(bundle_u.get("sigma", 6.0))

        # Tiered models
        tier_models = {}
        for tier in [0, 1, 2, 3]:
            tier_bundle = load_tier_model(tier)
            if tier_bundle is not None:
                tier_models[tier] = tier_bundle
                print(
                    f"  -> Tier {tier}: {len(tier_bundle['feature_cols'])} features, "
                    f"sigma={tier_bundle.get('sigma', 0):.3f}"
                )

        if not tier_models:
            print("[WARN] No tier models found, falling back to unified model")
            use_ensemble = False
            model = model_u
            feature_cols = feature_cols_u
            sigma = sigma_u
        else:
            # Use unified feature space as the primary input space; we'll map into tier space.
            model = model_u
            feature_cols = feature_cols_u
            sigma = sigma_u
            evaluate_slate._use_tiered_ensemble = True
            evaluate_slate._ensemble_weight = float(args.ensemble_weight)
            evaluate_slate._tier_models = tier_models
            print(f"[INFO] Using tiered+unified ensemble with w={float(args.ensemble_weight):.2f}")
    elif use_tiered:
        print("Loading tiered models...")
        from src.utils.load_tiered_model import load_tier_model
        
        tier_models = {}
        for tier in [0, 1, 2, 3]:
            tier_bundle = load_tier_model(tier)
            if tier_bundle is not None:
                tier_models[tier] = tier_bundle
                print(f"  -> Tier {tier}: {len(tier_bundle['feature_cols'])} features, sigma={tier_bundle.get('sigma', 0):.3f}")
        
        if not tier_models:
            print("[WARN] No tier models found, falling back to unified model")
            use_tiered = False
        else:
            # Use first tier model's feature cols as reference (they should all match)
            feature_cols = list(tier_models.values())[0]["feature_cols"]
            sigma = list(tier_models.values())[0].get("sigma", 6.0)
            # Create a dummy unified model for fallback
            bundle = load_model_bundle(model_path)
            model = bundle["model"]
            print(f"  -> Loaded {len(tier_models)} tier models, using as primary")
    else:
        # Load points model as default (used for player_points or as fallback)
        print("Loading model bundle(s)...")
        bundle_points = load_model_bundle(model_paths.get("player_points", model_path))
        model = bundle_points["model"]
        feature_cols = bundle_points["feature_cols"]
        sigma = float(bundle_points.get("sigma", 6.0))
        print(f"  -> Loaded points model with {len(feature_cols)} features, sigma={sigma:.3f}")

        # Load other market models (reb/ast/3PM) if present; they should share the same feature cols.
        per_market_models: Dict[str, Dict] = {"player_points": bundle_points}
        for mk, pth in model_paths.items():
            if mk == "player_points":
                continue
            try:
                if Path(pth).exists():
                    b = load_model_bundle(Path(pth))
                    per_market_models[mk] = b
                    print(f"  -> Loaded {mk} model from {pth}")
                else:
                    print(f"[WARN] Model for {mk} not found at {pth}; will skip {mk} rows.")
            except Exception as e:
                print(f"[WARN] Failed to load model for {mk} at {pth}: {e}")

        evaluate_slate._per_market_models = per_market_models
        evaluate_slate._use_quantile = False
        evaluate_slate._use_tiered_ensemble = False

    # Optional: heteroscedastic sigma model + calibrator for probabilities
    evaluate_slate._use_sigma_model = bool(args.use_sigma_model)
    evaluate_slate._use_calibrator = bool(args.use_calibrator)

    if evaluate_slate._use_sigma_model:
        sigma_model, sigma_cols, sigma_cfg = load_sigma_model_bundle(Path(args.sigma_model_path))
        if sigma_model is None:
            print(
                f"[WARN] --use-sigma-model set but sigma model not found at {args.sigma_model_path}; "
                "using global sigma."
            )
            evaluate_slate._use_sigma_model = False
        else:
            evaluate_slate._sigma_model = sigma_model
            evaluate_slate._sigma_feature_cols = sigma_cols
            evaluate_slate._sigma_cfg = sigma_cfg
            print(f"[INFO] Using sigma model from {args.sigma_model_path}")

    if evaluate_slate._use_calibrator:
        calibrator, cal_info = load_over_prob_calibrator(Path(args.calibrator_path))
        if calibrator is None:
            print(
                f"[WARN] --use-calibrator set but calibrator not found at {args.calibrator_path}; "
                "using raw probabilities."
            )
            evaluate_slate._use_calibrator = False
        else:
            evaluate_slate._calibrator = calibrator
            evaluate_slate._calibrator_info = cal_info
            print(f"[INFO] Using calibrator from {args.calibrator_path}")

    print(f"\nLoading features from {features_csv} ...")
    df_features = load_features(features_csv)
    print(f"  -> Loaded {len(df_features):,} rows for features.")

    if "minutes_pred" in feature_cols and "minutes_pred" not in df_features.columns:
        print("\n[INFO] 'minutes_pred' required by model but missing in features; computing it now...")
        ok = add_minutes_pred_feature(df_features, MINUTES_MODEL_PATH)
        if not ok:
            raise RuntimeError(
                "Failed to add minutes_pred feature required by model. "
                "Either train points model without --use-minutes-pred or fix minutes model."
            )

    print("\nBuilding matchup lookup tables (team defense + DvP)...")
    team_log = build_team_game_log(df_features)
    team_allowed = build_team_allowed_roll_asof(team_log, windows=(5, 15))
    allowed_asof = asof_lookup(
        team_allowed,
        key_cols=["team_abbrev"],
        date_col="game_date_ts",
        value_cols=["team_pts_allowed_roll5", "team_pts_allowed_roll15"],
    )
    print("[INFO] Team defense tables ready.")

    player_id_to_pos = load_player_positions(Path(args.player_positions_csv))
    dvp_asof: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
    if player_id_to_pos:
        dvp_df = build_dvp_pos_roll_asof(df_features, player_id_to_pos, windows=(5, 15))
        dvp_asof = asof_lookup(
            dvp_df,
            key_cols=["def_team", "pos_bucket"],
            date_col="game_date_ts",
            value_cols=["dvp_pos_pts_roll5", "dvp_pos_pts_roll15"],
        )
        print("[INFO] DvP tables ready.")
    else:
        print("[WARN] No player positions loaded; opp_dvp_pos_* will NOT be recomputed (stale or skipped).")

    print(f"\nLoading market lines from {market_lines_csv} ...")
    df_market = load_market_lines(market_lines_csv)
    print(f"  -> Loaded {len(df_market):,} rows for market lines.")

    # Load game lines (Vegas spreads/totals) if available
    game_lines_path = Path("data/game_lines.csv")
    game_lines_df = None
    if game_lines_path.exists():
        try:
            game_lines_df = pd.read_csv(game_lines_path)
            game_lines_df["game_date"] = pd.to_datetime(game_lines_df["game_date"]).dt.date
            print(f"[INFO] Loaded {len(game_lines_df)} game lines from {game_lines_path}")
        except Exception as e:
            print(f"[WARN] Failed to load game lines: {e}")
    else:
        print("[INFO] No game_lines.csv found; Vegas features will be 0.0")

    print(
        f"\nEvaluating slate with min_edge={min_edge:.3f} "
        f"(max_stale_days={args.max_stale_days}) ..."
    )
    df_edges = evaluate_slate(
        df_market=df_market,
        df_features=df_features,
        model=model,
        feature_cols=feature_cols,
        sigma=sigma,
        min_edge=min_edge,
        allowed_asof=allowed_asof,
        dvp_asof=dvp_asof,
        player_id_to_pos=player_id_to_pos,
        max_stale_days=args.max_stale_days,
        debug_skips=args.debug_skips,
        game_lines_df=game_lines_df,
        use_tiered_models=use_tiered,
        tier_models=tier_models,
    )

    if df_edges.empty:
        print("\nNo bets found with edge >= min_edge.")
        return

    print(f"\nFound {len(df_edges):,} candidate bets with edge >= {min_edge:.3f}.")

    preview_cols = [
        "player",
        "game_date",
        "prop_pts_line",
        "best_side",
        "best_edge",
        "model_mean_pts",
        "over_odds_best",
        "under_odds_best",
        "is_home_used",
        "days_rest_used",
        "opp_allowed5_used",
        "opp_allowed15_used",
        "opp_abbrev_used",
        "pos_bucket_used",
        "days_stale_used",
    ]
    print("\nTop 20 edges:")
    print(df_edges[preview_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:6.3f}"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_edges.to_csv(output_path, index=False)
    print(f"\nWrote edges with market info to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()