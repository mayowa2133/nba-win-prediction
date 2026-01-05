#!/usr/bin/env python
"""
star_best_bets_screener.py

Modes:

1) Manual star list:
   python star_best_bets_screener.py \
     --players "Devin Booker,Luka Doncic" \
     --odds-file data/odds_slate.csv \
     --season-min 2023 \
     --top-k 5 \
     --min-edge 3.0 \
     --books "Bet365,FanDuel,DraftKings,Bovada,BetMGM,BetRivers" \
     --ladder-thresholds "15,20,25,30,35,40" \
     --target-prob 0.60

2) Auto-stars from the slate (recommended):
   python star_best_bets_screener.py \
     --auto-stars \
     --odds-file data/odds_slate.csv \
     --season-min 2023 \
     --min-line 20.0 \
     --max-stars 10 \
     --books "Bet365,FanDuel,DraftKings,Bovada,BetMGM,BetRivers" \
     --top-k 5 \
     --min-edge 3.0 \
     --ladder-thresholds "15,20,25,30,35,40" \
     --target-prob 0.60

IMPORTANT (parlays):
- Parlays are built ONLY from each player's single "Recommended rung" from the MODEL ladder (Thr+),
  and ONLY if that exact alt line exists in odds_slate.csv (line = Thr - 0.5, side=OVER).
"""

import argparse
import datetime as dt
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from nba_api.stats.static import teams  # now actually used
from src.utils.minutes_utils import add_minutes_predictions  # NEW: use minutes model to add min_pred / minutes_pred


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

LOGS_CSV = Path("data/player_game_logs.csv")
MODEL_PATH = Path("models/points_regression.pkl")
CALIBRATOR_PATH = Path("models/over_prob_calibrator.pkl")
FEATURES_CSV = Path("data/player_points_features.csv")
SIGMA_MODEL_PATH = Path("models/points_sigma_model.pkl")
PLAYER_POSITIONS_CSV_DEFAULT = Path("data/player_positions.csv")

# IMPORTANT:
# This FEATURE_COLS list is kept in sync with whatever the MODEL BUNDLE
# says it uses. In practice, we rely on the `feature_cols` that come
# from points_regression.pkl, and use this as a fallback.
FEATURE_COLS = [
    "minutes_roll5",
    "minutes_roll15",
    "pts_roll5",
    "pts_roll15",
    "reb_roll5",
    "reb_roll15",
    "ast_roll5",
    "ast_roll15",
    "fg3m_roll5",
    "fg3m_roll15",
    "fg3a_roll5",
    "fg3a_roll15",
    "fga_roll5",
    "fga_roll15",
    "fta_roll5",
    "fta_roll15",
    # ✅ ADDED: these are in your trained model feature list
    "usg_events_roll5",
    "usg_events_roll15",
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
    # ✅ ADDED: these are in your trained model feature list
    "team_margin_roll5",
    "team_margin_roll15",
    "days_since_last_game",
    "is_home",
    "opp_dvp_pos_pts_roll5",
    "opp_dvp_pos_pts_roll15",
    "team_pace_roll5",
    "team_pace_roll15",
    "player_pts_career_mean",
    "player_pts_season_mean",
    "player_minutes_career_mean",
    "player_minutes_season_mean",
    # New relational / star-tier features (must be in sync with regression model)
    "rel_minutes_vs_career",
    "rel_pts_vs_career",
    "star_tier_pts",
    "star_tier_minutes",
    "pts_trend_5_15",
    "minutes_trend_5_15",
    "fga_trend_5_15",
    "pts_std5",
    "minutes_std5",
    "fga_std5",
    "pts_per_min_roll5",
    "fga_per_min_roll5",
    "fta_per_min_roll5",
    "is_b2b",
    "is_long_rest",
    
    # Vegas lines features (game-level context from sportsbooks)
    "vegas_game_total",   # O/U total points - predicts scoring environment
    "vegas_spread",       # spread for this team - predicts game script / blowout risk
    "vegas_abs_spread",   # absolute spread - blowout risk regardless of side
    
    # Injury/availability features (NEW: Phase 2)
    "is_injured",         # binary: 1 if injured/DNP, 0 otherwise
    "days_since_last_dnp", # days since last DNP (999 if never)
    "dnp_count_last_10",  # number of DNPs in last 10 games
]


# ----------------------------------------------------------------------
# Team name → abbreviation mapping (for opponent inference)
# ----------------------------------------------------------------------

NBA_TEAMS = teams.get_teams()
TEAM_NAME_TO_ABBREV: Dict[str, str] = {}

for t in NBA_TEAMS:
    abbr = t.get("abbreviation")
    if not abbr:
        continue
    abbr = abbr.upper()
    full = str(t.get("full_name", "")).strip().lower()
    nick = str(t.get("nickname", "")).strip().lower()
    city = str(t.get("city", "")).strip().lower()

    if full:
        TEAM_NAME_TO_ABBREV[full] = abbr
    if nick:
        TEAM_NAME_TO_ABBREV[nick] = abbr
    if city:
        TEAM_NAME_TO_ABBREV[city] = abbr
    if city and nick:
        TEAM_NAME_TO_ABBREV[f"{city} {nick}"] = abbr


def team_name_to_abbrev(name: Any) -> str:
    if not name:
        return "UNK"
    s = str(name).strip().lower()
    return TEAM_NAME_TO_ABBREV.get(s, "UNK")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def normalize_name(s: Any) -> str:
    import unicodedata
    import re

    if s is None:
        return ""
    s = str(s)

    # Normalize accents / unicode → ASCII
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    # Lowercase & strip
    s = s.lower().strip()

    # --- manual alias fixes for naming mismatches between logs and odds ---
    alias_map = {
        # Odds: "Nicolas Claxton" → Logs: "Nic Claxton"
        "nicolas claxton": "nic claxton",
        "nicholas claxton": "nic claxton",  # just in case any source uses this
    }
    if s in alias_map:
        s = alias_map[s]

    # Make "c.j. mccollum" and "cj mccollum" normalize the same
    s = s.replace(".", "")      # remove dots in initials
    s = s.replace("'", " ")     # turn apostrophes into space
    s = s.replace("-", " ")     # treat hyphens as spaces

    # Collapse any non-alphanumeric into single spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = "".join(s.split())     # trim / collapse multiple spaces

    return s


def rolling_mean_last_n(series: pd.Series, n: int) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(series.tail(min(n, len(series))).mean())


def rolling_std_last_n(series: pd.Series, n: int) -> float:
    """Std dev over last n entries; 0.0 if fewer than 2 games."""
    if series is None or len(series) == 0:
        return 0.0
    s = series.tail(min(n, len(series)))
    if len(s) < 2:
        return 0.0
    return float(s.std(ddof=1))


def safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(series.mean())


def prob_to_american(p: float) -> Optional[int]:
    """Convert probability p to American odds; None for invalid/degenerate."""
    if p is None:
        return None
    if isinstance(p, float) and math.isnan(p):
        return None
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    else:
        return int(round(100 * (1 - p) / p))


def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return -odds / (-odds + 100.0)
    else:
        raise ValueError("Odds cannot be 0.")


def american_to_profit_per_unit(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    elif odds < 0:
        return 100.0 / abs(odds)
    else:
        raise ValueError("Odds cannot be 0.")


# NEW: for parlay payout estimation
def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    elif odds < 0:
        return 1.0 + 100.0 / abs(odds)
    else:
        raise ValueError("Odds cannot be 0.")


def normal_over_probs(mu: float, sigma: float, line: float) -> Tuple[float, float]:
    if sigma <= 0:
        return (1.0 if mu > line else 0.0, 1.0 if mu <= line else 0.0)
    z = (line - mu) / sigma
    p_under = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_over = 1.0 - p_under
    return p_over, p_under


def align_features(row_df: pd.DataFrame, feature_cols: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    """
    Ensure row_df (expected to be a 1-row DataFrame) has all columns in feature_cols.
    Any missing columns are added with fill_value. Existing columns are untouched.
    """
    missing = [c for c in feature_cols if c not in row_df.columns]
    if missing:
        print(f"[WARN] Missing features, filling with {fill_value}: {missing}")
        for c in missing:
            row_df[c] = fill_value
    return row_df


# ----------------------------------------------------------------------
# DvP (by position) helpers
# ----------------------------------------------------------------------

def normalize_position(pos: Any) -> Optional[str]:
    if not isinstance(pos, str) or not pos.strip():
        return None
    p = pos.strip().upper()
    if "PG" in p or "SG" in p or p.startswith("G"):
        return "G"
    if "SF" in p or "PF" in p or p.startswith("F"):
        return "F"
    if "C" in p:
        return "C"
    if p and p[0] in ("G", "F", "C"):
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


def build_dvp_asof_table(
    features_df: pd.DataFrame,
    player_id_to_pos: Dict[int, str],
    windows: Tuple[int, int] = (5, 15),
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Build an as-of lookup table:
      (def_team_abbrev, pos_bucket) -> (dates_sorted, [[roll5, roll15], ...])

    Requires features_df columns:
      - game_date_parsed (dt.date) OR game_date
      - opp_abbrev
      - player_id
      - target_pts
    """
    if not player_id_to_pos:
        return {}

    df = features_df.copy()

    if "game_date_parsed" not in df.columns:
        if "game_date" not in df.columns:
            print("[WARN] FEATURES_CSV missing game_date; cannot build DvP table.")
            return {}
        df["game_date_parsed"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    required = {"opp_abbrev", "player_id", "target_pts", "game_date_parsed"}
    if not required.issubset(df.columns):
        print(f"[WARN] FEATURES_CSV missing required columns for DvP table: {sorted(required - set(df.columns))}")
        return {}

    tmp = df[["opp_abbrev", "player_id", "target_pts", "game_date_parsed"]].copy()
    tmp["pos_bucket"] = tmp["player_id"].map(player_id_to_pos)
    tmp = tmp.dropna(subset=["pos_bucket", "opp_abbrev", "game_date_parsed"])

    tmp["def_team"] = tmp["opp_abbrev"].astype(str).str.upper()
    tmp["game_date_ts"] = pd.to_datetime(tmp["game_date_parsed"])

    # Points allowed to position per game-date for each defense team
    dvp_day = (
        tmp.groupby(["def_team", "pos_bucket", "game_date_ts"], as_index=False)["target_pts"]
        .sum()
        .rename(columns={"target_pts": "pts_allowed_to_pos"})
        .sort_values(["def_team", "pos_bucket", "game_date_ts"])
    )

    w1, w2 = windows
    g = dvp_day.groupby(["def_team", "pos_bucket"], group_keys=False)
    dvp_day[f"dvp_pos_pts_roll{w1}"] = g["pts_allowed_to_pos"].apply(
        lambda s: s.shift(1).rolling(w1, min_periods=1).mean()
    )
    dvp_day[f"dvp_pos_pts_roll{w2}"] = g["pts_allowed_to_pos"].apply(
        lambda s: s.shift(1).rolling(w2, min_periods=1).mean()
    )

    table: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for (def_team, pos_bucket), sub in dvp_day.groupby(["def_team", "pos_bucket"]):
        dates = sub["game_date_ts"].to_numpy()
        vals = sub[[f"dvp_pos_pts_roll{w1}", f"dvp_pos_pts_roll{w2}"]].to_numpy()
        table[(str(def_team), str(pos_bucket))] = (dates, vals)

    print("[INFO] Built DvP as-of table.")
    return table


def dvp_asof_get(
    table: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    def_team: str,
    pos_bucket: str,
    when_date: dt.date,
) -> Optional[np.ndarray]:
    key = (str(def_team).upper(), str(pos_bucket))
    if key not in table:
        return None
    dates, vals = table[key]
    when = np.datetime64(pd.Timestamp(when_date))
    idx = np.searchsorted(dates, when, side="left") - 1
    if idx < 0:
        return None
    return vals[idx]


# ----------------------------------------------------------------------
# Model + calibrator loading
# ----------------------------------------------------------------------


def load_regression_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle["model"]
        sigma = float(bundle.get("sigma", 7.0))
        feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    else:
        model = bundle
        sigma = 7.0
        feature_cols = FEATURE_COLS
    return model, sigma, feature_cols


def load_sigma_model(path: Path = SIGMA_MODEL_PATH):
    """
    Optional heteroskedastic sigma model. Expects a bundle like:
      {
        "model": <regressor>,
        "feature_cols": [...],
        "config": {"use_log_target": True, "eps": 0.001, "sigma_scale": 1.0}
      }
    """
    if not path.exists():
        print(f"[INFO] Sigma model file not found at {path}, using fixed sigma from regression model.")
        # Return None but still give sane defaults
        return None, FEATURE_COLS + ["mu_hat"], {}

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        sigma_model = bundle.get("model") or bundle.get("sigma_model")
        sigma_feature_cols = bundle.get("feature_cols", FEATURE_COLS + ["mu_hat"])
        sigma_config = bundle.get("config", {})
    else:
        sigma_model = bundle
        sigma_feature_cols = FEATURE_COLS + ["mu_hat"]
        sigma_config = {}

    print(f"[INFO] Loaded sigma model from {path}")
    print(f"[INFO] Sigma model feature_cols: {sigma_feature_cols}")
    if sigma_config:
        print(f"[INFO] Sigma config: {sigma_config}")
    return sigma_model, sigma_feature_cols, sigma_config


def load_over_prob_calibrator(path: Path = CALIBRATOR_PATH):
    if not path.exists():
        print(f"[INFO] Calibrator file not found at {path}, using raw probabilities.")
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        calib = bundle.get("calibrator", None)
    else:
        calib = bundle
    if calib is None:
        print(f"[WARN] Calibrator bundle at {path} did not contain 'calibrator'. Using raw probs.")
    else:
        print(f"[INFO] Loaded calibrator from {path}")
    return calib


# ----------------------------------------------------------------------
# Logs + feature building
# ----------------------------------------------------------------------


def resolve_player_from_logs(
    logs_df: pd.DataFrame,
    player_query: str,
    season_min: int,
) -> Tuple[int, str, str, int, dt.date]:
    df = logs_df.copy()
    df["name_norm"] = df["player_name"].map(normalize_name)
    target = normalize_name(player_query)

    candidates = df[df["name_norm"] == target]
    if candidates.empty:
        candidates = df[df["name_norm"].str.contains(target, na=False)]
    if candidates.empty:
        sample_names = df["player_name"].dropna().unique().tolist()
        print(f"Could not resolve player name '{player_query}' in logs.")
        if sample_names:
            print("Here are some example names from the file:")
            for n in sample_names[:30]:
                print(f"  - {n}")
        raise SystemExit(1)

    cand2 = candidates[candidates["season"] >= season_min]
    if cand2.empty:
        cand2 = candidates

    cand2 = cand2.copy()
    cand2["game_date"] = pd.to_datetime(cand2["game_date"])
    latest = cand2.sort_values(["game_date", "game_id"]).iloc[-1]

    player_id = int(latest["player_id"])
    player_name = str(latest["player_name"])
    team_abbrev = str(latest["team_abbrev"])
    season = int(latest["season"])
    latest_game_date = latest["game_date"].date()

    return player_id, player_name, team_abbrev, season, latest_game_date


def build_feature_row_for_game(
    logs_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame],
    player_id: int,
    team_abbrev: str,
    season: int,
    target_date: dt.date,
    opp_abbrev: str,
    home_away: str,
    player_pos_bucket: Optional[str] = None,
    dvp_asof_table: Optional[Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    Build a single feature row for (player, target_date) consistent with
    build_player_points_features.py, using:
      - this season's history for rolling stats
      - all seasons' history for career averages
      - team pace + relational/star-tier features pulled from player_points_features.csv
        (most recent row before target_date for this player/season).
      - DvP-by-position recomputed "as-of" the upcoming game (if dvp_asof_table + player_pos_bucket present).
    """

    df = logs_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # All games for this player BEFORE target_date (career)
    player_career_games = df[
        (df["player_id"] == player_id)
        & (df["game_date"] < pd.Timestamp(target_date))
    ].sort_values("game_date")

    if player_career_games.empty:
        raise RuntimeError(
            f"No historical games for player_id={player_id} before {target_date}. "
            "Cannot build features."
        )

    # This-season games before target_date
    player_season_games = player_career_games[player_career_games["season"] == season]

    if player_season_games.empty:
        raise RuntimeError(
            f"No games for player_id={player_id} in season {season} before {target_date}. "
            "Cannot build season-based features."
        )

    # Days since last game (from this season history)
    last_game_date = player_season_games["game_date"].max().date()
    days_since_last_game = (target_date - last_game_date).days

    # Rolling means (this season)
    minutes_roll5 = rolling_mean_last_n(player_season_games["minutes"], 5)
    minutes_roll15 = rolling_mean_last_n(player_season_games["minutes"], 15)
    pts_roll5 = rolling_mean_last_n(player_season_games["pts"], 5)
    pts_roll15 = rolling_mean_last_n(player_season_games["pts"], 15)
    reb_roll5 = rolling_mean_last_n(player_season_games["reb"], 5)
    reb_roll15 = rolling_mean_last_n(player_season_games["reb"], 15)
    ast_roll5 = rolling_mean_last_n(player_season_games["ast"], 5)
    ast_roll15 = rolling_mean_last_n(player_season_games["ast"], 15)
    fg3m_roll5 = rolling_mean_last_n(player_season_games["fg3m"], 5)
    fg3m_roll15 = rolling_mean_last_n(player_season_games["fg3m"], 15)
    fg3a_roll5 = rolling_mean_last_n(player_season_games["fg3a"], 5)
    fg3a_roll15 = rolling_mean_last_n(player_season_games["fg3a"], 15)
    fga_roll5 = rolling_mean_last_n(player_season_games["fga"], 5)
    fga_roll15 = rolling_mean_last_n(player_season_games["fga"], 15)
    fta_roll5 = rolling_mean_last_n(player_season_games["fta"], 5)
    fta_roll15 = rolling_mean_last_n(player_season_games["fta"], 15)

    # --- ✅ NEW: usg_events_roll5/15 (usage events) ---
    # Prefer a column if present; otherwise compute a reasonable proxy.
    # We ALSO later override from features_df last-row if available to ensure training consistency.
    if "usg_events" in player_season_games.columns:
        usg_series = player_season_games["usg_events"].fillna(0.0)
    else:
        fga_s = player_season_games["fga"].fillna(0.0) if "fga" in player_season_games.columns else 0.0
        fta_s = player_season_games["fta"].fillna(0.0) if "fta" in player_season_games.columns else 0.0
        if "tov" in player_season_games.columns:
            tov_s = player_season_games["tov"].fillna(0.0)
        elif "turnovers" in player_season_games.columns:
            tov_s = player_season_games["turnovers"].fillna(0.0)
        else:
            tov_s = 0.0
        usg_series = fga_s + 0.44 * fta_s + tov_s

    usg_events_roll5 = rolling_mean_last_n(pd.Series(usg_series), 5)
    usg_events_roll15 = rolling_mean_last_n(pd.Series(usg_series), 15)

    # Opponent defensive form (points allowed) – still season-based
    team_games = (
        df.groupby(
            ["season", "game_id", "team_abbrev", "opp_abbrev", "game_date"],
            as_index=False,
        )
        .agg({"team_score": "max", "opp_score": "max"})
    )

    opp_games = team_games[
        (team_games["season"] == season)
        & (team_games["team_abbrev"] == opp_abbrev)
        & (team_games["game_date"] < pd.Timestamp(target_date))
    ].sort_values("game_date")

    if opp_games.empty:
        opp_pts_allowed_roll5 = 0.0
        opp_pts_allowed_roll15 = 0.0
    else:
        opp_pts_allowed_roll5 = rolling_mean_last_n(opp_games["opp_score"], 5)
        opp_pts_allowed_roll15 = rolling_mean_last_n(opp_games["opp_score"], 15)

    # --- ✅ NEW: team_margin_roll5/15 (team point margin form) ---
    team_hist = team_games[
        (team_games["season"] == season)
        & (team_games["team_abbrev"] == team_abbrev)
        & (team_games["game_date"] < pd.Timestamp(target_date))
    ].sort_values("game_date")

    if team_hist.empty:
        team_margin_roll5 = 0.0
        team_margin_roll15 = 0.0
    else:
        margin = (team_hist["team_score"] - team_hist["opp_score"]).astype(float)
        team_margin_roll5 = rolling_mean_last_n(margin, 5)
        team_margin_roll15 = rolling_mean_last_n(margin, 15)

    is_home = 1 if home_away.lower() == "home" else 0

    # Career + season means (all games before target_date)
    player_pts_career_mean = safe_mean(player_career_games["pts"])
    player_pts_season_mean = safe_mean(player_season_games["pts"])
    player_minutes_career_mean = safe_mean(player_career_games["minutes"])
    player_minutes_season_mean = safe_mean(player_season_games["minutes"])

    # Trend: last-5 minus last-15 (how hot/cold vs longer form)
    pts_trend_5_15 = pts_roll5 - pts_roll15
    minutes_trend_5_15 = minutes_roll5 - minutes_roll15
    fga_trend_5_15 = fga_roll5 - fga_roll15

    # Volatility over last 5 games (this season)
    pts_std5 = rolling_std_last_n(player_season_games["pts"], 5)
    minutes_std5 = rolling_std_last_n(player_season_games["minutes"], 5)
    fga_std5 = rolling_std_last_n(player_season_games["fga"], 5)

    # Per-minute rates based on last-5 averages
    eps = 1e-6
    pts_per_min_roll5 = pts_roll5 / max(minutes_roll5, eps)
    fga_per_min_roll5 = fga_roll5 / max(minutes_roll5, eps)
    fta_per_min_roll5 = fta_roll5 / max(minutes_roll5, eps)

    # Rest flags
    is_b2b = 1 if days_since_last_game == 1 else 0
    # ✅ FIX: align long-rest threshold with the model pipeline (>=3 is typical)
    is_long_rest = 1 if days_since_last_game >= 3 else 0

    # DvP + team pace features
    opp_dvp_pos_pts_roll5 = 0.0
    opp_dvp_pos_pts_roll15 = 0.0
    team_pace_roll5 = 0.0
    team_pace_roll15 = 0.0

    # New relational / star-tier features – default to 0, but try to pull from
    # the last row in features_df if available, to stay consistent with training.
    rel_minutes_vs_career = 0.0
    rel_pts_vs_career = 0.0
    star_tier_pts = 0.0
    star_tier_minutes = 0.0

    # Pull team pace + relational/star-tier + (optionally) usage/margin from last features row
    if features_df is not None:
        feats = features_df[
            (features_df["player_id"] == player_id)
            & (features_df["season"] == season)
            & (features_df["game_date_parsed"] < target_date)
        ].sort_values("game_date_parsed")

        if not feats.empty:
            last = feats.iloc[-1]

            team_pace_roll5 = float(last.get("team_pace_roll5", 0.0))
            team_pace_roll15 = float(last.get("team_pace_roll15", 0.0))

            rel_minutes_vs_career = float(last.get("rel_minutes_vs_career", 0.0))
            rel_pts_vs_career = float(last.get("rel_pts_vs_career", 0.0))
            star_tier_pts = float(last.get("star_tier_pts", 0.0))
            star_tier_minutes = float(last.get("star_tier_minutes", 0.0))

            # Optional consistency pulls if present
            usg_events_roll5 = float(last.get("usg_events_roll5", usg_events_roll5))
            usg_events_roll15 = float(last.get("usg_events_roll15", usg_events_roll15))
            team_margin_roll5 = float(last.get("team_margin_roll5", team_margin_roll5))
            team_margin_roll15 = float(last.get("team_margin_roll15", team_margin_roll15))

    # Recompute DvP-by-position as-of the upcoming opponent, if possible
    if dvp_asof_table is not None and player_pos_bucket and opp_abbrev and opp_abbrev != "UNK":
        vv = dvp_asof_get(dvp_asof_table, def_team=opp_abbrev, pos_bucket=player_pos_bucket, when_date=target_date)
        if vv is not None and len(vv) >= 2:
            opp_dvp_pos_pts_roll5 = float(vv[0])
            opp_dvp_pos_pts_roll15 = float(vv[1])

    row = {
        "minutes_roll5": minutes_roll5,
        "minutes_roll15": minutes_roll15,
        "pts_roll5": pts_roll5,
        "pts_roll15": pts_roll15,
        "reb_roll5": reb_roll5,
        "reb_roll15": reb_roll15,
        "ast_roll5": ast_roll5,
        "ast_roll15": ast_roll15,
        "fg3m_roll5": fg3m_roll5,
        "fg3m_roll15": fg3m_roll15,
        "fg3a_roll5": fg3a_roll5,
        "fg3a_roll15": fg3a_roll15,
        "fga_roll5": fga_roll5,
        "fga_roll15": fga_roll15,
        "fta_roll5": fta_roll5,
        "fta_roll15": fta_roll15,
        "usg_events_roll5": usg_events_roll5,
        "usg_events_roll15": usg_events_roll15,
        "opp_pts_allowed_roll5": opp_pts_allowed_roll5,
        "opp_pts_allowed_roll15": opp_pts_allowed_roll15,
        "team_margin_roll5": team_margin_roll5,
        "team_margin_roll15": team_margin_roll15,
        "days_since_last_game": float(days_since_last_game),
        "is_home": is_home,
        "opp_dvp_pos_pts_roll5": opp_dvp_pos_pts_roll5,
        "opp_dvp_pos_pts_roll15": opp_dvp_pos_pts_roll15,
        "team_pace_roll5": team_pace_roll5,
        "team_pace_roll15": team_pace_roll15,
        "player_pts_career_mean": player_pts_career_mean,
        "player_pts_season_mean": player_pts_season_mean,
        "player_minutes_career_mean": player_minutes_career_mean,
        "player_minutes_season_mean": player_minutes_season_mean,
        "rel_minutes_vs_career": rel_minutes_vs_career,
        "rel_pts_vs_career": rel_pts_vs_career,
        "star_tier_pts": star_tier_pts,
        "star_tier_minutes": star_tier_minutes,
        "pts_trend_5_15": pts_trend_5_15,
        "minutes_trend_5_15": minutes_trend_5_15,
        "fga_trend_5_15": fga_trend_5_15,
        "pts_std5": pts_std5,
        "minutes_std5": minutes_std5,
        "fga_std5": fga_std5,
        "pts_per_min_roll5": pts_per_min_roll5,
        "fga_per_min_roll5": fga_per_min_roll5,
        "fta_per_min_roll5": fta_per_min_roll5,
        "is_b2b": is_b2b,
        "is_long_rest": is_long_rest,
        # Vegas lines features (will be populated below if available)
        "vegas_game_total": 0.0,
        "vegas_spread": 0.0,
        "vegas_abs_spread": 0.0,
    }

    feat_df = pd.DataFrame([row]).fillna(0.0)

    # Attach minutes prediction(s) so this screener matches training-time features if used
    try:
        feat_df = add_minutes_predictions(feat_df)
        # Compatibility bridge: some utilities output min_pred, some models expect minutes_pred (or vice-versa)
        if "min_pred" in feat_df.columns and "minutes_pred" not in feat_df.columns:
            feat_df["minutes_pred"] = feat_df["min_pred"]
        if "minutes_pred" in feat_df.columns and "min_pred" not in feat_df.columns:
            feat_df["min_pred"] = feat_df["minutes_pred"]
    except Exception as e:
        print(f"[WARN] Failed to add minutes prediction: {e}")

    # Attach Vegas lines features if game_lines.csv exists
    try:
        game_lines_path = Path("data/game_lines.csv")
        if game_lines_path.exists():
            game_lines_df = pd.read_csv(game_lines_path)
            game_lines_df["game_date"] = pd.to_datetime(game_lines_df["game_date"]).dt.date
            
            # Find matching game by team (home or away)
            home_match = game_lines_df[
                (game_lines_df["game_date"] == target_date) &
                (game_lines_df["home_team"].str.lower().str.contains(team_abbrev.lower(), na=False) |
                 game_lines_df["away_team"].str.lower().str.contains(team_abbrev.lower(), na=False))
            ]
            
            # Also try matching by team abbreviation derived from team names
            if home_match.empty:
                # Try matching by looking up team name to abbrev
                for _, gl_row in game_lines_df[game_lines_df["game_date"] == target_date].iterrows():
                    home_abbr = team_name_to_abbrev(gl_row.get("home_team", ""))
                    away_abbr = team_name_to_abbrev(gl_row.get("away_team", ""))
                    if home_abbr == team_abbrev or away_abbr == team_abbrev:
                        vegas_total = gl_row.get("vegas_game_total")
                        vegas_abs = gl_row.get("vegas_abs_spread")
                        # Spread: use home spread if home, away spread if away
                        if home_abbr == team_abbrev:
                            vegas_spread = gl_row.get("vegas_home_spread", 0.0)
                        else:
                            vegas_spread = gl_row.get("vegas_away_spread", 0.0)
                        
                        if pd.notna(vegas_total):
                            feat_df["vegas_game_total"] = float(vegas_total)
                        if pd.notna(vegas_spread):
                            feat_df["vegas_spread"] = float(vegas_spread)
                        if pd.notna(vegas_abs):
                            feat_df["vegas_abs_spread"] = float(vegas_abs)
                        break
            elif not home_match.empty:
                gl_row = home_match.iloc[0]
                vegas_total = gl_row.get("vegas_game_total")
                vegas_abs = gl_row.get("vegas_abs_spread")
                home_abbr = team_name_to_abbrev(gl_row.get("home_team", ""))
                if home_abbr == team_abbrev:
                    vegas_spread = gl_row.get("vegas_home_spread", 0.0)
                else:
                    vegas_spread = gl_row.get("vegas_away_spread", 0.0)
                
                if pd.notna(vegas_total):
                    feat_df["vegas_game_total"] = float(vegas_total)
                if pd.notna(vegas_spread):
                    feat_df["vegas_spread"] = float(vegas_spread)
                if pd.notna(vegas_abs):
                    feat_df["vegas_abs_spread"] = float(vegas_abs)
    except Exception as e:
        print(f"[WARN] Failed to load Vegas lines: {e}")

    return feat_df


# ----------------------------------------------------------------------
# Odds utilities
# ----------------------------------------------------------------------


def infer_slate_date_and_auto_stars(
    odds_df: pd.DataFrame,
    min_line: float,
    max_stars: int,
) -> Tuple[dt.date, List[Dict[str, Any]]]:
    if "game_date" not in odds_df.columns:
        raise ValueError("odds file must have a 'game_date' column (YYYY-MM-DD strings).")

    dates = pd.to_datetime(odds_df["game_date"]).dt.date
    slate_date = dates.min()

    todays = odds_df[dates == slate_date]
    grp = todays.groupby("player")["line"].max().reset_index(name="max_line")
    stars = grp[grp["max_line"] >= min_line].sort_values("max_line", ascending=False)
    stars = stars.head(max_stars)

    star_list = stars.to_dict(orient="records")

    print(
        f"\n[Auto-stars] Slate date inferred as {slate_date} from odds file. "
        f"Found {len(star_list)} star(s) with max_line >= {min_line:.1f}:"
    )
    for s in star_list:
        print(f"  - {s['player']} (max line {s['max_line']})")

    return slate_date, star_list


def filter_books(odds_df: pd.DataFrame, books_str: Optional[str]) -> pd.DataFrame:
    df = odds_df.copy()
    df["book_norm"] = df["book"].astype(str).str.lower()
    if not books_str:
        return df
    wanted = {b.strip().lower() for b in books_str.split(",") if b.strip()}
    print(f"\nFiltering to books: {', '.join(sorted(wanted))}")
    df = df[df["book_norm"].isin(wanted)]
    return df


def infer_team_context_from_odds(
    player_odds: pd.DataFrame,
    default_team_abbrev: str,
) -> Tuple[str, str, str]:
    """
    Infer (team_abbrev, opp_abbrev, home_away) for this prop from the odds slate.

    Priority:
      1) Use team_abbrev / opp_abbrev / home_away / is_home if present in the odds file.
      2) Otherwise, use home_team / away_team from The Odds API + the player's
         team from logs (default_team_abbrev) to deduce which side he is on
         and who the opponent is.

    Falls back to:
      team_abbrev = default_team_abbrev
      opp_abbrev  = 'UNK'
      home_away   = 'home'
    """
    df = player_odds.copy()

    team_abbrev = default_team_abbrev
    opp_abbrev = "UNK"
    home_away = "home"

    # 1) Direct fields in odds_slate (if present)
    if "team_abbrev" in df.columns:
        vals = (
            df["team_abbrev"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )
        if vals:
            team_abbrev = vals[0]

    if "opp_abbrev" in df.columns:
        vals = (
            df["opp_abbrev"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )
        if vals:
            opp_abbrev = vals[0]

    if "home_away" in df.columns:
        vals = (
            df["home_away"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )
        if vals:
            v = vals[0]
            if v in {"home", "away"}:
                home_away = v
            elif v in {"h", "a"}:
                home_away = "home" if v == "h" else "away"
    elif "is_home" in df.columns:
        vals = df["is_home"].dropna().unique().tolist()
        if vals:
            try:
                home_away = "home" if bool(vals[0]) else "away"
            except Exception:
                pass

    # 2) If we still don't know the opponent but we DO have home_team/away_team,
    #    try to infer from those.
    has_home_away_names = {"home_team", "away_team"}.issubset(df.columns)
    if opp_abbrev == "UNK" and has_home_away_names:
        home_names = (
            df["home_team"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        away_names = (
            df["away_team"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if home_names and away_names:
            home_name = home_names[0]
            away_name = away_names[0]

            home_abbr = team_name_to_abbrev(home_name)
            away_abbr = team_name_to_abbrev(away_name)

            # If team_abbrev hasn't been overridden by odds, align it with logs-side
            if team_abbrev == default_team_abbrev:
                # Case 1: logs say player is on home team
                if home_abbr == default_team_abbrev and away_abbr != "UNK":
                    team_abbrev = home_abbr
                    opp_abbrev = away_abbr
                    home_away = "home"
                # Case 2: logs say player is on away team
                elif away_abbr == default_team_abbrev and home_abbr != "UNK":
                    team_abbrev = away_abbr
                    opp_abbrev = home_abbr
                    home_away = "away"
                else:
                    # Fallback: if we at least know both abbrevs and one matches team_abbrev,
                    # try to use that.
                    if home_abbr != "UNK" and away_abbr != "UNK":
                        if default_team_abbrev == home_abbr:
                            opp_abbrev = away_abbr
                            home_away = "home"
                        elif default_team_abbrev == away_abbr:
                            opp_abbrev = home_abbr
                            home_away = "away"
            else:
                # team_abbrev came from odds; figure out which side this is
                if team_abbrev == home_abbr and away_abbr != "UNK":
                    opp_abbrev = away_abbr
                    home_away = "home"
                elif team_abbrev == away_abbr and home_abbr != "UNK":
                    opp_abbrev = home_abbr
                    home_away = "away"

    return team_abbrev, opp_abbrev, home_away


# NEW: for recommended-rung parlay building
def best_market_over_for_line(player_odds: pd.DataFrame, target_line: float) -> Optional[Dict[str, Any]]:
    """
    Pick the best available OVER price for an exact market line (e.g., 19.5 for 20+).
    Returns the odds row (as dict) or None if not found.
    """
    if player_odds is None or player_odds.empty:
        return None

    df = player_odds.copy()
    df["side_norm"] = df["side"].astype(str).str.lower().str.strip()
    df["line_f"] = df["line"].astype(float)

    cand = df[(df["side_norm"] == "over") & (np.isclose(df["line_f"], float(target_line), atol=1e-6))]
    if cand.empty:
        return None

    cand = cand.copy()
    cand["odds_f"] = cand["odds"].astype(float)
    cand["dec"] = cand["odds_f"].apply(american_to_decimal)
    best = cand.sort_values("dec", ascending=False).iloc[0]
    return best.to_dict()


# ----------------------------------------------------------------------
# Scoring props + ladder
# ----------------------------------------------------------------------


def score_points_prop(
    mu: float,
    sigma: float,
    line: float,
    side: str,
    odds: float,
    calibrator,
) -> Dict[str, float]:
    side = side.lower().strip()
    p_over_raw, p_under_raw = normal_over_probs(mu, sigma, line)

    if calibrator is not None:
        p_over_calib = float(calibrator.predict([p_over_raw])[0])
    else:
        p_over_calib = p_over_raw

    if isinstance(p_over_calib, float) and math.isnan(p_over_calib):
        p_over_calib = p_over_raw

    p_over_calib = min(max(p_over_calib, 1e-6), 1.0 - 1e-6)
    p_under_calib = 1.0 - p_over_calib

    if side == "over":
        p_win = p_over_calib
    elif side == "under":
        p_win = p_under_calib
    else:
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    implied_p = american_to_prob(odds)
    profit_on_win = american_to_profit_per_unit(odds)
    edge = p_win - implied_p
    edge_pct = edge * 100.0
    ev = p_win * profit_on_win - (1.0 - p_win)

    fair_odds = prob_to_american(p_win)
    return {
        "p_win": p_win,
        "implied_prob": implied_p,
        "edge_pct": edge_pct,
        "ev_per_unit": ev,
        "fair_odds": fair_odds,
        "p_over_raw": p_over_raw,
        "p_over_calib": p_over_calib,
    }


def compute_ladder_probs(
    mu: float,
    sigma: float,
    thresholds: List[float],
    calibrator,
) -> Dict[float, Dict[str, float]]:
    """
    For each threshold T, treat it as "T+" and model using line = T - 0.5
    (i.e., over(T-0.5) ~ T+ ladder).
    """
    out: Dict[float, Dict[str, float]] = {}
    for thr in thresholds:
        line_for_thr = thr - 0.5
        p_over_raw, _ = normal_over_probs(mu, sigma, line_for_thr)
        if calibrator is not None:
            p_over_calib = float(calibrator.predict([p_over_raw])[0])
        else:
            p_over_calib = p_over_raw
        if isinstance(p_over_calib, float) and math.isnan(p_over_calib):
            p_over_calib = p_over_raw
        p_over_calib = min(max(p_over_calib, 1e-6), 1.0 - 1e-6)
        fair_odds = prob_to_american(p_over_calib)
        out[thr] = {
            "line_for_thr": line_for_thr,
            "p_over_raw": p_over_raw,
            "p_over_calib": p_over_calib,
            "fair_odds": fair_odds,
        }
    return out


def compute_ladder_best_bets_for_star(
    player_odds: pd.DataFrame,
    mu: float,
    sigma: float,
    calibrator,
    ladder_thresholds: List[float],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if player_odds.empty:
        return results

    ladder_probs = compute_ladder_probs(mu, sigma, ladder_thresholds, calibrator)

    player_odds = player_odds.copy()
    player_odds["side_norm"] = player_odds["side"].astype(str).str.lower()

    for thr in ladder_thresholds:
        info = ladder_probs[thr]
        target_line = info["line_for_thr"]
        p_win = info["p_over_calib"]

        mask_line = np.isclose(player_odds["line"].astype(float), target_line, atol=1e-6)
        cand = player_odds[mask_line & (player_odds["side_norm"] == "over")]

        if cand.empty:
            continue

        scored_rows = []
        for _, r in cand.iterrows():
            odds = float(r["odds"])
            implied_p = american_to_prob(odds)
            profit_on_win = american_to_profit_per_unit(odds)
            edge = p_win - implied_p
            edge_pct = edge * 100.0
            ev = p_win * profit_on_win - (1.0 - p_win)
            fair_odds = prob_to_american(p_win)

            scored_rows.append(
                {
                    "player": r["player"],
                    "book": r["book"],
                    "book_norm": r.get("book_norm", str(r["book"]).lower()),
                    "threshold": thr,
                    "market_line": target_line,
                    "odds": int(odds),
                    "p_win": p_win,
                    "fair_odds": fair_odds,
                    "edge_pct": edge_pct,
                    "ev_per_unit": ev,
                    # NEW: keep matchup identifiers for ladder parlay building
                    "event_id": r.get("event_id", None),
                    "game_date": r.get("game_date", None),
                    "home_team": r.get("home_team", None),
                    "away_team": r.get("away_team", None),
                }
            )

        if not scored_rows:
            continue

        # Parlays: for this threshold, take the best price (highest payout / least juice)
        best = max(scored_rows, key=lambda d: american_to_profit_per_unit(d["odds"]))
        results.append(best)

    return results


# ----------------------------------------------------------------------
# Printing helpers
# ----------------------------------------------------------------------

# -------------------- NEW: Hybrid ladder recommendation helpers --------------------

def fmt_american(odds: Optional[float]) -> str:
    if odds is None:
        return "None"
    o = int(round(float(odds)))
    return f"+{o}" if o > 0 else f"{o}"


def pick_hybrid_ladder_recommendation(
    ladder_best_rows: List[Dict[str, Any]],
    target_prob: float,
) -> Optional[Dict[str, Any]]:
    """
    Hybrid (parlay-first):
      - only consider ladder bets where p_win >= target_prob
      - among those, pick the HIGHEST threshold that clears target_prob
      - if multiple rows exist at that threshold, take the best price
    """
    eligible = [
        r for r in ladder_best_rows
        if float(r.get("p_win", 0.0)) >= float(target_prob)
    ]
    if not eligible:
        return None

    best_thr = max(float(r.get("threshold", 0.0)) for r in eligible)
    at_thr = [r for r in eligible if float(r.get("threshold", 0.0)) == best_thr]
    if not at_thr:
        return None

    return max(at_thr, key=lambda r: american_to_profit_per_unit(float(r.get("odds", -1e9))))


def print_hybrid_ladder_recommendation(
    player_name: str,
    rec: Optional[Dict[str, Any]],
    target_prob: float,
):
    if rec is None:
        print(
            f"  [Hybrid ladder rec] None available with p_win >= {target_prob:.2f} "
            f"(need alt lines in odds_slate.csv for this player)."
        )
        return

    thr_label = f"{int(round(rec['threshold']))}+"
    print(
        f"  [Hybrid ladder rec] {player_name} {thr_label} @ {rec['book']} {fmt_american(rec['odds'])} | "
        f"p_win={rec['p_win']:.3f} | fair={fmt_american(rec.get('fair_odds'))} | "
        f"edge={rec['edge_pct']:.2f}% | EV/unit={rec['ev_per_unit']:.3f}"
    )

# ---------------------------------------------------------------------------------


# NEW: multi-parlay builder from legs (unique games + unique players)

def ladder_leg_game_key(leg: Dict[str, Any]) -> str:
    eid = leg.get("event_id")
    if eid is not None and str(eid).strip():
        return str(eid).strip()

    gd = str(leg.get("game_date") or "").strip()
    away = str(leg.get("away_team") or "").strip()
    home = str(leg.get("home_team") or "").strip()
    return f"{gd}|{away}@{home}"


def ladder_leg_player_key(leg: Dict[str, Any]) -> str:
    return normalize_name(leg.get("player"))


def build_many_uncorrelated_ladder_parlays(
    ladder_legs: List[Dict[str, Any]],
    legs_per_parlay: int,
) -> List[List[Dict[str, Any]]]:
    """
    Greedy: keep creating parlays until we can’t fill another one.

    Constraints inside a parlay:
      - unique game (event_id if present; else fallback to game_date|away@home)
      - unique player (avoid multiple legs from the same player in one parlay)

    Each leg can be used at most once overall.
    """
    if legs_per_parlay <= 0:
        return []

    remaining = sorted(
        ladder_legs,
        key=lambda r: (
            float(r.get("p_win", 0.0)),
            float(r.get("ev_per_unit", 0.0)),
            american_to_decimal(float(r.get("odds", -110))),
        ),
        reverse=True,
    )

    parlays: List[List[Dict[str, Any]]] = []

    while True:
        parlay: List[Dict[str, Any]] = []
        used_games = set()
        used_players = set()

        i = 0
        while i < len(remaining) and len(parlay) < legs_per_parlay:
            leg = remaining[i]
            gk = ladder_leg_game_key(leg)
            pk = ladder_leg_player_key(leg)

            if gk not in used_games and pk not in used_players:
                parlay.append(leg)
                used_games.add(gk)
                used_players.add(pk)
                remaining.pop(i)
            else:
                i += 1

        if len(parlay) < legs_per_parlay:
            break

        parlays.append(parlay)

    return parlays


def print_ladder_parlays(parlays: List[List[Dict[str, Any]]]):
    if not parlays:
        print("\n=== Ladder-threshold parlays ===")
        print("  (No ladder-threshold parlays could be formed with the current constraints.)")
        return

    print(f"\n=== Ladder-threshold parlays (as many as possible) ===")
    for idx, legs in enumerate(parlays, start=1):
        joint_p = 1.0
        dec_joint = 1.0

        print(f"\nParlay #{idx}")
        for j, r in enumerate(legs, 1):
            joint_p *= float(r.get("p_win", 0.0))
            dec_joint *= american_to_decimal(float(r.get("odds", -110)))

            thr_label = f"{int(round(float(r['threshold'])))}+"
            matchup = ""
            if r.get("away_team") and r.get("home_team"):
                matchup = f"{r['away_team']} @ {r['home_team']}"
            elif r.get("game_date"):
                matchup = str(r.get("game_date"))

            print(
                f"{j}. {r['player']} {thr_label} @ {r['book']} {fmt_american(r['odds'])} | "
                f"p_win={float(r['p_win']):.3f} | line={float(r['market_line']):.1f} | {matchup}"
            )

        print(f"  Joint hit prob (model): {joint_p:.4f}")
        print(f"  Parlay decimal payout (approx): {dec_joint:.2f}x")


# NEW: recommended-rung parlay printing wrapper (labels are accurate)
def print_recommended_rung_parlays(parlays: List[List[Dict[str, Any]]]):
    if not parlays:
        print("\n=== Recommended-rung parlays ===")
        print("  (No recommended-rung parlays could be formed: missing markets or not enough unique games.)")
        return

    print("\n=== Recommended-rung parlays (only each player's single model Thr+) ===")
    for idx, legs in enumerate(parlays, start=1):
        joint_p = 1.0
        dec_joint = 1.0

        print(f"\nParlay #{idx}")
        for j, r in enumerate(legs, 1):
            joint_p *= float(r.get("p_win", 0.0))
            dec_joint *= american_to_decimal(float(r.get("odds", -110)))

            thr_label = f"{int(round(float(r['threshold'])))}+"
            matchup = ""
            if r.get("away_team") and r.get("home_team"):
                matchup = f"{r['away_team']} @ {r['home_team']}"
            elif r.get("game_date"):
                matchup = str(r.get("game_date"))

            print(
                f"{j}. {r['player']} {thr_label} @ {r['book']} {fmt_american(r['odds'])} | "
                f"p_win={float(r['p_win']):.3f} | line={float(r['market_line']):.1f} | {matchup}"
            )

        print(f"  Joint hit prob (model): {joint_p:.4f}")
        print(f"  Parlay decimal payout (approx): {dec_joint:.2f}x")


# ---------------------------------------------------------------------------------


# NEW: uncorrelated parlay builder (different games via event_id)
def print_uncorrelated_parlay(all_rows: List[Dict[str, Any]], legs: int, min_prob: float):
    if legs <= 0:
        return

    eligible = [
        r for r in all_rows
        if r.get("event_id") and float(r.get("p_win", 0.0)) >= float(min_prob)
    ]

    print(f"\n=== Suggested uncorrelated parlay ({legs} legs, p_win >= {min_prob:.2f}) ===")
    if not eligible:
        print(f"  (none found with p_win >= {min_prob:.2f} and event_id present)")
        return

    # one pick per event: highest p_win; tie-break by better payout (higher decimal)
    best_by_event: Dict[str, Dict[str, Any]] = {}
    for r in eligible:
        eid = str(r["event_id"])
        cur = best_by_event.get(eid)
        if cur is None:
            best_by_event[eid] = r
            continue

        r_key = (float(r["p_win"]), american_to_decimal(r["odds"]))
        c_key = (float(cur["p_win"]), american_to_decimal(cur["odds"]))
        if r_key > c_key:
            best_by_event[eid] = r

    pool = list(best_by_event.values())
    pool.sort(key=lambda r: (float(r["p_win"]), american_to_decimal(r["odds"])), reverse=True)

    chosen = pool[:legs]
    if len(chosen) < legs:
        print(f"  (only found {len(chosen)} unique games that meet the prob floor)")
        return

    p_joint = 1.0
    dec_joint = 1.0
    for r in chosen:
        p_joint *= float(r["p_win"])
        dec_joint *= american_to_decimal(r["odds"])

    for i, r in enumerate(chosen, 1):
        matchup = ""
        if r.get("away_team") and r.get("home_team"):
            matchup = f"{r['away_team']} @ {r['home_team']}"
        odds_int = int(float(r["odds"]))
        odds_str = f"+{odds_int}" if odds_int > 0 else str(odds_int)
        print(
            f"{i}. {r['player']} {r['side']} {float(r['line']):.1f} ({odds_str}) | "
            f"p_win={float(r['p_win']):.3f} | {matchup} | book={r['book']}"
        )

    print(f"\n  Joint hit prob (model): {p_joint:.4f}")
    print(f"  Parlay decimal payout (approx): {dec_joint:.2f}x")


def print_best_per_star_table(rows: List[Dict[str, Any]], min_edge: float, header: str):
    print(f"\n=== {header} (min_edge filter for display={min_edge:.1f}%) ===")
    print(
        f"{'Player':20s} | {'Book':12s} | {'Line':>5s} | {'Side':>4s} | {'Odds':>5s} | "
        f"{'p_win':>5s} | {'Fair':>7s} | {'Edge%':>7s} | {'EV/unit':>8s}"
    )
    print("-" * 92)

    any_printed = False
    for r in rows:
        edge = r["edge_pct"]
        if edge < min_edge:
            continue
        any_printed = True
        fair_odds = r["fair_odds"]
        fair_str = "None" if fair_odds is None else f"{fair_odds:+d}"
        print(
            f"{r['player'][:20]:20s} | {r['book'][:12]:12s} | "
            f"{r['line']:5.1f} | {r['side'][:4]:>4s} | {int(r['odds']):5d} | "
            f"{r['p_win']:5.3f} | {fair_str:>7s} | {edge:7.2f} | {r['ev_per_unit']:8.3f}"
        )

    if not any_printed:
        print("  (no bets meet the min_edge display filter)")


def print_top_k_table(rows: List[Dict[str, Any]], top_k: int, min_edge: float):
    rows_sorted = sorted(rows, key=lambda r: r["edge_pct"], reverse=True)
    rows_filtered = [r for r in rows_sorted if r["edge_pct"] >= min_edge]
    rows_top = rows_filtered[:top_k]

    print(
        f"\n=== Top {top_k} star props overall by model edge (min_edge={min_edge:.1f}%) ==="
    )
    print(
        f"{'Player':20s} | {'Book':12s} | {'Line':>5s} | {'Side':>4s} | {'Odds':>5s} | "
        f"{'p_win':>5s} | {'Fair':>7s} | {'Edge%':>7s} | {'EV/unit':>8s}"
    )
    print("-" * 92)

    if not rows_top:
        print("  (none)")
        return

    for r in rows_top:
        fair_odds = r["fair_odds"]
        fair_str = "None" if fair_odds is None else f"{fair_odds:+d}"
        print(
            f"{r['player'][:20]:20s} | {r['book'][:12]:12s} | "
            f"{r['line']:5.1f} | {r['side'][:4]:>4s} | {int(r['odds']):5d} | "
            f"{r['p_win']:5.3f} | {fair_str:>7s} | {r['edge_pct']:7.2f} | {r['ev_per_unit']:8.3f}"
        )


def print_ladder_best_bets_for_star(
    player_name: str,
    ladder_rows: List[Dict[str, Any]],
    min_edge: float,
):
    print(f"\n--- Ladder best bets for {player_name} ---")
    print(
        f"{'Threshold+':>10s} | {'Book':12s} | {'Line':>5s} | {'Odds':>5s} | "
        f"{'p_win':>5s} | {'Fair':>7s} | {'Edge%':>7s} | {'EV/unit':>8s}"
    )
    print("-" * 78)
    if not ladder_rows:
        print("  (no ladder bets found for these thresholds)")
        return

    any_printed = False
    for r in sorted(ladder_rows, key=lambda d: d["threshold"]):
        if r["edge_pct"] < min_edge:
            continue
        any_printed = True
        fair_odds = r["fair_odds"]
        fair_str = "None" if fair_odds is None else f"{fair_odds:+d}"
        thr_label = f"{int(round(r['threshold']))}+"
        print(
            f"{thr_label:>10s} | {r['book'][:12]:12s} | "
            f"{r['market_line']:5.1f} | {int(r['odds']):5d} | "
            f"{r['p_win']:5.3f} | {fair_str:>7s} | {r['edge_pct']:7.2f} | {r['ev_per_unit']:8.3f}"
        )

    if not any_printed:
        print("  (no ladder bets meet the min_edge display filter)")


def print_model_ladder_for_star(
    player_name: str,
    ladder_probs: Dict[float, Dict[str, float]],
    target_prob: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    print(f"\n--- Model ladder for {player_name} (P(points >= threshold)) ---")
    print(f"{'Threshold+':>10s} | {'p_over':>7s} | {'Fair':>7s}")
    print("-" * 34)

    thresholds_sorted = sorted(ladder_probs.keys())
    for thr in thresholds_sorted:
        info = ladder_probs[thr]
        p_over = info["p_over_calib"]
        fair_odds = info["fair_odds"]
        fair_str = "None" if fair_odds is None else f"{fair_odds:+d}"
        label = f"{int(round(thr))}+"
        print(f"{label:>10s} | {p_over:7.3f} | {fair_str:>7s}")

    best_info: Optional[Dict[str, Any]] = None

    if target_prob is not None:
        candidates = [
            (thr, ladder_probs[thr]["p_over_calib"], ladder_probs[thr]["fair_odds"])
            for thr in thresholds_sorted
            if ladder_probs[thr]["p_over_calib"] >= target_prob
        ]
        print()
        if candidates:
            best_thr, best_p, best_fair = max(candidates, key=lambda x: x[0])
            best_label = f"{int(round(best_thr))}+"
            fair_str = "None" if best_fair is None else f"{best_fair:+d}"
            print(
                f"Recommended rung (target_prob={target_prob:.2f}):\n"
                f"  {best_label} with P(points >= {best_thr:.0f}) = {best_p:.3f}, fair odds {fair_str}"
            )
            best_info = {
                "threshold": best_thr,
                "p_over": best_p,
                "fair_odds": best_fair,
            }
        else:
            print(
                f"No threshold in this ladder has P(points >= T) >= {target_prob:.2f} "
                f"(consider lowering --target-prob or choosing smaller thresholds)."
            )

    return best_info


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Scan star players' points markets and surface best bets + ladder best bets."
    )
    parser.add_argument(
        "--players",
        type=str,
        default=None,
        help="Comma-separated list of player names. Ignored if --auto-stars is used.",
    )
    parser.add_argument(
        "--auto-stars",
        action="store_true",
        help="Infer stars from the odds slate (based on max points line >= --min-line).",
    )
    parser.add_argument(
        "--odds-file",
        required=True,
        help="Path to odds slate CSV (e.g. data/odds_slate.csv).",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season year to consider for form (default 2023).",
    )
    parser.add_argument(
        "--min-line",
        type=float,
        default=20.0,
        help="(auto-stars) minimum max line for a player to be considered a star.",
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=10,
        help="(auto-stars) maximum number of stars to include.",
    )
    parser.add_argument(
        "--books",
        type=str,
        default=None,
        help="Comma-separated list of book names to keep (case-insensitive).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top props overall to show by edge.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=3.0,
        help="Minimum edge%% for display (does not affect how bets are chosen, only printing).",
    )
    parser.add_argument(
        "--ladder-thresholds",
        type=str,
        default="15,20,25,30,35,40",
        help="Comma-separated ladder thresholds, e.g. '15,20,25,30,35,40'. "
        "Each threshold T is treated as 'T+ points' using market line (T - 0.5).",
    )
    parser.add_argument(
        "--target-prob",
        type=float,
        default=0.60,
        help=(
            "Target probability for highlighting the 'aggressive' ladder rung "
            "in the model ladder (default 0.60)."
        ),
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=60,
        help="Skip a player if their latest log is older than this many days before the slate game_date.",
    )
    parser.add_argument(
        "--player-positions-csv",
        type=str,
        default=str(PLAYER_POSITIONS_CSV_DEFAULT),
        help="Optional player_id→position mapping used to recompute opp_dvp_pos_* features.",
    )
    # UPDATED (wording only)
    parser.add_argument(
        "--parlay-legs",
        type=int,
        default=0,
        help="If >0, print recommended-rung parlays with this many legs (unique games).",
    )

    args = parser.parse_args()

    ladder_thresholds = [
        float(x.strip()) for x in args.ladder_thresholds.split(",") if x.strip()
    ]
    ladder_thresholds = sorted(ladder_thresholds)

    # Load logs
    if not LOGS_CSV.exists():
        raise FileNotFoundError(f"Logs file not found: {LOGS_CSV}")
    logs_df = pd.read_csv(LOGS_CSV)
    print(f"Loaded player logs from {LOGS_CSV} with {len(logs_df):,} rows.")

    # Load features for team pace + relational/star-tier stuff (+ DvP table build)
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_CSV}")
    features_df = pd.read_csv(FEATURES_CSV)
    features_df["game_date_parsed"] = pd.to_datetime(features_df["game_date"]).dt.date

    # Load player positions + build DvP as-of table (optional)
    player_id_to_pos = load_player_positions(Path(args.player_positions_csv))
    dvp_asof_table = build_dvp_asof_table(features_df, player_id_to_pos, windows=(5, 15)) if player_id_to_pos else None

    # Load odds slate
    odds_path = Path(args.odds_file)
    if not odds_path.exists():
        raise FileNotFoundError(f"Odds file not found: {odds_path}")
    odds_df = pd.read_csv(odds_path)
    print(f"Loaded odds slate from {odds_path} with {len(odds_df):,} rows.")

    # Derive game_date from commence_time (UTC → America/Toronto) if missing
    if "game_date" not in odds_df.columns:
        if "commence_time" in odds_df.columns:
            dates_utc = pd.to_datetime(
                odds_df["commence_time"], errors="coerce", utc=True
            )
        elif "commence_time_utc" in odds_df.columns:
            dates_utc = pd.to_datetime(
                odds_df["commence_time_utc"], errors="coerce", utc=True
            )
        else:
            raise SystemExit(
                "No game_date / commence_time column found to derive game_date."
            )

        dates_local = dates_utc.dt.tz_convert("America/Toronto")
        odds_df["game_date"] = dates_local.dt.date

    # Filter to relevant books if requested
    odds_df = filter_books(odds_df, args.books)

    # Load model + sigma + calibrator
    model, base_sigma, feature_cols = load_regression_model(MODEL_PATH)
    sigma_model, sigma_feature_cols, sigma_config = load_sigma_model(SIGMA_MODEL_PATH)
    calibrator = load_over_prob_calibrator(CALIBRATOR_PATH)

    used_cols = feature_cols if feature_cols else FEATURE_COLS

    # Decide star list
    if args.auto_stars:
        slate_date, star_list = infer_slate_date_and_auto_stars(
            odds_df, min_line=args.min_line, max_stars=args.max_stars
        )
        players_to_process = [s["player"] for s in star_list]
        use_slate_date = slate_date
    else:
        if not args.players:
            raise SystemExit("Either --players or --auto-stars must be provided.")
        players_to_process = [p.strip() for p in args.players.split(",") if p.strip()]
        use_slate_date = None

    best_per_star: List[Dict[str, Any]] = []
    all_scored: List[Dict[str, Any]] = []
    recommended_rungs_summary: List[Dict[str, Any]] = []

    # Existing: collect ladder legs (still used for per-player ladder printing/hybrid)
    all_ladder_legs: List[Dict[str, Any]] = []

    # NEW: collect ONLY each player's single recommended rung (market-backed) for parlay building
    recommended_rung_legs: List[Dict[str, Any]] = []

    # Process each player
    for player_name_input in players_to_process:
        # Subset odds for this player
        p_mask = odds_df["player"].astype(str) == player_name_input
        player_odds_all = odds_df[p_mask]

        if player_odds_all.empty:
            print(f"\n[PLAYER] {player_name_input} – no odds rows found in odds file.")
            continue

        if "game_date" not in player_odds_all.columns:
            raise ValueError("odds file must have a 'game_date' column.")

        player_odds_all = player_odds_all.copy()
        player_odds_all["game_date_parsed"] = pd.to_datetime(
            player_odds_all["game_date"]
        ).dt.date

        if use_slate_date is not None:
            game_date = use_slate_date
            player_odds = player_odds_all[
                player_odds_all["game_date_parsed"] == game_date
            ]
        else:
            game_date = player_odds_all["game_date_parsed"].min()
            player_odds = player_odds_all[
                player_odds_all["game_date_parsed"] == game_date
            ]

        if player_odds.empty:
            print(
                f"\n[PLAYER] {player_name_input} – no odds on chosen game date {game_date}, skipping."
            )
            continue

        # Resolve player from logs
        player_id, player_name_resolved, team_abbrev_logs, season, latest_game_date = (
            resolve_player_from_logs(
                logs_df, player_name_input, args.season_min
            )
        )

        # Staleness guard (prevents “months old last game” inference)
        days_stale = (game_date - latest_game_date).days
        if days_stale < 0 or days_stale > int(args.max_stale_days):
            print(
                f"\n[PLAYER] {player_name_resolved} – skipping (days_stale={days_stale}, max_stale_days={args.max_stale_days})."
            )
            continue

        # Infer team context from odds slate
        team_abbrev_used, opp_abbrev_used, home_away_used = infer_team_context_from_odds(
            player_odds, default_team_abbrev=team_abbrev_logs
        )

        opp_label = opp_abbrev_used if opp_abbrev_used != "UNK" else "UNK"
        print(
            f"\n[PLAYER] {player_name_resolved} ({team_abbrev_used}) – using odds slate game on "
            f"{game_date} vs {opp_label} ({home_away_used.upper()})"
        )

        # Determine pos bucket for DvP (optional)
        player_pos_bucket = player_id_to_pos.get(player_id) if player_id_to_pos else None

        # Build features for this game
        feat_row = build_feature_row_for_game(
            logs_df=logs_df,
            features_df=features_df,
            player_id=player_id,
            team_abbrev=team_abbrev_used,
            season=season,
            target_date=game_date,
            opp_abbrev=opp_abbrev_used,
            home_away=home_away_used,
            player_pos_bucket=player_pos_bucket,
            dvp_asof_table=dvp_asof_table,
        )

        # Ensure all model features exist (handles newly added columns gracefully)
        feat_row = align_features(feat_row, used_cols, fill_value=0.0)

        # Predict mean (mu)
        X = feat_row[used_cols].to_numpy()
        mu = float(model.predict(X)[0])

        # Predict sigma (if sigma model available), else use base_sigma
        sigma_used = base_sigma
        if sigma_model is not None:
            sigma_features = feat_row.copy()
            sigma_features["mu_hat"] = mu
            sigma_features = align_features(sigma_features, sigma_feature_cols, fill_value=0.0)
            X_sigma = sigma_features[sigma_feature_cols].to_numpy()

            sigma_pred = float(sigma_model.predict(X_sigma)[0])
            use_log = bool(sigma_config.get("use_log_target", True))
            eps = float(sigma_config.get("eps", 1e-3))
            sigma_scale = float(sigma_config.get("sigma_scale", 1.0))

            if use_log:
                sigma_pred = math.exp(sigma_pred)
            sigma_used = max(sigma_pred * sigma_scale, eps)

        print(f"  Model expected points (mu): {mu:.2f} (sigma_used={sigma_used:.3f})")

        # --- Pure model ladder for this star (independent of market lines) ---
        ladder_probs_for_star = compute_ladder_probs(
            mu=mu,
            sigma=sigma_used,
            thresholds=ladder_thresholds,
            calibrator=calibrator,
        )
        best_rung_info = print_model_ladder_for_star(
            player_name_resolved,
            ladder_probs_for_star,
            target_prob=args.target_prob,
        )
        if best_rung_info is not None:
            recommended_rungs_summary.append(
                {
                    "player": player_name_resolved,
                    "team": team_abbrev_used,
                    "threshold": int(round(best_rung_info["threshold"])),
                    "p_over": best_rung_info["p_over"],
                    "fair_odds": best_rung_info["fair_odds"],
                }
            )

            # NEW: create a market-backed leg ONLY for this exact recommended threshold
            rec_thr = float(best_rung_info["threshold"])
            rec_line = rec_thr - 0.5
            p_win_rec = float(best_rung_info["p_over"])

            best_market = best_market_over_for_line(player_odds, rec_line)
            if best_market is None:
                print(
                    f"  [Recommended-rung parlay] No market found for {player_name_resolved} "
                    f"{int(round(rec_thr))}+ (line={rec_line:.1f}). Skipping for recommended-rung parlays."
                )
            else:
                odds_val = float(best_market["odds"])
                profit_on_win = american_to_profit_per_unit(odds_val)
                ev = p_win_rec * profit_on_win - (1.0 - p_win_rec)

                recommended_rung_legs.append(
                    {
                        "player": player_name_resolved,
                        "book": best_market["book"],
                        "threshold": rec_thr,
                        "market_line": float(rec_line),
                        "odds": odds_val,
                        "p_win": p_win_rec,
                        "fair_odds": best_rung_info.get("fair_odds"),
                        "ev_per_unit": ev,
                        "event_id": best_market.get("event_id", None),
                        "game_date": best_market.get("game_date", str(game_date)),
                        "home_team": best_market.get("home_team", None),
                        "away_team": best_market.get("away_team", None),
                    }
                )

        # Score all available props for this player on this date
        scored_for_player: List[Dict[str, Any]] = []
        for _, r in player_odds.iterrows():
            side = str(r["side"]).lower().strip()
            if side not in {"over", "under"}:
                continue

            line = float(r["line"])
            odds_val = float(r["odds"])

            score = score_points_prop(
                mu=mu,
                sigma=sigma_used,
                line=line,
                side=side,
                odds=odds_val,
                calibrator=calibrator,
            )

            scored_row = {
                "player": player_name_resolved,
                "book": r["book"],
                "line": line,
                "side": side,
                "odds": odds_val,
                "p_win": score["p_win"],
                "fair_odds": score["fair_odds"],
                "edge_pct": score["edge_pct"],
                "ev_per_unit": score["ev_per_unit"],
                # NEW: for uncorrelated parlay building
                "event_id": r.get("event_id", None),
                "home_team": r.get("home_team", None),
                "away_team": r.get("away_team", None),
                "game_date": str(game_date),
            }
            scored_for_player.append(scored_row)
            all_scored.append(scored_row)

        if not scored_for_player:
            print("  [WARN] No valid over/under points props found for this player.")
            continue

        # Best bet per star by EV
        best = max(scored_for_player, key=lambda d: d["ev_per_unit"])
        best_per_star.append(best)

        # Ladder best bets for this star (tied to actual market lines, if any)
        ladder_best_rows = compute_ladder_best_bets_for_star(
            player_odds=player_odds,
            mu=mu,
            sigma=sigma_used,
            calibrator=calibrator,
            ladder_thresholds=ladder_thresholds,
        )
        # Make sure resolved player name carries through on ladder rows
        for lr in ladder_best_rows:
            lr["player"] = player_name_resolved
            if not lr.get("game_date"):
                lr["game_date"] = str(game_date)

        all_ladder_legs.extend(ladder_best_rows)

        print_ladder_best_bets_for_star(
            player_name_resolved,
            ladder_best_rows,
            args.min_edge,
        )

        # ---------------- NEW: Hybrid ladder recommendation (parlay-first) ----------------
        hybrid_rec = pick_hybrid_ladder_recommendation(
            ladder_best_rows=ladder_best_rows,
            target_prob=args.target_prob,
        )
        print_hybrid_ladder_recommendation(
            player_name=player_name_resolved,
            rec=hybrid_rec,
            target_prob=args.target_prob,
        )
        # ---------------------------------------------------------------------------------

    # Print best bet per star
    if best_per_star:
        print_best_per_star_table(
            best_per_star,
            min_edge=args.min_edge,
            header="Best bet per star",
        )
    else:
        print("\n(No best bets per star — no players processed or no props found.)")

    # Print global top-K
    if all_scored:
        print_top_k_table(all_scored, top_k=args.top_k, min_edge=args.min_edge)
    else:
        print("\n(No scored props overall — nothing to show for top-K.)")

    # Print compact summary of recommended model rungs per star, grouped by team
    if args.target_prob is not None and recommended_rungs_summary:
        print(
            "\n=== Recommended model rungs (one per star, based on target_prob) ==="
        )

        # sort by team, then player for stable grouping
        rows_sorted = sorted(
            recommended_rungs_summary,
            key=lambda r: (r["team"], r["player"])
        )

        current_team = None
        for r in rows_sorted:
            team = r["team"]
            fair = r["fair_odds"]
            fair_str = "None" if fair is None else f"{fair:+d}"

            # when we hit a new team, print a team header + column header
            if team != current_team:
                current_team = team
                print(f"\nTeam {team}")
                print(f"{'Player':20s} | {'Thr+':>5s} | {'p_over':>7s} | {'Fair':>7s}")
                print("-" * 48)

            print(
                f"{r['player'][:20]:20s} | {r['threshold']:5d}+ | "
                f"{r['p_over']:7.3f} | {fair_str:>7s}"
            )

    # NEW: recommended-rung parlays ONLY (uses each player's single model Thr+)
    if args.parlay_legs and args.parlay_legs > 0:
        eligible_recommended = [
            r for r in recommended_rung_legs
            if float(r.get("p_win", 0.0)) >= float(args.target_prob)
        ]
        parlays = build_many_uncorrelated_ladder_parlays(
            ladder_legs=eligible_recommended,
            legs_per_parlay=int(args.parlay_legs),
        )
        print_recommended_rung_parlays(parlays)


if __name__ == "__main__":
    main()