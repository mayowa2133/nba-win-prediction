#!/usr/bin/env python
"""
scan_slate.py

Scan a slate of player points props and evaluate model edge / EV for each.

Input CSV (example):

    player,line,side,odds,book
    Devin Booker,23.5,over,-115,Bet365
    Luka Doncic,29.5,under,+105,Bet365

Required columns:
  - player : player name (e.g. "Devin Booker")
  - line   : points line (float, e.g. 23.5)
  - side   : "over" or "under"
  - odds   : American odds (e.g. -115, +105)

Optional columns (passed through to output):
  - book, notes, etc.
  - For rows coming from The Odds API, we also expect:
      home_team, away_team, commence_time (UTC ISO string)

It uses:
  - data/player_game_logs.csv
  - models/points_regression.pkl
  - models/over_prob_calibrator.pkl  (if present)

Usage:

  python scan_slate.py --input data/slate.csv

This will write data/slate_scored.csv with extra columns:
  game_date,team,opp,home_away,mu,p_over_raw,p_over_calib,
  p_win_side,implied_prob,edge_pct,ev_per_unit, ...

You can also set a minimum edge to highlight in the console:

  python scan_slate.py --input data/slate.csv --min-edge 5.0
"""

import argparse
import datetime as dt
import math
import pickle
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

LOGS_CSV = Path("data/player_game_logs.csv")
MODEL_PATH = Path("models/points_regression.pkl")
CALIBRATOR_PATH = Path("models/over_prob_calibrator.pkl")

# These MUST match the feature columns used to train points_regression.pkl
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
    "opp_pts_allowed_roll5",
    "opp_pts_allowed_roll15",
    "days_since_last_game",
    "is_home",
]


# ----------------------------------------------------------------------
# Team label ↔ abbreviation helpers (for Odds API rows)
# ----------------------------------------------------------------------

def _build_team_label_map() -> Dict[str, str]:
    """
    Build a mapping from various team labels (full name, nickname, city+nickname,
    abbreviation) to the standard abbreviation used in logs (e.g. 'LAL').
    """
    mapping: Dict[str, str] = {}
    for t in teams.get_teams():
        abbr = t["abbreviation"]
        full_name = t.get("full_name") or ""
        nickname = t.get("nickname") or ""
        city = t.get("city") or ""

        def add(key: str):
            key = key.strip().lower()
            if key:
                mapping.setdefault(key, abbr)

        add(full_name)              # "Los Angeles Lakers"
        add(nickname)               # "Lakers"
        add(f"{city} {nickname}")   # "Los Angeles Lakers" (again)
        add(abbr)                   # "LAL"

    return mapping


TEAM_LABEL_TO_ABBREV = _build_team_label_map()


def map_team_label_to_abbrev(label: Optional[str]) -> Optional[str]:
    """
    Map a label like 'Los Angeles Lakers' or 'Lakers' or 'LAL' to 'LAL'.
    Returns None if we can't map it.
    """
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    return TEAM_LABEL_TO_ABBREV.get(key)


def parse_commence_date(row: Dict[str, Any]) -> Optional[dt.date]:
    """
    Parse The Odds API 'commence_time' (UTC ISO8601) and convert it
    to a local date in America/New_York (to roughly match logs/model).
    """
    ct = row.get("commence_time") or row.get("commence_time_utc")
    if not isinstance(ct, str) or not ct.strip():
        return None

    s = ct.strip()
    # The Odds API uses 'Z' for UTC, Python wants '+00:00'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt_utc = dt.datetime.fromisoformat(s)
    except ValueError:
        return None

    try:
        dt_local = dt_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        # If zoneinfo fails for some reason, just use UTC date
        dt_local = dt_utc

    return dt_local.date()


def infer_matchup_from_odds_row(row: Dict[str, Any], player_team_abbrev: str):
    """
    For a row coming from The Odds API (odds_slate.csv), infer:
      - game_date (local date)
      - opponent abbreviation
      - home_away ('home' or 'away')

    Uses:
      - home_team / away_team labels from the odds row
      - TEAM_LABEL_TO_ABBREV mapping
      - commence_time -> local date

    Returns:
      (game_date: date, opp_abbrev: str, home_away: str)
      or (None, None, None) if we cannot infer.
    """
    player_team_abbrev = (player_team_abbrev or "").upper().strip()
    if not player_team_abbrev:
        return None, None, None

    home_label = row.get("home_team") or row.get("home")
    away_label = row.get("away_team") or row.get("away")

    home_abbrev = map_team_label_to_abbrev(home_label)
    away_abbrev = map_team_label_to_abbrev(away_label)

    if home_abbrev is None or away_abbrev is None:
        return None, None, None

    game_date = parse_commence_date(row)
    if game_date is None:
        return None, None, None

    if player_team_abbrev == home_abbrev:
        opp_abbrev = away_abbrev
        home_away = "home"
    elif player_team_abbrev == away_abbrev:
        opp_abbrev = home_abbrev
        home_away = "away"
    else:
        # Player team doesn't match either side in this event
        return None, None, None

    return game_date, opp_abbrev, home_away


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def normalize_name(s: Any) -> str:
    """Strip accents, simple punctuation, lower-case, for matching player names."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    # Remove simple punctuation that often differs between sources
    for ch in [".", "'", ","]:
        s = s.replace(ch, "")

    return s.lower().strip()


def rolling_mean_last_n(series: pd.Series, n: int) -> float:
    """Mean of last n values (or fewer if not enough games). 0 if no games."""
    if series is None or len(series) == 0:
        return 0.0
    return float(series.tail(min(n, len(series))).mean())


def prob_to_american(p: float) -> Optional[int]:
    """Convert probability p to American odds. Returns None for p<=0 or p>=1."""
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        # favorite (negative odds)
        return int(round(-100 * p / (1 - p)))
    else:
        # underdog (positive odds)
        return int(round(100 * (1 - p) / p))


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (no vig)."""
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return -odds / (-odds + 100.0)
    else:
        raise ValueError("Odds cannot be 0.")


def american_to_profit_per_unit(odds: float) -> float:
    """
    Profit per 1 unit stake if the bet wins at the given American odds.

    e.g.
      -115 -> profit_on_win ≈ 0.8696
      +105 -> profit_on_win = 1.05
    """
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    elif odds < 0:
        return 100.0 / abs(odds)
    else:
        raise ValueError("Odds cannot be 0.")


def normal_over_probs(mu: float, sigma: float, line: float) -> Tuple[float, float]:
    """
    Approximate P(OVER line) and P(UNDER line) using N(mu, sigma^2).
    """
    if sigma <= 0:
        return (1.0 if mu > line else 0.0, 1.0 if mu <= line else 0.0)

    z = (line - mu) / sigma
    # CDF for standard normal using erf
    p_under = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_over = 1.0 - p_under
    return p_over, p_under


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_regression_model(path: Path = MODEL_PATH):
    """
    Load the regression bundle {model, sigma, feature_cols} from pickle.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle["model"]
        sigma = float(bundle.get("sigma", 7.0))
        feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    else:
        # Fallback if only model was pickled
        model = bundle
        sigma = 7.0
        feature_cols = FEATURE_COLS

    return model, sigma, feature_cols


def load_over_prob_calibrator(path: Path = CALIBRATOR_PATH):
    """
    Load isotonic regression calibrator for P(OVER), if present.
    Returns None if file does not exist.
    """
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
# Schedule lookup using ScoreboardV2 + team IDs (fallback only)
# ----------------------------------------------------------------------

def find_next_game_for_team(
    team_abbrev: str,
    start_date: dt.date,
    max_days_ahead: int = 30,
    sleep_s: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """
    Look ahead from `start_date` up to `max_days_ahead` days using ScoreboardV2
    and return the first game involving `team_abbrev` (by team ID).

    Returns a dict with:
      {
        "game_id": str,
        "game_date": date,
        "opp_abbrev": str,
        "home_away": "home" | "away",
      }
    or None if nothing found.

    NOTE: This is now used as a fallback only. For The Odds API rows, we
    infer matchup directly from home_team/away_team/commence_time.
    """
    team_info = teams.find_team_by_abbreviation(team_abbrev)
    if not team_info:
        print(f"[WARN] Could not map team abbreviation '{team_abbrev}' to an NBA team id.")
        return None

    team_id = int(team_info["id"])

    for offset in range(max_days_ahead + 1):
        d = start_date + dt.timedelta(days=offset)
        date_str = d.strftime("%m/%d/%Y")

        try:
            sb = ScoreboardV2(game_date=date_str, league_id="00")
            games_df = sb.game_header.get_data_frame()
        except Exception as e:
            print(f"    [ScoreboardV2 error on {date_str}: {e}]")
            time.sleep(sleep_s)
            continue

        if games_df.empty:
            time.sleep(sleep_s)
            continue

        # Filter rows where our team is either home or away
        mask = (games_df["HOME_TEAM_ID"] == team_id) | (games_df["VISITOR_TEAM_ID"] == team_id)
        cand = games_df.loc[mask]

        if cand.empty:
            time.sleep(sleep_s)
            continue

        # Take the first matching game as the "next" one
        row = cand.iloc[0]
        is_home = row["HOME_TEAM_ID"] == team_id
        opp_team_id = int(row["VISITOR_TEAM_ID"] if is_home else row["HOME_TEAM_ID"])

        # Map opponent ID -> abbrev
        opp_info = teams.find_team_name_by_id(opp_team_id)
        opp_abbrev = opp_info["abbreviation"] if opp_info else "UNK"

        return {
            "game_id": row["GAME_ID"],
            "game_date": d,
            "opp_abbrev": opp_abbrev,
            "home_away": "home" if is_home else "away",
        }

    return None


# ----------------------------------------------------------------------
# Feature construction for NEXT game
# ----------------------------------------------------------------------

def build_next_game_feature_row(
    logs_df: pd.DataFrame,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    season: int,
    target_date: dt.date,
    opp_abbrev: str,
    home_away: str,
) -> pd.DataFrame:
    """
    Build one feature row for the player's next game vs opp_abbrev on target_date.

    - Uses all games for this player in this season with game_date < target_date
    - Uses all games for opp team in this season with game_date < target_date
    """
    df = logs_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Filter player games in this season, before target_date
    player_games = df[
        (df["player_id"] == player_id) &
        (df["season"] == season) &
        (df["game_date"] < pd.Timestamp(target_date))
    ].sort_values("game_date")

    if player_games.empty:
        raise RuntimeError(
            f"No historical games found for player_id={player_id} in season {season} "
            f"before {target_date}. Can't build rolling features."
        )

    last_game_date = player_games["game_date"].max().date()
    days_since_last_game = (target_date - last_game_date).days

    # Rolling form for player (use last N games, or fewer if not enough)
    minutes_roll5 = rolling_mean_last_n(player_games["minutes"], 5)
    minutes_roll15 = rolling_mean_last_n(player_games["minutes"], 15)

    pts_roll5 = rolling_mean_last_n(player_games["pts"], 5)
    pts_roll15 = rolling_mean_last_n(player_games["pts"], 15)

    reb_roll5 = rolling_mean_last_n(player_games["reb"], 5)
    reb_roll15 = rolling_mean_last_n(player_games["reb"], 15)

    ast_roll5 = rolling_mean_last_n(player_games["ast"], 5)
    ast_roll15 = rolling_mean_last_n(player_games["ast"], 15)

    fg3m_roll5 = rolling_mean_last_n(player_games["fg3m"], 5)
    fg3m_roll15 = rolling_mean_last_n(player_games["fg3m"], 15)

    fg3a_roll5 = rolling_mean_last_n(player_games["fg3a"], 5)
    fg3a_roll15 = rolling_mean_last_n(player_games["fg3a"], 15)

    fga_roll5 = rolling_mean_last_n(player_games["fga"], 5)
    fga_roll15 = rolling_mean_last_n(player_games["fga"], 15)

    fta_roll5 = rolling_mean_last_n(player_games["fta"], 5)
    fta_roll15 = rolling_mean_last_n(player_games["fta"], 15)

    # Team-level defensive form for opponent
    team_games = (
        df.groupby(
            ["season", "game_id", "team_abbrev", "opp_abbrev", "game_date"],
            as_index=False,
        )
        .agg({"team_score": "max", "opp_score": "max"})
    )

    opp_games = team_games[
        (team_games["season"] == season) &
        (team_games["team_abbrev"] == opp_abbrev) &
        (team_games["game_date"] < pd.Timestamp(target_date))
    ].sort_values("game_date")

    if opp_games.empty:
        # Worst-case: no history yet this season (very early).
        # Just fall back to zero-ish values; model will mostly ignore these.
        opp_pts_allowed_roll5 = 0.0
        opp_pts_allowed_roll15 = 0.0
    else:
        opp_pts_allowed_roll5 = rolling_mean_last_n(opp_games["opp_score"], 5)
        opp_pts_allowed_roll15 = rolling_mean_last_n(opp_games["opp_score"], 15)

    is_home = 1 if home_away.lower() == "home" else 0

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
        "opp_pts_allowed_roll5": opp_pts_allowed_roll5,
        "opp_pts_allowed_roll15": opp_pts_allowed_roll15,
        "days_since_last_game": float(days_since_last_game),
        "is_home": is_home,
    }

    feat_df = pd.DataFrame([row])
    # Make sure no NaNs sneak in
    feat_df = feat_df.fillna(0.0)
    return feat_df


# ----------------------------------------------------------------------
# Player resolution
# ----------------------------------------------------------------------

def resolve_player_from_logs(
    logs_df: pd.DataFrame,
    player_query: str,
    season_min: int,
) -> Tuple[int, str, str, int, dt.date]:
    """
    Use player_game_logs.csv to resolve a player name to:
      - player_id
      - canonical player_name
      - current team_abbrev (from latest game)
      - latest season
      - latest game_date
    """
    df = logs_df.copy()
    df["name_norm"] = df["player_name"].map(normalize_name)

    target = normalize_name(player_query)

    # Exact normalized match first
    candidates = df[df["name_norm"] == target]

    if candidates.empty:
        # Try substring match
        candidates = df[df["name_norm"].str.contains(target, na=False)]

    if candidates.empty:
        # Give a few examples to help debugging
        sample_names = df["player_name"].dropna().unique().tolist()
        print(f"Could not resolve player name '{player_query}' in logs.")
        if sample_names:
            print("Here are some example names from the file:")
            for n in sample_names[:30]:
                print(f"  - {n}")
        # Raise a normal error so scan_slate can skip this row instead of exiting
        raise ValueError(f"Could not resolve player name '{player_query}' in logs.")

    # Filter to seasons >= season_min if possible
    cand2 = candidates[candidates["season"] >= season_min]
    if cand2.empty:
        cand2 = candidates

    cand2 = cand2.copy()
    cand2["game_date"] = pd.to_datetime(cand2["game_date"])

    # Latest game
    latest = cand2.sort_values(["game_date", "game_id"]).iloc[-1]

    player_id = int(latest["player_id"])
    player_name = str(latest["player_name"])
    team_abbrev = str(latest["team_abbrev"])
    season = int(latest["season"])
    latest_game_date = latest["game_date"].date()

    return player_id, player_name, team_abbrev, season, latest_game_date


# ----------------------------------------------------------------------
# Scoring a single prop row
# ----------------------------------------------------------------------

def score_single_prop_row(
    row: Dict[str, Any],
    logs_df: pd.DataFrame,
    model,
    sigma: float,
    feature_cols,
    calibrator,
    season_min: int,
) -> Dict[str, Any]:
    """
    Given a dict-like row with keys:
      - player, line, side, odds
    compute model-based edge / EV for that prop and return a dict of fields
    to merge into the output.
    """
    player_input = str(row["player"])
    line = float(row["line"])
    side_raw = str(row["side"]).strip().lower()
    if side_raw not in {"over", "under"}:
        raise ValueError(f"side must be 'over' or 'under', got {row['side']!r}")

    odds = float(row["odds"])

    # Resolve player & team from logs
    player_id, player_name, team_abbrev, season, latest_game_date = resolve_player_from_logs(
        logs_df, player_input, season_min
    )

    # ------------------------------------------------------------------
    # Decide matchup: date, opponent, home/away
    # ------------------------------------------------------------------
    opp_abbrev: Optional[str] = None
    home_away: Optional[str] = None
    target_date: Optional[dt.date] = None

    # 1) If the row explicitly has game info (manual slate), use that.
    row_game_date = row.get("game_date")
    row_opp = row.get("opp")
    row_home_away = row.get("home_away")

    if row_game_date and row_opp and row_home_away:
        try:
            target_date = dt.date.fromisoformat(str(row_game_date))
        except Exception:
            target_date = None

        opp_abbrev = str(row_opp).upper().strip()
        home_away = str(row_home_away).lower().strip()

    # 2) Otherwise, if this is a The Odds API row with commence_time + home/away labels,
    #    infer matchup directly from that (this avoids ScoreboardV2).
    if (target_date is None or opp_abbrev is None or home_away is None) and row.get("commence_time"):
        gd, opp, ha = infer_matchup_from_odds_row(row, team_abbrev)
        if gd is not None and opp is not None and ha is not None:
            target_date = gd
            opp_abbrev = opp
            home_away = ha

    # 3) Fallback: use ScoreboardV2 to find the next game (expensive; try to avoid).
    if target_date is None or opp_abbrev is None or home_away is None:
        today = dt.date.today()
        start_date = max(today, latest_game_date + dt.timedelta(days=1))
        info = find_next_game_for_team(team_abbrev, start_date=start_date, max_days_ahead=30)

        if info is None:
            raise RuntimeError(
                f"Could not find next scheduled game for team {team_abbrev} within "
                f"30 days of {start_date} using ScoreboardV2."
            )

        target_date = info["game_date"]
        opp_abbrev = info["opp_abbrev"]
        home_away = info["home_away"]

    # ------------------------------------------------------------------
    # Build features for this next game
    # ------------------------------------------------------------------
    feat_row = build_next_game_feature_row(
        logs_df=logs_df,
        player_id=player_id,
        player_name=player_name,
        team_abbrev=team_abbrev,
        season=season,
        target_date=target_date,
        opp_abbrev=opp_abbrev,
        home_away=home_away,
    )

    used_cols = feature_cols if feature_cols else FEATURE_COLS
    X = feat_row[used_cols].to_numpy()
    mu = float(model.predict(X)[0])

    # Raw P(OVER) from normal approximation
    p_over_raw, p_under_raw = normal_over_probs(mu, sigma, line)

    # Calibrate if we have a calibrator
    if calibrator is not None:
        p_over_calib = float(calibrator.predict([p_over_raw])[0])
    else:
        p_over_calib = p_over_raw

    # Clip for numerical sanity
    p_over_calib = min(max(p_over_calib, 1e-6), 1.0 - 1e-6)
    p_under_calib = 1.0 - p_over_calib

    # Probability our chosen side wins
    if side_raw == "over":
        p_win = p_over_calib
    else:
        p_win = p_under_calib

    implied_p = american_to_prob(odds)
    profit_on_win = american_to_profit_per_unit(odds)
    edge = p_win - implied_p
    edge_pct = edge * 100.0

    ev_per_unit = p_win * profit_on_win - (1.0 - p_win) * 1.0
    fair_odds_for_side = prob_to_american(p_win)

    return {
        "player_input": player_input,
        "player_resolved": player_name,
        "team": team_abbrev,
        "opp": opp_abbrev,
        "home_away": home_away,
        "game_date": target_date.isoformat(),
        "line": line,
        "side": side_raw,
        "odds": int(odds),
        "implied_prob": implied_p,
        "mu": mu,
        "p_over_raw": p_over_raw,
        "p_over_calib": p_over_calib,
        "p_win_side": p_win,
        "edge_pct": edge_pct,
        "ev_per_unit": ev_per_unit,
        "fair_odds_for_side": fair_odds_for_side,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan a slate of props and compute model edge / EV for each."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with columns: player,line,side,odds[,book,...]",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for output CSV. "
             "Defaults to <input_stem>_scored.csv in same folder.",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season start year to consider for form (default 2023).",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="If >0, print summary of props with edge_pct >= this value.",
    )

    args = parser.parse_args()

    # Load logs
    if not LOGS_CSV.exists():
        raise FileNotFoundError(f"Logs file not found: {LOGS_CSV}")
    logs_df = pd.read_csv(LOGS_CSV)
    print(f"Loaded raw player logs from {LOGS_CSV} with {len(logs_df):,} rows.")

    # Load model + calibrator
    model, sigma, feature_cols = load_regression_model(MODEL_PATH)
    calibrator = load_over_prob_calibrator(CALIBRATOR_PATH)

    # Load slate
    input_path = Path(args.input)
    slate_df = pd.read_csv(input_path)
    print(f"Loaded slate from {input_path} with {len(slate_df):,} rows.")

    required_cols = {"player", "line", "side", "odds"}
    missing = required_cols.difference(slate_df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    results = []
    for idx, row in slate_df.iterrows():
        row_dict = row.to_dict()
        try:
            scored = score_single_prop_row(
                row=row_dict,
                logs_df=logs_df,
                model=model,
                sigma=sigma,
                feature_cols=feature_cols,
                calibrator=calibrator,
                season_min=args.season_min,
            )
            # merge original row + scored fields (scored overrides on conflicts)
            merged = {**row_dict, **scored}
        except Exception as e:
            print(f"[WARN] Failed to score row {idx}: {e}")
            merged = {**row_dict}
            merged["error"] = str(e)
        results.append(merged)

    out_df = pd.DataFrame(results)

    # Decide output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_scored.csv")

    out_df.to_csv(output_path, index=False)
    print(f"\nWrote scored slate to {output_path}")

    # Optional: console summary of best edges
    if "edge_pct" in out_df.columns and args.min_edge > 0.0:
        print(f"\nProps with edge_pct >= {args.min_edge:.1f} (sorted by edge):")
        candidates = out_df[out_df["edge_pct"] >= args.min_edge].sort_values(
            "edge_pct", ascending=False
        )

        if candidates.empty:
            print("  (none)")
        else:
            for _, r in candidates.head(30).iterrows():
                if "book" in r:
                    book = f" [{r['book']}]"
                else:
                    book = ""
                print(
                    f"  {r.get('player_resolved', r['player'])} "
                    f"{r['side'].upper()} {r['line']} @ {int(r['odds'])}{book} "
                    f"vs {r.get('opp','?')} ({r.get('home_away','?').upper()}) "
                    f"- edge={r['edge_pct']:.1f}% EV={r['ev_per_unit']:.3f} "
                    f"(p_win={r.get('p_win_side', float('nan')):.3f})"
                )


if __name__ == "__main__":
    main()