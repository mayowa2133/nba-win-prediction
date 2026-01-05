#!/usr/bin/env python
"""
star_ladder_vs_market.py

For a given player, do TWO things:

1) Use your regression model + calibrator to estimate the probability they hit
   various points thresholds (15+, 20+, 25+, 30+, etc.) in their NEXT game.

2) Compare those model probabilities to REAL PLAYER POINTS PROPS from an odds file
   (e.g. data/odds_slate.csv produced by fetch_props_from_the_odds_api.py), and
   compute edge / EV for each book line on that player.

It uses:
  - data/player_game_logs.csv
  - models/points_regression.pkl
  - models/over_prob_calibrator.pkl  (if present)
  - data/odds_slate.csv (or another odds file you pass in)

Example:

  # 1) Refresh odds file from The Odds API:
  python fetch_props_from_the_odds_api.py

  # 2) Run star ladder + market comparison:
  python star_ladder_vs_market.py \
    --player "Devin Booker" \
    --thresholds "15,20,25,30,35,40" \
    --season-min 2023 \
    --target-prob 0.60 \
    --odds-file data/odds_slate.csv \
    --books "Bet365,FanDuel,DraftKings"

"""

import argparse
import datetime as dt
import math
import pickle
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

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
# Small helpers
# ----------------------------------------------------------------------

def normalize_name(s: Any) -> str:
    """Strip accents, lower-case, for matching player names."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def rolling_mean_last_n(series: pd.Series, n: int) -> float:
    """Mean of last n values (or fewer if not enough games). 0 if no games."""
    if series is None or len(series) == 0:
        return 0.0
    return float(series.tail(min(n, len(series))).mean())


def prob_to_american(p: float) -> Optional[int]:
    """
    Convert probability p to American odds.

    Returns:
      - None if p is NaN, <=0, or >=1 (degenerate / invalid)
      - int American odds otherwise
    """
    if p is None:
        return None

    if isinstance(p, float) and math.isnan(p):
        return None

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
# Schedule lookup using ScoreboardV2 + team IDs
# ----------------------------------------------------------------------

def find_next_game_for_team(
    team_abbrev: str,
    start_date: dt.date,
    max_days_ahead: int = 60,
    sleep_s: float = 0.4,
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
        sample_names = df["player_name"].dropna().unique().tolist()
        print(f"Could not resolve player name '{player_query}' in logs.")
        if sample_names:
            print("Here are some example names from the file:")
            for n in sample_names[:30]:
                print(f"  - {n}")
        raise SystemExit(1)

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
# Ladder logic
# ----------------------------------------------------------------------

def compute_ladder(
    mu: float,
    sigma: float,
    thresholds: List[float],
    calibrator,
) -> List[Dict[str, Any]]:
    """
    For each threshold, compute raw + calibrated P(points >= threshold)
    and implied fair odds.
    Returns a list of dicts.
    """
    rows = []
    for line in thresholds:
        p_over_raw, _ = normal_over_probs(mu, sigma, line)

        if calibrator is not None:
            p_over_calib = float(calibrator.predict([p_over_raw])[0])
        else:
            p_over_calib = p_over_raw

        # If calibrator returns NaN for some reason, fall back to raw prob
        if isinstance(p_over_calib, float) and math.isnan(p_over_calib):
            p_over_calib = p_over_raw

        # Clip for sanity
        p_over_calib = min(max(p_over_calib, 1e-6), 1.0 - 1e-6)

        fair_odds = prob_to_american(p_over_calib)

        rows.append(
            {
                "threshold": line,
                "p_over_raw": p_over_raw,
                "p_over_calib": p_over_calib,
                "fair_odds": fair_odds,
            }
        )
    return rows


def print_ladder_table(rows: List[Dict[str, Any]]):
    """
    Pretty-print the ladder table.
    """
    print("\n=== Model points ladder (P(points >= threshold)) ===")
    print(f"{'Threshold':>10} | {'Raw p_over':>10} | {'Calib p_over':>13} | {'Fair odds':>9}")
    print("-" * 54)
    for r in rows:
        line = r["threshold"]
        p_raw = r["p_over_raw"]
        p_calib = r["p_over_calib"]
        fair_odds = r["fair_odds"]
        if fair_odds is None:
            fair_str = "degenerate"
        else:
            fair_str = f"{fair_odds:+d}"
        print(
            f"{line:10.1f} | {p_raw:10.3f} | {p_calib:13.3f} | {fair_str:>9}"
        )


def highlight_recommended_rung(rows: List[Dict[str, Any]], target_prob: float):
    """
    Find the highest threshold with calibrated p_over >= target_prob.
    """
    candidates = [r for r in rows if r["p_over_calib"] >= target_prob]
    if not candidates:
        print(f"\nNo threshold has calibrated P(>= line) >= {target_prob:.2f}.")
        return

    # highest threshold among candidates
    best = max(candidates, key=lambda r: r["threshold"])
    line = best["threshold"]
    p_calib = best["p_over_calib"]
    fair_odds = best["fair_oods"] if "fair_oods" in best else best["fair_odds"]
    if fair_odds is None:
        fair_str = "degenerate"
    else:
        fair_str = f"{fair_odds:+d}"

    print(
        f"\nRecommended aggressive rung (target_prob={target_prob:.2f}):\n"
        f"  Threshold: {line:.1f} points\n"
        f"  Calibrated P(>= {line:.1f}): {p_calib:.3f}\n"
        f"  Fair odds: {fair_str}"
    )


# ----------------------------------------------------------------------
# Market odds logic for this player
# ----------------------------------------------------------------------

def load_market_rows_for_player(
    odds_path: Path,
    player_name: str,
    books_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load rows from the odds file for the given player name.
    Expects columns at least: player, line, side, odds, [book].

    books_filter: optional list of book names to keep (exact match on 'book' col).
    """
    if not odds_path.exists():
        raise FileNotFoundError(f"Odds file not found: {odds_path}")

    odds_df = pd.read_csv(odds_path)

    required = {"player", "line", "side", "odds"}
    missing = required.difference(odds_df.columns)
    if missing:
        raise ValueError(f"Odds CSV {odds_path} is missing required columns: {missing}")

    odds_df["player_norm"] = odds_df["player"].map(normalize_name)
    target_norm = normalize_name(player_name)

    df_player = odds_df[odds_df["player_norm"] == target_norm].copy()

    if df_player.empty:
        print(f"[WARN] No odds rows found in {odds_path} for player '{player_name}'.")
        return df_player

    # Optional book filter
    if books_filter:
        books_filter_norm = {b.strip().lower() for b in books_filter}
        df_player["book_norm"] = df_player.get("book", "").astype(str).str.lower()
        df_player = df_player[df_player["book_norm"].isin(books_filter_norm)].copy()

    # Only keep columns we care about in the printout
    return df_player


def score_market_row(
    row: pd.Series,
    mu: float,
    sigma: float,
    calibrator,
) -> Dict[str, Any]:
    """
    For a single market row (line, side, odds), compute model prob / fair odds / edge / EV.
    """
    line = float(row["line"])
    side = str(row["side"]).lower().strip()
    odds = float(row["odds"])

    p_over_raw, p_under_raw = normal_over_probs(mu, sigma, line)

    if calibrator is not None:
        p_over_calib = float(calibrator.predict([p_over_raw])[0])
        # Fallback if NaN
        if isinstance(p_over_calib, float) and math.isnan(p_over_calib):
            p_over_calib = p_over_raw
    else:
        p_over_calib = p_over_raw

    # Clip for sanity
    p_over_calib = min(max(p_over_calib, 1e-6), 1.0 - 1e-6)
    p_under_calib = 1.0 - p_over_calib

    if side == "over":
        p_win = p_over_calib
    elif side == "under":
        p_win = p_under_calib
    else:
        # Unknown side, just return NaNs
        return {
            "model_p_win": math.nan,
            "model_fair_odds": None,
            "edge_pct": math.nan,
            "ev_per_unit": math.nan,
            "p_over_calib": p_over_calib,
            "p_under_calib": p_under_calib,
        }

    implied_p = american_to_prob(odds)
    profit_on_win = american_to_profit_per_unit(odds)
    edge = p_win - implied_p
    edge_pct = edge * 100.0
    ev_per_unit = p_win * profit_on_win - (1.0 - p_win) * 1.0
    fair_odds = prob_to_american(p_win)

    return {
        "model_p_win": p_win,
        "model_fair_odds": fair_odds,
        "edge_pct": edge_pct,
        "ev_per_unit": ev_per_unit,
        "p_over_calib": p_over_calib,
        "p_under_calib": p_under_calib,
    }


def print_market_table(market_df: pd.DataFrame):
    """
    Pretty-print a table of market lines with model edge.
    Assumes columns: book,line,side,odds,model_p_win,model_fair_odds,edge_pct,ev_per_unit.
    """
    if market_df.empty:
        print("\nNo market rows to show for this player.")
        return

    cols_present = {c for c in ["book", "line", "side", "odds",
                                "model_p_win", "model_fair_odds",
                                "edge_pct", "ev_per_unit"] if c in market_df.columns}

    # Sort by line then by book for readability
    market_df = market_df.sort_values(["line", "side", "book"], ascending=[True, True, True])

    print("\n=== Market comparison for this player (points props) ===")
    header = f"{'Book':>12} | {'Line':>6} | {'Side':>5} | {'Odds':>6} | {'Model p_win':>11} | {'Fair odds':>9} | {'Edge%':>7} | {'EV/unit':>8}"
    print(header)
    print("-" * len(header))
    for _, r in market_df.iterrows():
        book = str(r.get("book", ""))
        line = float(r["line"])
        side = str(r["side"])
        odds = int(r["odds"])
        p_win = float(r["model_p_win"])
        fair = r["model_fair_odds"]
        if fair is None or (isinstance(fair, float) and math.isnan(fair)):
            fair_str = "n/a"
        else:
            fair_str = f"{int(fair):+d}"
        edge_pct = float(r["edge_pct"])
        ev_unit = float(r["ev_per_unit"])

        print(
            f"{book:>12} | {line:6.1f} | {side:>5} | {odds:6d} | "
            f"{p_win:11.3f} | {fair_str:>9} | {edge_pct:7.2f} | {ev_unit:8.3f}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare a star's points ladder vs real market odds for their NEXT game."
    )
    parser.add_argument(
        "--player",
        required=True,
        help="Player name (e.g. 'Devin Booker')",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season start year to consider for form (default 2023).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="10,15,20,25,30,35",
        help="Comma-separated list of points thresholds, e.g. '15,20,25,30,35'.",
    )
    parser.add_argument(
        "--target-prob",
        type=float,
        default=0.60,
        help="Target calibrated probability for picking the 'aggressive' rung (default 0.60).",
    )
    parser.add_argument(
        "--game-date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD. If omitted, script will try to auto-detect next game.",
    )
    parser.add_argument(
        "--opp",
        type=str,
        default=None,
        help="Optional opponent team abbrev (e.g., DEN). If provided with --home-away, "
             "we'll skip schedule lookup.",
    )
    parser.add_argument(
        "--home-away",
        type=str,
        choices=["home", "away"],
        default=None,
        help="Use with --opp to specify whether the player is home or away.",
    )
    parser.add_argument(
        "--odds-file",
        type=str,
        default="data/odds_slate.csv",
        help="Path to CSV of odds (e.g. from fetch_props_from_the_odds_api.py).",
    )
    parser.add_argument(
        "--books",
        type=str,
        default=None,
        help="Optional comma-separated list of books to keep, e.g. 'Bet365,FanDuel'.",
    )

    args = parser.parse_args()

    # Parse thresholds
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    thresholds = sorted(thresholds)

    # Parse books filter
    books_filter = None
    if args.books:
        books_filter = [b.strip() for b in args.books.split(",") if b.strip()]

    # ------------------------------------------------------------------
    # Load logs & resolve player
    # ------------------------------------------------------------------
    if not LOGS_CSV.exists():
        raise FileNotFoundError(f"Logs file not found: {LOGS_CSV}")
    logs_df = pd.read_csv(LOGS_CSV)
    print(f"Loading raw player logs from {LOGS_CSV} ...")

    player_id, player_name, team_abbrev, season, latest_game_date = resolve_player_from_logs(
        logs_df, args.player, args.season_min
    )

    print("\nResolved player:")
    print(f"  Name:         {player_name} (id={player_id})")
    print(f"  Team:         {team_abbrev}")
    print(f"  Latest game:  {latest_game_date} (season {season})")

    # ------------------------------------------------------------------
    # Decide which game we're predicting for
    # ------------------------------------------------------------------
    if args.opp and args.home_away:
        # Manual override: user tells us opponent + home/away.
        opp_abbrev = args.opp.upper()
        home_away = args.home_away.lower()

        if args.game_date:
            target_date = dt.date.fromisoformat(args.game_date)
        else:
            # If no date given, assume 2 days after last game as a rough default.
            target_date = latest_game_date + dt.timedelta(days=2)

        print("\nUsing MANUAL matchup info:")
        print(f"  Date:      {target_date}")
        print(f"  Opp:       {opp_abbrev}")
        print(f"  Home/Away: {home_away.upper()}")
    else:
        # We will use ScoreboardV2 to find the game
        if args.game_date:
            # Use that specific date
            target_date = dt.date.fromisoformat(args.game_date)
            print(f"\nLooking up scheduled game for {team_abbrev} on {target_date} via ScoreboardV2...")
            info = find_next_game_for_team(
                team_abbrev, start_date=target_date, max_days_ahead=0
            )
            if info is None:
                print(
                    f"Could not automatically find a scheduled game for team {team_abbrev} "
                    f"on {target_date} using nba_api."
                )
                print("You can instead pass --opp and --home-away manually, e.g.:")
                print("  --opp DEN --home-away home")
                raise SystemExit(1)
        else:
            # Auto: scan forward from max(today, latest_game_date+1)
            today = dt.date.today()
            start_date = max(today, latest_game_date + dt.timedelta(days=1))
            print(f"\nAuto-detecting next game for {team_abbrev} starting from {start_date}...")
            info = find_next_game_for_team(
                team_abbrev, start_date=start_date, max_days_ahead=60
            )
            if info is None:
                print(
                    f"Could not find a next scheduled game for team {team_abbrev} within "
                    f"60 days of {start_date} using ScoreboardV2."
                )
                print("You can pass --game-date explicitly, or override with --opp / --home-away.")
                raise SystemExit(1)

        # If we got here, info is defined
        target_date = info["game_date"]
        opp_abbrev = info["opp_abbrev"]
        home_away = info["home_away"]

        print("\nUsing AUTO-detected matchup:")
        print(f"  Date:      {target_date}")
        print(f"  Opponent:  {opp_abbrev}")
        print(f"  Home/Away: {home_away.upper()}")

    # ------------------------------------------------------------------
    # Build feature row for this next game
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

    # ------------------------------------------------------------------
    # Load regression model & calibrator, then predict
    # ------------------------------------------------------------------
    print(f"\nLoading regression model from {MODEL_PATH} ...")
    model, sigma, model_feature_cols = load_regression_model(MODEL_PATH)
    calibrator = load_over_prob_calibrator(CALIBRATOR_PATH)

    used_cols = model_feature_cols if model_feature_cols else FEATURE_COLS
    X = feat_row[used_cols].to_numpy()
    mu = float(model.predict(X)[0])

    print(f"\n=== Star ladder model view ===")
    print(f"Player:     {player_name}")
    print(f"Team:       {team_abbrev}")
    print(f"Opponent:   {opp_abbrev}  ({home_away.upper()})")
    print(f"Game date:  {target_date}")
    print(f"\nRegression expected points (mu): {mu:.2f}")
    print(f"Sigma used for normal approximation: {sigma:.3f}")

    # Compute ladder
    ladder_rows = compute_ladder(mu, sigma, thresholds, calibrator)
    print_ladder_table(ladder_rows)
    highlight_recommended_rung(ladder_rows, args.target_prob)

    # ------------------------------------------------------------------
    # Market comparison: load odds file and score this player's props
    # ------------------------------------------------------------------
    odds_path = Path(args.odds_file)
    market_df = load_market_rows_for_player(odds_path, player_name, books_filter=books_filter)

    if not market_df.empty:
        scored_rows = []
        for _, r in market_df.iterrows():
            scored = score_market_row(r, mu, sigma, calibrator)
            merged = {**r.to_dict(), **scored}
            scored_rows.append(merged)
        market_scored_df = pd.DataFrame(scored_rows)
    else:
        market_scored_df = market_df

    print_market_table(market_scored_df)


if __name__ == "__main__":
    main()