#!/usr/bin/env python
"""
predict_points_over_line_next_game.py

Given a player name and a points line, estimate P(OVER) for the player's
NEXT SCHEDULED GAME using:

- Rolling form features (last 5 / 15 games)
- Opponent defensive form (points allowed last 5 / 15 games)
- Days since last game
- Home / away flag
- PLUS richer env/baseline/trend/volatility features
- PLUS a minutes prediction from the minutes model (min_pred)

It uses:
  - data/player_game_logs.csv         (raw per-game box score logs)
  - models/points_regression.pkl      (regression model bundle)
  - models/minutes_regression.pkl     (via minutes_utils.add_minutes_predictions)
  - models/over_prob_calibrator.pkl   (optional isotonic calibrator)
"""

import argparse
import datetime as dt
import math
import pickle
import time
import unicodedata
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams

# NEW: bring in minutes model helper
from minutes_utils import add_minutes_predictions

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

LOGS_CSV = Path("data/player_game_logs.csv")
MODEL_PATH = Path("models/points_regression.pkl")
CALIBRATOR_PATH = Path("models/over_prob_calibrator.pkl")

# Fallback; normally we use feature_cols from the model bundle
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

    # richer features (match training)
    "opp_dvp_pos_pts_roll5",
    "opp_dvp_pos_pts_roll15",
    "team_pace_roll5",
    "team_pace_roll15",
    "player_pts_career_mean",
    "player_pts_season_mean",
    "player_minutes_career_mean",
    "player_minutes_season_mean",
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
    "min_pred",
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
    """Convert probability p to American odds. Returns None for p<=0 or p>=1."""
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        # favorite (negative odds)
        return int(round(-100 * p / (1 - p)))
    else:
        # underdog (positive odds)
        return int(round(100 * (1 - p) / p))


def normal_over_probs(mu: float, sigma: float, line: float) -> Tuple[float, float]:
    """
    Approximate P(OVER line) and P(UNDER line) using N(mu, sigma^2) for scalars.
    """
    if sigma <= 0:
        return (1.0 if mu > line else 0.0, 1.0 if mu <= line else 0.0)

    z = (line - mu) / sigma
    p_under = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_over = 1.0 - p_under
    return p_over, p_under


def _bucket_star_tier_pts(pts: float) -> int:
    """
    Same buckets as build_player_points_features.py:

    0 = low-usage / bench
    1 = rotation scorer
    2 = primary / secondary option
    3 = star / elite scorer
    """
    if pd.isna(pts):
        return 0
    if pts < 8:
        return 0
    elif pts < 15:
        return 1
    elif pts < 22:
        return 2
    else:
        return 3


def _bucket_star_tier_minutes(m: float) -> int:
    """
    Same buckets as build_player_points_features.py:

    0 = deep bench
    1 = rotation (15–24 min)
    2 = strong starter (24–30)
    3 = heavy-minutes (30+)
    """
    if pd.isna(m):
        return 0
    if m < 15:
        return 0
    elif m < 24:
        return 1
    elif m < 30:
        return 2
    else:
        return 3


# ----------------------------------------------------------------------
# Model + calibrator loading
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


def load_calibrator(path: Path = CALIBRATOR_PATH):
    """
    Load isotonic calibrator bundle {calibrator, info} if available.
    Returns the calibrator object or None.
    """
    if not path.exists():
        print("(No calibration file found; using raw probabilities.)")
        return None

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict) and "calibrator" in bundle:
        return bundle["calibrator"]

    # Fallback: maybe the object itself was pickled directly
    return bundle


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
    print(f"Using team_id={team_id} for {team_abbrev}")

    for offset in range(max_days_ahead + 1):
        d = start_date + dt.timedelta(days=offset)
        date_str = d.strftime("%m/%d/%Y")
        print(f"  Checking scoreboard for {date_str} ...")

        try:
            sb = ScoreboardV2(game_date=date_str, league_id="00")
            games_df = sb.game_header.get_data_frame()
        except Exception as e:
            print(f"    [ScoreboardV2 error on {date_str}: {e}]")
            time.sleep(sleep_s)
            continue

        if games_df.empty:
            print("    No games returned for this date.")
            time.sleep(sleep_s)
            continue

        # Filter rows where our team is either home or away
        mask = (games_df["HOME_TEAM_ID"] == team_id) | (games_df["VISITOR_TEAM_ID"] == team_id)
        cand = games_df.loc[mask]

        if cand.empty:
            print("    Our team is not on the schedule for this date.")
            time.sleep(sleep_s)
            continue

        # Take the first matching game as the "next" one
        row = cand.iloc[0]
        is_home = row["HOME_TEAM_ID"] == team_id
        opp_team_id = int(row["VISITOR_TEAM_ID"] if is_home else row["HOME_TEAM_ID"])

        # Map opponent ID -> abbrev
        opp_info = teams.find_team_name_by_id(opp_team_id)
        opp_abbrev = opp_info["abbreviation"] if opp_info else "UNK"

        print(
            f"    Found game: GAME_ID={row['GAME_ID']}, "
            f"{team_abbrev} vs {opp_abbrev} on {d} "
            f"({'HOME' if is_home else 'AWAY'})"
        )

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
    - Approximates richer env/baseline/trend/volatility features
    - Then calls minutes_utils.add_minutes_predictions(...) to add 'min_pred'
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

    # Team-level defensive form for opponent: points allowed
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
        opp_pts_allowed_roll5 = 0.0
        opp_pts_allowed_roll15 = 0.0
    else:
        opp_pts_allowed_roll5 = rolling_mean_last_n(opp_games["opp_score"], 5)
        opp_pts_allowed_roll15 = rolling_mean_last_n(opp_games["opp_score"], 15)

    # Approximate DvP-vs-position with overall points allowed
    opp_dvp_pos_pts_roll5 = opp_pts_allowed_roll5
    opp_dvp_pos_pts_roll15 = opp_pts_allowed_roll15

    is_home_flag = 1 if home_away.lower() == "home" else 0

    # --- Player baselines (career + season) ---------------------------
    # Career = all seasons up to this point
    all_player_games = df[df["player_id"] == player_id].copy()
    all_player_games = all_player_games[all_player_games["game_date"] < pd.Timestamp(target_date)]

    if all_player_games.empty:
        player_pts_career_mean = 0.0
        player_minutes_career_mean = 0.0
    else:
        player_pts_career_mean = float(all_player_games["pts"].mean())
        player_minutes_career_mean = float(all_player_games["minutes"].mean())

    player_pts_season_mean = float(player_games["pts"].mean())
    player_minutes_season_mean = float(player_games["minutes"].mean())

    rel_minutes_vs_career = minutes_roll5 - player_minutes_career_mean
    rel_pts_vs_career = pts_roll5 - player_pts_career_mean

    star_tier_pts = _bucket_star_tier_pts(player_pts_career_mean)
    star_tier_minutes = _bucket_star_tier_minutes(player_minutes_career_mean)

    # --- Trend features ----------------------------------------------
    pts_trend_5_15 = pts_roll5 - pts_roll15
    minutes_trend_5_15 = minutes_roll5 - minutes_roll15
    fga_trend_5_15 = fga_roll5 - fga_roll15

    # --- Volatility (std over last 5 games) --------------------------
    def _last5_std(series: pd.Series) -> float:
        series = series.tail(5)
        if len(series) < 3:
            return 0.0
        return float(series.std(ddof=1))

    pts_std5 = _last5_std(player_games["pts"])
    minutes_std5 = _last5_std(player_games["minutes"])
    fga_std5 = _last5_std(player_games["fga"])

    # --- Usage ratios (roll5) ----------------------------------------
    eps = 1e-3
    pts_per_min_roll5 = pts_roll5 / (minutes_roll5 + eps) if minutes_roll5 > 0 else 0.0
    fga_per_min_roll5 = fga_roll5 / (minutes_roll5 + eps) if minutes_roll5 > 0 else 0.0
    fta_per_min_roll5 = fta_roll5 / (minutes_roll5 + eps) if minutes_roll5 > 0 else 0.0

    # --- Rest flags ---------------------------------------------------
    is_b2b = 1 if days_since_last_game <= 1 else 0
    is_long_rest = 1 if days_since_last_game >= 3 else 0

    # --- Team pace features ------------------------------------------
    # Estimate possessions: poss ≈ FGA + 0.44*FTA - OREB + TOV
    if all(c in df.columns for c in ["fga", "fta", "oreb", "tov"]):
        team_pace_df = (
            df.groupby(
                ["season", "game_id", "game_date", "team_abbrev", "opp_abbrev"],
                as_index=False,
            )
            .agg(
                fga=("fga", "sum"),
                fta=("fta", "sum"),
                oreb=("oreb", "sum"),
                tov=("tov", "sum"),
            )
        )
        team_pace_df["game_date"] = pd.to_datetime(team_pace_df["game_date"])
        team_pace_df["team_possessions_est"] = (
            team_pace_df["fga"]
            + 0.44 * team_pace_df["fta"]
            - team_pace_df["oreb"]
            + team_pace_df["tov"]
        )

        team_games_this_team = team_pace_df[
            (team_pace_df["season"] == season) &
            (team_pace_df["team_abbrev"] == team_abbrev) &
            (team_pace_df["game_date"] < pd.Timestamp(target_date))
        ].sort_values("game_date")

        if team_games_this_team.empty:
            team_pace_roll5 = 100.0
            team_pace_roll15 = 100.0
        else:
            team_pace_roll5 = rolling_mean_last_n(
                team_games_this_team["team_possessions_est"], 5
            )
            team_pace_roll15 = rolling_mean_last_n(
                team_games_this_team["team_possessions_est"], 15
            )
    else:
        # Fallback to a league-average-ish pace
        team_pace_roll5 = 100.0
        team_pace_roll15 = 100.0

    # ------------------------------------------------------------------
    # Assemble row
    # ------------------------------------------------------------------
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
        "is_home": is_home_flag,

        # richer env/matchup features
        "opp_dvp_pos_pts_roll5": opp_dvp_pos_pts_roll5,
        "opp_dvp_pos_pts_roll15": opp_dvp_pos_pts_roll15,
        "team_pace_roll5": team_pace_roll5,
        "team_pace_roll15": team_pace_roll15,

        # player baselines
        "player_pts_career_mean": player_pts_career_mean,
        "player_pts_season_mean": player_pts_season_mean,
        "player_minutes_career_mean": player_minutes_career_mean,
        "player_minutes_season_mean": player_minutes_season_mean,

        # role vs career & star tiers
        "rel_minutes_vs_career": rel_minutes_vs_career,
        "rel_pts_vs_career": rel_pts_vs_career,
        "star_tier_pts": star_tier_pts,
        "star_tier_minutes": star_tier_minutes,

        # trends
        "pts_trend_5_15": pts_trend_5_15,
        "minutes_trend_5_15": minutes_trend_5_15,
        "fga_trend_5_15": fga_trend_5_15,

        # volatility
        "pts_std5": pts_std5,
        "minutes_std5": minutes_std5,
        "fga_std5": fga_std5,

        # usage ratios
        "pts_per_min_roll5": pts_per_min_roll5,
        "fga_per_min_roll5": fga_per_min_roll5,
        "fta_per_min_roll5": fta_per_min_roll5,

        # rest flags
        "is_b2b": is_b2b,
        "is_long_rest": is_long_rest,
    }

    feat_df = pd.DataFrame([row]).fillna(0.0)

    # ------------------------------------------------------------------
    # NEW: attach minutes model prediction as min_pred
    # ------------------------------------------------------------------
    feat_df = add_minutes_predictions(feat_df)

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
        raise SystemExit(1)

    # Filter to seasons >= season_min if possible
    cand2 = candidates[candidates["season"] >= season_min]
    if cand2.empty:
        cand2 = candidates

    # Avoid SettingWithCopyWarning
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
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict P(OVER line) for a player's NEXT game using rolling features."
    )
    parser.add_argument("--player", required=True, help="Player name (e.g. 'Luka Doncic')")
    parser.add_argument("--line", required=True, type=float, help="Points line (e.g. 29.5)")
    parser.add_argument(
        "--season-min",
        type=int,
        default=2023,
        help="Minimum season start year to consider for form (default 2023).",
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

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load raw logs & resolve player
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
    # Load regression model & predict
    # ------------------------------------------------------------------
    print(f"\nLoading regression model from {MODEL_PATH} ...")
    model, sigma, model_feature_cols = load_regression_model(MODEL_PATH)

    # Ensure we use the exact feature ordering the model expects
    used_cols = model_feature_cols if model_feature_cols else FEATURE_COLS
    X = feat_row[used_cols].to_numpy()

    mu = float(model.predict(X)[0])

    # Raw probabilities from normal approximation
    p_over_raw, p_under_raw = normal_over_probs(mu, sigma, args.line)
    p_over_raw = float(max(0.0, min(1.0, p_over_raw)))
    p_under_raw = 1.0 - p_over_raw

    # Try to load and apply calibrator
    calibrator = load_calibrator(CALIBRATOR_PATH)
    used_calibrated = False
    if calibrator is not None:
        try:
            p_over_cal = float(calibrator.predict([p_over_raw])[0])
            p_over = max(0.0, min(1.0, p_over_cal))
            p_under = 1.0 - p_over
            used_calibrated = True
        except Exception as e:
            print(f"(Warning: failed to apply calibrator: {e}; using raw probabilities.)")
            p_over = p_over_raw
            p_under = p_under_raw
    else:
        p_over = p_over_raw
        p_under = p_under_raw

    odds_over = prob_to_american(p_over)
    odds_under = prob_to_american(p_under)

    # ------------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------------
    print(f"\n=== Regression-based prediction for points OVER {args.line:.1f} ===")
    print(f"Player:     {player_name}")
    print(f"Team:       {team_abbrev}")
    print(f"Opponent:   {opp_abbrev}  ({home_away.upper()})")
    print(f"Game date:  {target_date}")
    print(f"\nExpected points (mu): {mu:.2f}")

    # Show raw vs calibrated if calibrator was applied
    print(f"Raw   P(OVER {args.line:.1f}): {p_over_raw:.3f}")
    if used_calibrated:
        print(f"Calib P(OVER {args.line:.1f}): {p_over:.3f}")
    else:
        print(f"P(OVER {args.line:.1f}):       {p_over:.3f}")

    print(f"P(UNDER {args.line:.1f}): {p_under:.3f}")

    print("\nFair odds (no vig) using the displayed P(OVER):")
    if odds_over is None:
        print(f"  OVER  {args.line:.1f} -> (degenerate, p={p_over:.3f})")
    else:
        print(f"  OVER  {args.line:.1f} -> {odds_over:+d}")
    if odds_under is None:
        print(f"  UNDER {args.line:.1f} -> (degenerate, p={p_under:.3f})")
    else:
        print(f"  UNDER {args.line:.1f} -> {odds_under:+d}")

    print("\nNotes:")
    print("  - Raw P(OVER) comes from N(mu, sigma^2) around the regression prediction.")
    if used_calibrated:
        print("  - Calibrated P(OVER) has been adjusted using isotonic regression "
              "fit on historical data (synthetic lines from pts_roll5).")
    else:
        print("  - No calibration file found; using raw probabilities.")
    print("  - The feature row now includes a minutes prediction (min_pred) from the "
          "minutes_regression model, plus richer env/baseline/trend features.")


if __name__ == "__main__":
    main()