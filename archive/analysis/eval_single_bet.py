#!/usr/bin/env python
"""
eval_single_bet.py

Given:
  - a player (e.g. "Devin Booker")
  - a points line (e.g. 23.5)
  - a side ("over" or "under")
  - book odds in American format (e.g. -115, +120)

This script will:
  - Auto-detect the player's NEXT game (unless overridden)
  - Build rolling features (last 5/15 games, opp defence, rest, home/away)
  - Use your regression model (points_regression.pkl) to get expected points mu
  - Use the normal approximation + isotonic calibrator to get P(OVER)
  - Convert that into:
      * Fair odds
      * Edge vs book implied probability
      * EV per 1 unit

Example:

  python eval_single_bet.py \
    --player "Devin Booker" \
    --line 23.5 \
    --side over \
    --odds -115 \
    --season-min 2023

Optional overrides:

  - Force date:
      --game-date 2025-11-28
  - Force opponent + home/away:
      --opp OKC --home-away away
"""

import argparse
import datetime as dt
import math
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# We reuse helpers from your existing script
from predict_points_over_line_next_game import (
    LOGS_CSV,
    MODEL_PATH,
    FEATURE_COLS,
    load_regression_model,
    build_next_game_feature_row,
    resolve_player_from_logs,
    find_next_game_for_team,
)

# Path to the calibrator we just trained
CALIBRATOR_PATH = Path("models/over_prob_calibrator.pkl")


# ----------------------------------------------------------------------
# Probability / odds helpers
# ----------------------------------------------------------------------

def normal_over_probs(mu: float, sigma: float, line: float) -> Tuple[float, float]:
    """
    Approximate P(OVER line) and P(UNDER line) using N(mu, sigma^2).
    Same formula you used in your other scripts.
    """
    if sigma <= 0:
        return (1.0 if mu > line else 0.0, 1.0 if mu <= line else 0.0)

    z = (line - mu) / sigma
    p_under = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_over = 1.0 - p_under
    return p_over, p_under


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
    """
    Convert American odds to implied probability (ignoring vig).
      -115 -> ~0.535
      +120 -> ~0.455
    """
    odds = float(odds)
    if odds < 0:
        return (-odds) / (-odds + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def ev_per_unit(p: float, odds: float) -> float:
    """
    Expected value per 1 unit staked given:
      p    = model probability of winning
      odds = American odds (e.g., -115, +120)

    For odds < 0: risk 1 to win 100/|odds|
    For odds > 0: risk 1 to win odds/100
    """
    odds = float(odds)
    if odds < 0:
        payout = 100.0 / -odds
    else:
        payout = odds / 100.0

    # EV = p * payout - (1 - p) * 1
    return p * payout - (1.0 - p) * 1.0


# ----------------------------------------------------------------------
# Calibrator loading
# ----------------------------------------------------------------------

def load_over_calibrator(path: Path = CALIBRATOR_PATH):
    """
    Load the isotonic regression calibrator from disk.

    In build_over_prob_calibrator.py we likely pickled either:
      - the calibrator directly, or
      - a dict like {"calibrator": iso}

    This handles both cases.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration model not found at {path}. "
            f"Run build_over_prob_calibrator.py first."
        )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "calibrator" in obj:
        return obj["calibrator"]
    return obj


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single player points bet (OVER/UNDER) vs book odds."
    )
    parser.add_argument("--player", required=True, help="Player name (e.g. 'Devin Booker')")
    parser.add_argument("--line", required=True, type=float, help="Points line (e.g. 23.5)")
    parser.add_argument("--side", required=True, choices=["over", "under"], help="Bet side")
    parser.add_argument("--odds", required=True, type=float, help="Book odds (American, e.g. -115, +120)")
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
        help="Optional opponent team abbrev (e.g., DEN). Used with --home-away to override schedule.",
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
        # Use ScoreboardV2-based helper to find next game
        if args.game_date:
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

    used_cols = model_feature_cols if model_feature_cols else FEATURE_COLS
    X = feat_row[used_cols].to_numpy()

    mu = float(model.predict(X)[0])
    p_over_raw, p_under_raw = normal_over_probs(mu, sigma, args.line)

    calibrator = load_over_calibrator(CALIBRATOR_PATH)
    # Calibrator expects 1D array
    p_over_calib = float(calibrator.predict(np.array([p_over_raw]))[0])
    # Safety clipping
    p_over_calib = max(0.0, min(1.0, p_over_calib))
    p_under_calib = 1.0 - p_over_calib

    # ------------------------------------------------------------------
    # Book odds, edge, EV
    # ------------------------------------------------------------------
    side = args.side.lower()
    odds = float(args.odds)

    if side == "over":
        p_model = p_over_calib
    else:
        p_model = p_under_calib

    p_book = american_to_prob(odds)
    edge_pct = (p_model - p_book) * 100.0
    ev = ev_per_unit(p_model, odds)

    fair_odds_for_side = prob_to_american(p_model)

    # ------------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------------
    print(f"\n=== Single-bet evaluation for {side.upper()} {args.line:.1f} points ===")
    print(f"Player:        {player_name}")
    print(f"Team:          {team_abbrev}")
    print(f"Opponent:      {opp_abbrev}  ({home_away.upper()})")
    print(f"Game date:     {target_date}")

    print(f"\nRegression expected points (mu): {mu:.2f}")
    print(f"Raw   P(OVER {args.line:.1f}):  {p_over_raw:.3f}")
    print(f"Calib P(OVER {args.line:.1f}):  {p_over_calib:.3f}")
    print(f"Calib P(UNDER {args.line:.1f}): {p_under_calib:.3f}")

    print(f"\nBook odds on {side.upper()}: {odds:+.0f}")
    print(f"Implied prob from odds:       {p_book:.3f}")

    if fair_odds_for_side is None:
        print(f"Fair odds for {side.upper()}: (degenerate, p={p_model:.3f})")
    else:
        print(f"Model fair odds for {side.upper()}: {fair_odds_for_side:+d}")

    print(f"\nModel edge on this side: {edge_pct:+.2f} percentage points")
    print(f"EV per 1 unit staked:   {ev:+.3f} units")

    # Simple verbal verdict
    if edge_pct < 0:
        verdict = "NEGATIVE edge (model thinks this is -EV). Likely a pass."
    elif edge_pct < 2:
        verdict = "Tiny positive edge, probably not enough alone. Lean at best."
    elif edge_pct < 5:
        verdict = "Small positive edge. Interesting, but still needs discipline + bankroll rules."
    else:
        verdict = "Sizeable edge on this side (on paper). Prime candidate to track / bet small."

    print(f"\nVerdict: {verdict}")
    print("\n(As always: judge this by backtesting on real logged lines, not just one-off outputs.)")


if __name__ == "__main__":
    main()
