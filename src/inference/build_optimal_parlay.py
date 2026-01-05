#!/usr/bin/env python
"""
build_optimal_parlay.py

Builds optimal 4-leg parlays targeting a specific payout multiplier (default: 4x)
while maximizing the probability of success using model predictions.

Strategy:
1. Load all available player props and model predictions
2. Calculate edge (model_prob - market_implied_prob) for each prop
3. Find 4-leg combinations that:
   - Have cumulative market odds ~= target payout (4x)
   - Maximize cumulative model probability
   - Ensure positive expected value

NEW:
- Can build parlays **per sportsbook** using `data/odds_slate.csv` so legs are not mixed
  between sportsbooks (DraftKings-only parlay, FanDuel-only parlay, etc.)

Usage:
    python build_optimal_parlay.py
    python build_optimal_parlay.py --target-payout 4.0 --min-leg-prob 0.55
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimal 4-leg parlays")
    parser.add_argument("--edges-csv", type=str, default="data/edges_with_market.csv",
                        help="Path to edges CSV from scan_slate_with_model.py")
    parser.add_argument("--odds-slate-csv", type=str, default="data/odds_slate.csv",
                        help="Path to raw odds slate CSV from fetch_props_from_the_odds_api.py "
                             "(default: data/odds_slate.csv)")
    parser.add_argument("--target-payout", type=float, default=4.0,
                        help="Target parlay payout multiplier (default: 4.0). "
                             "Used when --min-payout is not set.")
    parser.add_argument("--payout-tolerance", type=float, default=0.5,
                        help="Tolerance around target payout (default: ±0.5). "
                             "Used when --min-payout is not set.")
    parser.add_argument("--min-payout", type=float, default=None,
                        help="If set, find the highest win-probability parlay with payout >= this value "
                             "(e.g. 4.0 for at-least-4x).")
    parser.add_argument("--max-payout", type=float, default=None,
                        help="Optional cap on payout when using --min-payout (e.g. 6.0 to avoid 10x+ parlays).")
    parser.add_argument("--target-parlay-prob", type=float, default=0.0,
                        help="Target parlay win probability (default: 0.0 = no constraint).")
    parser.add_argument("--min-parlay-prob", type=float, default=None,
                        help="Minimum parlay win probability. Defaults to --target-parlay-prob.")
    parser.add_argument("--num-legs", type=int, default=4,
                        help="Number of legs in parlay (default: 4)")
    parser.add_argument("--min-leg-prob", type=float, default=0.52,
                        help="Minimum model probability per leg (default: 0.52)")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Minimum edge per leg (default: 0.0, i.e., any positive)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Show top N parlay combinations (default: 10)")
    parser.add_argument("--max-correlation", type=float, default=1.0,
                        help="Max players from same game (default: 1.0 = no limit)")
    parser.add_argument("--stars-only", action="store_true",
                        help="Only include star players in parlays")
    parser.add_argument("--ladder-only", action="store_true",
                        help="Only use standard ladder thresholds (5, 10, 15, 20, 25, 30)")
    parser.add_argument("--ladder-thresholds", type=str, default="5,10,15,20,25,30",
                        help="Comma-separated ladder thresholds (default: 5,10,15,20,25,30)")
    parser.add_argument("--per-book", action="store_true",
                        help="Build separate parlays per sportsbook using odds_slate.csv. "
                             "No mixing between sportsbooks.")
    parser.add_argument("--require-in-range", action="store_true",
                        help="If set, only return parlays whose payout falls within "
                             "[target-payout ± payout-tolerance]. If none exist, print none.")
    parser.add_argument("--books", type=str, default=None,
                        help="Optional comma-separated list of sportsbook titles/keys to include "
                             "(e.g., DraftKings,FanDuel,BetMGM). If omitted, uses all seen in odds_slate.csv.")
    return parser.parse_args()


# List of star players to prioritize
STAR_PLAYERS = [
    'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
    'Luka Dončić', 'Luka Doncic', 'Jayson Tatum', 'Jaylen Brown', 'Joel Embiid', 
    'Nikola Jokić', 'Nikola Jokic', 'Anthony Davis', 'Damian Lillard', 'Devin Booker', 
    'Trae Young', 'Ja Morant', 'Shai Gilgeous-Alexander', 'Donovan Mitchell', 
    'Anthony Edwards', 'Tyrese Maxey', 'Paolo Banchero', 'Cade Cunningham', 
    'Scottie Barnes', 'Jalen Brunson', "De'Aaron Fox", 'DeMar DeRozan', 
    'Zion Williamson', 'Paul George', 'Jimmy Butler', 'Kawhi Leonard', 
    'James Harden', 'Karl-Anthony Towns', 'Kyrie Irving', 'Domantas Sabonis', 
    'Bam Adebayo', 'Jaren Jackson Jr', 'Brandon Ingram', 'Chet Holmgren', 
    'Victor Wembanyama', 'RJ Barrett', 'Derrick White', 'Jalen Williams',
    'Franz Wagner', 'Evan Mobley', 'Alperen Sengun', 'Lauri Markkanen',
    'Desmond Bane', 'Tyler Herro', 'Mikal Bridges', 'Dejounte Murray',
]


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal >= 2.0:
        return (decimal - 1) * 100
    else:
        return -100 / (decimal - 1)


def implied_prob_from_american(american: float) -> float:
    """Get implied probability from American odds."""
    if american >= 100:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def fair_odds_from_prob(prob: float) -> float:
    """Convert probability to fair American odds."""
    if prob <= 0:
        return 10000
    if prob >= 1:
        return -10000
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def calculate_parlay_payout(decimal_odds: List[float]) -> float:
    """Calculate parlay payout multiplier from decimal odds."""
    payout = 1.0
    for odds in decimal_odds:
        payout *= odds
    return payout


def calculate_parlay_probability(probs: List[float]) -> float:
    """Calculate probability of all legs hitting."""
    prob = 1.0
    for p in probs:
        prob *= p
    return prob


def calculate_ev(win_prob: float, payout: float, stake: float = 1.0) -> float:
    """Calculate expected value."""
    return win_prob * (payout - stake) - (1 - win_prob) * stake


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _derive_game_date_from_commence_time(commence_time: str) -> str:
    # commence_time is ISO like 2025-12-28T20:40:00Z; edges use YYYY-MM-DD
    if not commence_time:
        return ""
    return str(commence_time)[:10]


def prepare_book_edges_df(
    edges_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    book_title: str,
) -> pd.DataFrame:
    """
    Build a per-book legs table by joining model probs (from edges_df) with
    sportsbook-specific odds (from odds_df).
    Returns rows at the same granularity as edges_df lines, with per-book odds.
    """
    od = odds_df.copy()
    od = od[od["book_title"].astype(str) == str(book_title)].copy()
    if od.empty:
        return pd.DataFrame()

    # Normalize types / names
    od["game_date"] = od["commence_time"].astype(str).apply(_derive_game_date_from_commence_time)
    od = od.rename(columns={"player": "player_name", "line": "prop_pts_line"})

    # Pivot to get over/under odds per player-line for this book
    od["side"] = od["side"].astype(str).str.lower().str.strip()
    od["odds"] = pd.to_numeric(od["odds"], errors="coerce")
    od["prop_pts_line"] = pd.to_numeric(od["prop_pts_line"], errors="coerce")

    # Allow all markets (points/rebounds/assists/threes) so parlays can mix attributes.
    od = od[od["market_key"].notna()].copy()
    od = od.dropna(subset=["player_name", "prop_pts_line", "odds", "side", "home_team", "away_team", "game_date"])

    # Multiple rows can exist; keep the best price for each side at the book (max american odds is best)
    pivot = (
        od.groupby(["game_date", "home_team", "away_team", "market_key", "player_name", "prop_pts_line", "side"])["odds"]
        .max()
        .unstack("side")
        .reset_index()
    )
    pivot = pivot.rename(columns={"over": "over_odds", "under": "under_odds"})

    # Join with model predictions
    join_cols = ["game_date", "home_team", "away_team", "market_key", "player_name", "prop_pts_line"]
    merged = edges_df.merge(pivot, on=join_cols, how="inner")
    merged["book_title"] = str(book_title)
    return merged


def select_best_parlays_for_df(df_in: pd.DataFrame, args: argparse.Namespace) -> List[Dict]:
    """
    Runs the parlay search on a prepared dataframe that must contain:
      player_name, prop_pts_line, model_prob, market_odds, decimal_odds, market_implied_prob, edge
    """
    # Filter legs
    mask = (df_in["model_prob"] >= args.min_leg_prob) & (df_in["edge"] >= args.min_edge)

    # Filter for stars if requested
    if args.stars_only:
        mask = mask & df_in["player_name"].isin(STAR_PLAYERS)

    # Filter for ladder thresholds if requested
    if args.ladder_only:
        ladder_values = [float(x.strip()) for x in args.ladder_thresholds.split(",")]
        mask = mask & df_in["prop_pts_line"].apply(
            lambda x: any(abs(float(x) - (thresh + 0.5)) < 0.01 for thresh in ladder_values)
        )

    df_eligible = df_in[mask].copy()
    if len(df_eligible) < args.num_legs:
        return []

    # Sort by model probability (highest first), then by edge
    df_eligible = df_eligible.sort_values(["model_prob", "edge"], ascending=[False, False])

    # Limit candidates to keep combinations tractable
    max_candidates = 120
    if len(df_eligible) > max_candidates:
        df_eligible = df_eligible.head(max_candidates)

    candidates = df_eligible.to_dict("records")
    min_parlay_prob = float(args.min_parlay_prob) if args.min_parlay_prob is not None else float(args.target_parlay_prob)

    # Mode A: payout >= min_payout (optionally <= max_payout)
    if args.min_payout is not None:
        return _best_at_least_payout(
            df_in=df_eligible,
            k=int(args.num_legs),
            min_payout=float(args.min_payout),
            max_payout=float(args.max_payout) if args.max_payout is not None else None,
            min_parlay_prob=min_parlay_prob,
            top_n=int(args.top_n),
        )

    # Mode B: payout in [target ± tolerance]
    target_min = args.target_payout - args.payout_tolerance
    target_max = args.target_payout + args.payout_tolerance

    # Heuristic candidate narrowing around the payout target to keep search fast:
    # For 4 legs, average decimal per leg should be about target_payout^(1/4).
    # We keep legs near this range, but don't hard filter if it would leave too few.
    try:
        per_leg_target = float(args.target_payout) ** (1.0 / float(args.num_legs))
    except Exception:
        per_leg_target = 1.4
    dec = pd.to_numeric(df_eligible["decimal_odds"], errors="coerce").fillna(1.0)
    df_eligible["_dec_dist"] = (dec - per_leg_target).abs()
    df_eligible2 = df_eligible.sort_values(["_dec_dist", "model_prob"], ascending=[True, False]).drop(columns=["_dec_dist"])
    if len(df_eligible2) >= 60:
        candidates = df_eligible2.head(160).to_dict("records")
    else:
        candidates = df_eligible.to_dict("records")

    results = _best_in_range_parlay(candidates=candidates, k=int(args.num_legs), target_min=target_min, target_max=target_max, min_parlay_prob=min_parlay_prob)
    if results:
        return results

    if args.require_in_range:
        return []

    # Fallback: pick parlays closest to target payout, then by win probability
    fallback = _best_closest_parlays(candidates=candidates, k=int(args.num_legs), target=float(args.target_payout), top_n=int(args.top_n))
    if not fallback:
        return []
    return fallback


def _best_closest_parlays(candidates: List[Dict], k: int, target: float, top_n: int) -> List[Dict]:
    """
    Fallback: return a small set of parlays whose payout is closest to target,
    prioritized by closeness then by win probability.
    """
    # Use a smaller candidate pool for fallback to stay fast
    cand = sorted(candidates, key=lambda r: float(r.get("model_prob", 0.0)), reverse=True)[:80]
    out: List[Dict] = []
    for combo in combinations(range(len(cand)), k):
        legs = [cand[i] for i in combo]
        names = [leg["player_name"] for leg in legs]
        if len(set(names)) != len(names):
            continue
        payout = calculate_parlay_payout([float(leg["decimal_odds"]) for leg in legs])
        p_win = calculate_parlay_probability([float(leg["model_prob"]) for leg in legs])
        ev = calculate_ev(p_win, payout)
        out.append(
            {
                "legs": legs,
                "payout": payout,
                "parlay_prob": p_win,
                "ev": ev,
                "avg_edge": float(np.mean([float(leg.get("edge", 0.0)) for leg in legs])),
            }
        )
    out = sorted(out, key=lambda x: (abs(float(x["payout"]) - float(target)), -float(x["parlay_prob"])))
    return out[: max(1, top_n)]


def _best_in_range_parlay(
    candidates: List[Dict],
    k: int,
    target_min: float,
    target_max: float,
    min_parlay_prob: float,
) -> List[Dict]:
    """
    Find the single best (highest win probability) parlay within payout range,
    enforcing unique players. Uses pruning in decimal-odds order.
    Returns a list (either empty or [best]).
    """
    # Sort by decimal odds so we can compute tight-ish lower bounds.
    cand = [c for c in candidates if c.get("decimal_odds") is not None and c.get("model_prob") is not None]
    for c in cand:
        c["decimal_odds"] = float(c["decimal_odds"])
        c["model_prob"] = float(c["model_prob"])
        c["edge"] = float(c.get("edge", 0.0))
        c["market_odds"] = float(c.get("market_odds", -110.0))
    cand = [c for c in cand if c["decimal_odds"] > 1.0 and 0.0 < c["model_prob"] < 1.0]
    if len(cand) < k:
        return []

    cand.sort(key=lambda r: r["decimal_odds"])  # ascending
    decimals = [c["decimal_odds"] for c in cand]

    # Precompute min product for picking r legs starting at i (using the next r smallest).
    # If i+r exceeds, treat as inf.
    def min_prod(i: int, r: int) -> float:
        if i + r > len(decimals):
            return float("inf")
        p = 1.0
        for j in range(i, i + r):
            p *= decimals[j]
        return p

    # Precompute global max product for picking r legs from anywhere (use r largest)
    def max_prod_global(r: int) -> float:
        if r <= 0:
            return 1.0
        p = 1.0
        for d in decimals[-r:]:
            p *= d
        return p

    best: Optional[Dict] = None

    def dfs(start: int, picked: List[Dict], prod_odds: float, prod_prob: float, used_players: set) -> None:
        nonlocal best
        r = k - len(picked)
        if r == 0:
            if target_min <= prod_odds <= target_max and prod_prob >= min_parlay_prob:
                if best is None or prod_prob > float(best["parlay_prob"]):
                    ev = calculate_ev(prod_prob, prod_odds)
                    best = {
                        "legs": list(picked),
                        "payout": prod_odds,
                        "parlay_prob": prod_prob,
                        "ev": ev,
                        "avg_edge": float(np.mean([float(l.get("edge", 0.0)) for l in picked])) if picked else 0.0,
                    }
            return

        # Prune: even with smallest remaining odds, already too large for target_max
        if prod_odds * min_prod(start, r) > target_max:
            return
        # Prune: even with largest remaining odds, can't reach target_min (rare but safe)
        if prod_odds * max_prod_global(r) < target_min:
            return
        # Prune: even if remaining probs were 1.0, cannot beat current best
        if best is not None and prod_prob <= float(best["parlay_prob"]) and r > 0:
            # Can't prove, so don't prune aggressively.
            pass

        for i in range(start, len(cand) - r + 1):
            leg = cand[i]
            pn = leg.get("player_name")
            if pn in used_players:
                continue
            # Quick bound: if odds already too small, may never reach target_min; handled by max_prod_global.
            used_players.add(pn)
            picked.append(leg)
            dfs(i + 1, picked, prod_odds * leg["decimal_odds"], prod_prob * leg["model_prob"], used_players)
            picked.pop()
            used_players.remove(pn)

    dfs(0, [], 1.0, 1.0, set())
    return [best] if best is not None else []


def _best_at_least_payout(
    df_in: pd.DataFrame,
    k: int,
    min_payout: float,
    max_payout: Optional[float],
    min_parlay_prob: float,
    top_n: int,
) -> List[Dict]:
    """
    Return the best parlay(s) with payout >= min_payout (and <= max_payout if provided),
    prioritized by highest win probability. Tie-break: lower payout.
    """
    df = df_in.copy()
    if "decimal_odds" not in df.columns:
        return []

    df["decimal_odds"] = pd.to_numeric(df["decimal_odds"], errors="coerce")
    df = df.dropna(subset=["decimal_odds", "player_name", "model_prob"])
    df = df[df["decimal_odds"] > 1.0].copy()
    if len(df) < k:
        return []

    # To maximize probability, we want more-favorite legs, but must hit >= min_payout.
    # Keep a mix: top probs + some higher-decimal legs so reaching min is feasible.
    df = df.sort_values(["model_prob", "edge"], ascending=[False, False]).head(160)
    candidates = df.to_dict("records")

    best: List[Dict] = []
    # brute-force combinations with a cap (160 choose 4 is huge) -> narrow further:
    # Keep 80 most-probable plus 40 most-underdog (largest decimal) to allow reaching min_payout.
    df_hi_prob = df.sort_values("model_prob", ascending=False).head(80)
    df_hi_dec = df.sort_values("decimal_odds", ascending=False).head(40)
    df_pool = pd.concat([df_hi_prob, df_hi_dec], ignore_index=True).drop_duplicates(
        subset=["game_date", "home_team", "away_team", "market_key", "player_name", "prop_pts_line", "market_odds"],
        keep="first",
    )
    candidates = df_pool.to_dict("records")
    if len(candidates) < k:
        return []

    # Precompute global max product for reachability pruning
    decs = [float(c.get("decimal_odds", 1.0)) for c in candidates]
    decs_sorted = sorted(decs)

    def max_prod_global(r: int) -> float:
        if r <= 0:
            return 1.0
        p = 1.0
        for d in decs_sorted[-r:]:
            p *= d
        return p

    # Enumerate all combos of this smaller pool
    out: List[Dict] = []
    for combo in combinations(range(len(candidates)), k):
        legs = [candidates[i] for i in combo]
        names = [leg["player_name"] for leg in legs]
        if len(set(names)) != len(names):
            continue
        payout = calculate_parlay_payout([float(leg["decimal_odds"]) for leg in legs])
        if payout < min_payout:
            continue
        if max_payout is not None and payout > max_payout:
            continue
        p_win = calculate_parlay_probability([float(leg["model_prob"]) for leg in legs])
        if p_win < min_parlay_prob:
            continue
        ev = calculate_ev(p_win, payout)
        out.append(
            {
                "legs": legs,
                "payout": payout,
                "parlay_prob": p_win,
                "ev": ev,
                "avg_edge": float(np.mean([float(leg.get("edge", 0.0)) for leg in legs])),
            }
        )

    if not out:
        return []

    out = sorted(out, key=lambda x: (-float(x["parlay_prob"]), float(x["payout"])))
    return out[: max(1, top_n)]


def main():
    args = parse_args()
    if args.min_parlay_prob is None:
        args.min_parlay_prob = args.target_parlay_prob
    
    print("=" * 70)
    print("OPTIMAL PARLAY BUILDER")
    print("=" * 70)
    if args.min_payout is not None:
        if args.max_payout is not None:
            print(f"Payout constraint: >= {float(args.min_payout):.2f}x and <= {float(args.max_payout):.2f}x")
        else:
            print(f"Payout constraint: >= {float(args.min_payout):.2f}x")
    else:
        print(f"Target payout: {args.target_payout}x (±{args.payout_tolerance})")
    if float(args.min_parlay_prob) > 0:
        print(f"Min parlay win prob: {float(args.min_parlay_prob):.0%}")
    print(f"Number of legs: {args.num_legs}")
    print(f"Min probability per leg: {args.min_leg_prob:.0%}")
    print()

    edges_path = Path(args.edges_csv)
    if not edges_path.exists():
        print(f"[ERROR] Edges file not found: {edges_path}")
        print("[INFO] Run the full pipeline first: python run_full_slate_pipeline.py")
        return

    df_edges = pd.read_csv(edges_path)
    print(f"[INFO] Loaded {len(df_edges)} props from {edges_path}")

    # Normalize column names
    col_map = {
        "player": "player_name",
        "model_p_over": "p_over",
        "model_p_under": "p_under",
        "model_mean_pts": "mu",
    }
    df_edges = df_edges.rename(columns={k: v for k, v in col_map.items() if k in df_edges.columns})

    # Required columns
    required = ["player_name", "prop_pts_line", "p_over"]
    missing = [c for c in required if c not in df_edges.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        print(f"[INFO] Available columns: {list(df_edges.columns)}")
        return

    # For now, parlays are built on OVER legs (ladder thresholds map to OVER).
    # If you want UNDER legs too, we can extend this.
    df_edges["model_prob"] = df_edges["p_over"].clip(0.01, 0.99)

    def _print_best(label: str, best: Dict) -> None:
        target_min = args.target_payout - args.payout_tolerance
        target_max = args.target_payout + args.payout_tolerance
        in_range = target_min <= float(best["payout"]) <= target_max
        meets_p = float(best["parlay_prob"]) >= float(args.min_parlay_prob)
        print("\n" + "=" * 70)
        tag = "IN_RANGE" if in_range else "CLOSEST_AVAILABLE"
        if float(args.min_parlay_prob) > 0:
            tag2 = "MEETS_P" if meets_p else "LOW_P"
            print(f"TOP PARLAY ({label}) [{tag}] [{tag2}]")
        else:
            print(f"TOP PARLAY ({label}) [{tag}]")
        print("=" * 70)
        print(f"Win Probability: {best['parlay_prob']:.1%}")
        print(f"Payout: {best['payout']:.2f}x")
        print(f"EV: {best['ev']:+.2%}")
        print()
        print(f"{'Leg':<3} {'Player':<22} {'Line':>5} {'Odds':>7} {'Prob':>6} {'Edge':>6}")
        print("-" * 60)
        for i, leg in enumerate(best["legs"], 1):
            print(
                f"{i:<3} {str(leg['player_name'])[:21]:<22} "
                f"{float(leg['prop_pts_line']):>5.1f} {float(leg['market_odds']):>+7.0f} "
                f"{float(leg['model_prob']):>5.0%} {float(leg['edge']):>+5.1%}"
            )

    # If per-book, build and print one best parlay per sportsbook
    if args.per_book:
        odds_path = Path(args.odds_slate_csv)
        if not odds_path.exists():
            print(f"[ERROR] --per-book requested but odds slate not found: {odds_path}")
            print("[INFO] Generate it by running: python fetch_props_from_the_odds_api.py")
            return

        df_odds = pd.read_csv(odds_path, low_memory=False)
        # Ensure expected cols exist
        need = ["book_title", "commence_time", "home_team", "away_team", "market_key", "player", "line", "side", "odds"]
        miss2 = [c for c in need if c not in df_odds.columns]
        if miss2:
            print(f"[ERROR] odds_slate.csv missing columns: {miss2}")
            return

        # Build list of books to process
        books_all = sorted({str(x) for x in df_odds["book_title"].dropna().unique()})
        if args.books:
            wanted = {b.strip() for b in args.books.split(",") if b.strip()}
            books = [b for b in books_all if b in wanted]
        else:
            books = books_all

        print(f"[INFO] Building separate parlays for {len(books)} sportsbooks (no mixing).")

        # Pre-filter edges to required join columns
        join_cols = ["game_date", "home_team", "away_team", "market_key", "player_name", "prop_pts_line", "model_prob"]
        if not all(c in df_edges.columns for c in join_cols):
            print("[ERROR] edges_with_market.csv missing join columns needed for per-book join.")
            print(f"[INFO] Expected at least: {join_cols}")
            return

        out_lines: List[str] = []
        out_lines.append(f"OPTIMAL PARLAYS BY SPORTSBOOK (target {args.target_payout}x ±{args.payout_tolerance})\n")

        for book in books:
            merged = prepare_book_edges_df(df_edges, df_odds, book)
            if merged.empty:
                print("\n" + "=" * 70)
                print(f"{book}: No overlap between model edges and this book's offered lines.")
                print("=" * 70)
                out_lines.append(f"\n=== {book} ===\n")
                out_lines.append("No overlap between model edges and this book's offered lines.\n")
                continue

            # Use the book's OVER odds for payout / implied prob
            merged["market_odds"] = pd.to_numeric(merged.get("over_odds"), errors="coerce")
            merged = merged.dropna(subset=["market_odds"]).copy()
            if merged.empty:
                print("\n" + "=" * 70)
                print(f"{book}: No OVER odds available to price parlays for this book.")
                print("=" * 70)
                out_lines.append(f"\n=== {book} ===\n")
                out_lines.append("No OVER odds available to price parlays for this book.\n")
                continue

            merged["decimal_odds"] = merged["market_odds"].apply(lambda x: american_to_decimal(float(x)))
            merged["market_implied_prob"] = merged["market_odds"].apply(lambda x: implied_prob_from_american(float(x)))
            merged["edge"] = merged["model_prob"] - merged["market_implied_prob"]

            # Run selection
            results = select_best_parlays_for_df(merged, args)
            if not results:
                print("\n" + "=" * 70)
                print(f"{book}: No 4-leg parlay found under current filters.")
                print("=" * 70)
                out_lines.append(f"\n=== {book} ===\n")
                out_lines.append("No 4-leg parlay found under current filters; try lowering --min-leg-prob or --min-edge, or widen --payout-tolerance.\n")
                continue

            best = results[0]
            _print_best(f"{book}", best)

            out_lines.append(f"\n=== {book} ===\n")
            out_lines.append(f"WinProb={best['parlay_prob']:.3%}  Payout={best['payout']:.3f}x  EV={best['ev']:+.3%}\n")
            for leg in best["legs"]:
                out_lines.append(
                    f"- {leg['player_name']} OVER {float(leg['prop_pts_line']):.1f} @ {float(leg['market_odds']):+.0f} "
                    f"(p={float(leg['model_prob']):.3%}, edge={float(leg['edge']):+.3%})\n"
                )

        out_path = Path("data/run_logs/optimal_parlays_by_book.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(out_lines), encoding="utf-8")
        print(f"\n[INFO] Saved per-book parlays to {out_path}")
        return

    # Otherwise: legacy behavior using best-available odds from edges_with_market.csv
    df = df_edges.copy()
    if "over_odds_best" in df.columns:
        df["market_odds"] = df["over_odds_best"]
    elif "over_odds" in df.columns:
        df["market_odds"] = df["over_odds"]
    else:
        df["market_odds"] = -110

    df["decimal_odds"] = df["market_odds"].apply(lambda x: american_to_decimal(_safe_float(x, -110.0)))
    df["market_implied_prob"] = df["market_odds"].apply(lambda x: implied_prob_from_american(_safe_float(x, -110.0)))
    df["edge"] = df["model_prob"] - df["market_implied_prob"]
    results = select_best_parlays_for_df(df, args)
    if not results:
        print(f"\n[BEST_AVAILABLE] No qualifying parlays found for target {args.target_payout}x (±{args.payout_tolerance}).")
        return
    _print_best("BEST_AVAILABLE", results[0])

    # Filter legs
    mask = (df["model_prob"] >= args.min_leg_prob) & (df["edge"] >= args.min_edge)
    
    # Filter for stars if requested
    if args.stars_only:
        stars_mask = df["player_name"].isin(STAR_PLAYERS)
        mask = mask & stars_mask
        print(f"[INFO] Filtering for star players only")
    
    # Filter for ladder thresholds if requested
    if args.ladder_only:
        ladder_values = [float(x.strip()) for x in args.ladder_thresholds.split(",")]
        # Match lines that are exactly at ladder thresholds (e.g., 10.5 matches 10, 15.5 matches 15)
        ladder_mask = df["prop_pts_line"].apply(
            lambda x: any(abs(x - (thresh + 0.5)) < 0.01 for thresh in ladder_values)
        )
        mask = mask & ladder_mask
        print(f"[INFO] Filtering for ladder thresholds: {ladder_values}")
    
    df_eligible = df[mask].copy()
    
    print(f"[INFO] {len(df_eligible)} legs meet criteria (prob >= {args.min_leg_prob:.0%}, edge >= {args.min_edge:.0%})")
    
    if len(df_eligible) < args.num_legs:
        print(f"[WARN] Not enough eligible legs for {args.num_legs}-leg parlay")
        print("[INFO] Relaxing criteria...")
        df_eligible = df[df["model_prob"] >= 0.50].copy()
        print(f"[INFO] {len(df_eligible)} legs with prob >= 50%")

    if len(df_eligible) < args.num_legs:
        print("[ERROR] Still not enough legs. Check your edges file.")
        return

    # Sort by model probability (highest first)
    df_eligible = df_eligible.sort_values("model_prob", ascending=False)
    
    # Keep only the best line per player (highest model_prob)
    df_eligible = df_eligible.drop_duplicates(subset=["player_name"], keep="first")
    print(f"[INFO] {len(df_eligible)} unique players after deduplication")
    
    # Limit candidates to top 25 to make combinations tractable
    max_candidates = 25
    if len(df_eligible) > max_candidates:
        print(f"[INFO] Limiting to top {max_candidates} candidates by model probability")
        df_eligible = df_eligible.head(max_candidates)

    print()
    print("=" * 70)
    print("CANDIDATE LEGS (sorted by model probability)")
    print("=" * 70)
    print(f"{'Player':<25} {'Line':>6} {'Model%':>8} {'Market%':>8} {'Edge':>7} {'Odds':>8}")
    print("-" * 70)
    
    for _, row in df_eligible.head(15).iterrows():
        name = row["player_name"][:24]
        line = row["prop_pts_line"]
        model_p = row["model_prob"]
        market_p = row["market_implied_prob"]
        edge = row["edge"]
        odds = row["market_odds"]
        print(f"{name:<25} {line:>6.1f} {model_p:>7.1%} {market_p:>8.1%} {edge:>+6.1%} {odds:>+8.0f}")

    print()
    print("=" * 70)
    print(f"FINDING OPTIMAL {args.num_legs}-LEG PARLAYS")
    print("=" * 70)

    # Generate all combinations
    candidates = df_eligible.to_dict("records")
    all_combos = list(combinations(range(len(candidates)), args.num_legs))
    print(f"[INFO] Evaluating {len(all_combos):,} combinations...")

    # Evaluate each combination
    results = []
    target_min = args.target_payout - args.payout_tolerance
    target_max = args.target_payout + args.payout_tolerance

    for combo in all_combos:
        legs = [candidates[i] for i in combo]
        
        # IMPORTANT: Each player can only appear once per parlay
        player_names = [leg["player_name"] for leg in legs]
        if len(set(player_names)) != len(player_names):
            continue  # Duplicate player, skip
        
        # Check game correlation (optional: limit same-game legs)
        if "game_id" in legs[0]:
            game_ids = [leg.get("game_id", i) for i, leg in enumerate(legs)]
            if len(set(game_ids)) < len(game_ids) * args.max_correlation:
                continue  # Too correlated
        
        # Calculate parlay stats
        decimal_odds = [leg["decimal_odds"] for leg in legs]
        model_probs = [leg["model_prob"] for leg in legs]
        
        payout = calculate_parlay_payout(decimal_odds)
        parlay_prob = calculate_parlay_probability(model_probs)
        ev = calculate_ev(parlay_prob, payout)
        
        # Check if payout is in target range
        if target_min <= payout <= target_max:
            results.append({
                "legs": legs,
                "payout": payout,
                "parlay_prob": parlay_prob,
                "ev": ev,
                "avg_edge": np.mean([leg["edge"] for leg in legs]),
            })

    # Sort by parlay probability (highest first)
    results = sorted(results, key=lambda x: x["parlay_prob"], reverse=True)

    if not results:
        print(f"[WARN] No parlays found in payout range {target_min:.1f}x - {target_max:.1f}x")
        print("[INFO] Try adjusting --payout-tolerance or --min-leg-prob")
        
        # Show best available anyway
        print("\n[INFO] Showing best available parlays regardless of payout:")
        results_all = []
        for combo in all_combos[:1000]:  # Limit for speed
            legs = [candidates[i] for i in combo]
            decimal_odds = [leg["decimal_odds"] for leg in legs]
            model_probs = [leg["model_prob"] for leg in legs]
            payout = calculate_parlay_payout(decimal_odds)
            parlay_prob = calculate_parlay_probability(model_probs)
            ev = calculate_ev(parlay_prob, payout)
            results_all.append({
                "legs": legs,
                "payout": payout,
                "parlay_prob": parlay_prob,
                "ev": ev,
            })
        results = sorted(results_all, key=lambda x: x["parlay_prob"], reverse=True)[:5]

    print()
    print("=" * 70)
    print(f"TOP {min(args.top_n, len(results))} PARLAYS (by win probability)")
    print("=" * 70)

    for rank, result in enumerate(results[:args.top_n], 1):
        print()
        print(f"🎯 PARLAY #{rank}")
        print(f"   Win Probability: {result['parlay_prob']:.1%}")
        print(f"   Payout: {result['payout']:.2f}x  (${result['payout']:.2f} on $1 bet)")
        print(f"   Expected Value: {result['ev']:+.2%}")
        print()
        print(f"   {'Leg':<3} {'Player':<22} {'Line':>5} {'Over':>6} {'Prob':>6} {'Edge':>6}")
        print("   " + "-" * 55)
        
        for i, leg in enumerate(result["legs"], 1):
            name = leg["player_name"][:21]
            line = leg["prop_pts_line"]
            odds = leg["market_odds"]
            prob = leg["model_prob"]
            edge = leg["edge"]
            print(f"   {i:<3} {name:<22} {line:>5.1f} {odds:>+6.0f} {prob:>5.0%} {edge:>+5.1%}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        best = results[0]
        print(f"Best parlay win probability: {best['parlay_prob']:.1%}")
        print(f"Expected return on $1 bet: ${best['parlay_prob'] * best['payout']:.2f}")
        print()
        print("RECOMMENDED BET:")
        print("-" * 40)
        for leg in best["legs"]:
            name = leg["player_name"]
            line = leg["prop_pts_line"]
            print(f"  ✓ {name} OVER {line:.1f} pts")
        print()
        print(f"  💰 Combined payout: {best['payout']:.2f}x")
        print(f"  📊 Model confidence: {best['parlay_prob']:.1%}")

    # Save to file
    output_path = Path("data/run_logs/optimal_parlays.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"OPTIMAL {args.num_legs}-LEG PARLAYS - Target {args.target_payout}x\n")
        f.write("=" * 60 + "\n\n")
        
        for rank, result in enumerate(results[:args.top_n], 1):
            f.write(f"PARLAY #{rank}\n")
            f.write(f"Win Probability: {result['parlay_prob']:.1%}\n")
            f.write(f"Payout: {result['payout']:.2f}x\n")
            f.write(f"EV: {result['ev']:+.2%}\n\n")
            for leg in result["legs"]:
                f.write(f"  {leg['player_name']} OVER {leg['prop_pts_line']:.1f} @ {leg['market_odds']:+.0f}\n")
            f.write("\n")
    
    print(f"\n[INFO] Saved to {output_path}")


if __name__ == "__main__":
    main()

