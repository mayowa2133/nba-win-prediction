#!/usr/bin/env python
"""Score current NBA game markets from precomputed odds snapshots and free lineup/injury data."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.api.mapping import prob_to_american
from src.utils.artifact_metadata import stable_id
from src.utils.betting import american_to_prob
from src.utils.game_market_features import GAME_MARKET_FEATURE_COLUMNS, build_upcoming_game_market_frame
from src.utils.nba_teams import canonical_team_abbrev


LOGS_CSV = Path("data/player_game_logs.csv")
ODDS_SNAPSHOTS_CSV = Path("data/game_odds_snapshots.csv")
INJURIES_CSV = Path("data/injury_reports.csv")
LINEUPS_CSV = Path("data/lineup_projections.csv")
STARTERS_CSV = Path("data/starter_history.csv")
MODELS_DIR = Path("models")
OUTPUT_CSV = Path("data/game_market_recommendations.csv")


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_model_version(bundle: dict, market: str) -> str:
    metadata = bundle.get("metadata") or {}
    created_at = metadata.get("artifact_created_at")
    return f"{market}:{created_at}" if created_at else f"{market}:unknown"


def _load_bundle(models_dir: Path, market: str) -> dict:
    path = models_dir / f"{market}_model.pkl"
    if not path.exists():
        raise RuntimeError(f"Missing model bundle for {market}: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _latest_market_rows(snapshot_df: pd.DataFrame, *, sportsbook: Optional[str], target_date: Optional[str]) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()

    odds = snapshot_df.copy()
    odds["game_date"] = odds["game_date"].astype(str)
    if target_date:
        odds = odds[odds["game_date"] == target_date].copy()
    if odds.empty:
        return odds

    odds["captured_at_ts"] = pd.to_datetime(odds["captured_at"], utc=True, errors="coerce")
    odds["line_value"] = pd.to_numeric(odds["line_value"], errors="coerce").fillna(0.0).abs()
    odds["price"] = pd.to_numeric(odds["price"], errors="coerce")
    keys = ["fixture_id", "market", "side"]

    preferred = pd.DataFrame()
    if sportsbook:
        preferred = odds[odds["sportsbook"].astype(str).str.lower() == sportsbook.lower()].copy()
        preferred = preferred.sort_values("captured_at_ts").groupby(keys, as_index=False).tail(1)

    if preferred.empty:
        return odds.sort_values("captured_at_ts").groupby(keys, as_index=False).tail(1)

    preferred_keys = set(tuple(row[key] for key in keys) for row in preferred.to_dict(orient="records"))
    fallback = odds.sort_values("captured_at_ts").groupby(keys, as_index=False).tail(1)
    fallback = fallback[
        ~fallback.apply(lambda row: tuple(row[key] for key in keys) in preferred_keys, axis=1)
    ].copy()
    return pd.concat([preferred, fallback], ignore_index=True)


def _build_upcoming_games(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    records = []
    for fixture_id, frame in rows.groupby("fixture_id", dropna=False):
        moneyline = frame[frame["market"] == "game_moneyline"].copy()
        spread = frame[frame["market"] == "game_spread"].copy()
        total = frame[frame["market"] == "game_total"].copy()
        first = frame.iloc[0]
        home_ml = moneyline[moneyline["side"] == "home"]["price"]
        away_ml = moneyline[moneyline["side"] == "away"]["price"]
        spread_home = spread[spread["side"] == "home"]["line_value"]
        total_over = total[total["side"] == "over"]["line_value"]

        records.append(
            {
                "fixture_id": str(fixture_id),
                "game_id": str(first.get("game_id") or stable_id(first.get("game_date"), first.get("home_team"), first.get("away_team"), prefix="game")),
                "game_date": str(first["game_date"]),
                "home_team": str(first["home_team"]),
                "away_team": str(first["away_team"]),
                "home_team_abbrev": canonical_team_abbrev(first.get("home_team")) or str(first.get("home_team")),
                "away_team_abbrev": canonical_team_abbrev(first.get("away_team")) or str(first.get("away_team")),
                "market_snapshot_at": str(frame["captured_at_ts"].max().isoformat()),
                "market_total_line": float(total_over.iloc[-1]) if not total_over.empty else np.nan,
                "market_home_spread_line": float(spread_home.iloc[-1]) if not spread_home.empty else np.nan,
                "market_home_ml_implied": american_to_prob(float(home_ml.iloc[-1])) if not home_ml.empty else np.nan,
                "market_away_ml_implied": american_to_prob(float(away_ml.iloc[-1])) if not away_ml.empty else np.nan,
            }
        )
    return pd.DataFrame(records)


def _row_lookup(rows: pd.DataFrame) -> Dict[tuple[str, str, str], dict]:
    return {
        (str(row["fixture_id"]), str(row["market"]), str(row["side"])): row
        for row in rows.to_dict(orient="records")
    }


def _injury_context(feature_row: pd.Series) -> dict:
    parts = []
    for side, team in (("home", feature_row["home_team"]), ("away", feature_row["away_team"])):
        summary = str(feature_row.get(f"{side}_summary") or "").strip()
        if summary:
            parts.append(f"{team}: {summary}")
    return {"summary": " | ".join(parts)} if parts else {}


def _lineup_context(feature_row: pd.Series) -> dict:
    return {
        "home_projected_returning_starters": float(feature_row.get("home_projected_returning_starters") or 0.0),
        "away_projected_returning_starters": float(feature_row.get("away_projected_returning_starters") or 0.0),
        "projected_returning_starters": float(feature_row.get("home_projected_returning_starters") or 0.0)
        + float(feature_row.get("away_projected_returning_starters") or 0.0),
        "projected_replacements": float(feature_row.get("home_projected_replacements") or 0.0)
        + float(feature_row.get("away_projected_replacements") or 0.0),
    }


def _recommendation_row(
    *,
    feature_row: pd.Series,
    market: str,
    selection: str,
    sportsbook_line: float,
    sportsbook_odds: Optional[float],
    fair_line: float,
    selected_probability: float,
    edge: float,
    market_snapshot_at: str,
    model_version: str,
) -> dict:
    recommendation_id = stable_id(
        feature_row["game_id"],
        market,
        selection,
        feature_row["game_date"],
        sportsbook_line,
        prefix="rec",
    )
    injury_context = _injury_context(feature_row)
    lineup_context = _lineup_context(feature_row)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "recommendation_id": recommendation_id,
        "game_id": str(feature_row["game_id"]),
        "player": "",
        "game_date": str(feature_row["game_date"]),
        "home_team": str(feature_row["home_team"]),
        "away_team": str(feature_row["away_team"]),
        "market": market,
        "selection": selection,
        "sportsbook_line": float(sportsbook_line),
        "sportsbook_odds": sportsbook_odds,
        "fair_line": float(fair_line),
        "fair_odds": prob_to_american(selected_probability),
        "edge": float(edge),
        "selected_probability": float(selected_probability),
        "market_implied_probability": american_to_prob(sportsbook_odds) if sportsbook_odds is not None else np.nan,
        "model_version": model_version,
        "generated_at_utc": generated_at,
        "published_line": float(sportsbook_line),
        "published_odds": sportsbook_odds,
        "published_at": generated_at,
        "market_snapshot_at": market_snapshot_at,
        "market_readiness_status": "experimental",
        "lineup_context_json": json.dumps(lineup_context),
        "injury_context_json": json.dumps(injury_context),
    }


def build_game_market_recommendations(
    *,
    logs_df: pd.DataFrame,
    odds_snapshots_df: pd.DataFrame,
    models_dir: Path,
    sportsbook: Optional[str],
    target_date: Optional[str],
    min_edge: float,
    injuries_df: Optional[pd.DataFrame] = None,
    lineup_df: Optional[pd.DataFrame] = None,
    starter_history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    latest_rows = _latest_market_rows(odds_snapshots_df, sportsbook=sportsbook, target_date=target_date)
    if latest_rows.empty:
        return pd.DataFrame()

    upcoming_games = _build_upcoming_games(latest_rows)
    feature_frame = build_upcoming_game_market_frame(
        logs_df,
        upcoming_games,
        injuries_df=injuries_df if injuries_df is not None and not injuries_df.empty else None,
        lineup_df=lineup_df if lineup_df is not None and not lineup_df.empty else None,
        starter_history_df=starter_history_df if starter_history_df is not None and not starter_history_df.empty else None,
    )
    if feature_frame.empty:
        return pd.DataFrame()

    latest_lookup = _row_lookup(latest_rows)
    bundles = {
        market: _load_bundle(models_dir, market)
        for market in ("game_moneyline", "game_spread", "game_total")
    }

    rows = []
    for _, feature_row in feature_frame.iterrows():
        X = (
            pd.DataFrame([feature_row.to_dict()])
            .reindex(columns=GAME_MARKET_FEATURE_COLUMNS, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        fixture_id = str(feature_row.get("fixture_id") or "")
        snapshot_at = str(feature_row.get("market_snapshot_at") or "")

        ml_bundle = bundles["game_moneyline"]
        prob_home = float(ml_bundle["model"].predict_proba(X[ml_bundle["feature_cols"]])[:, 1][0])
        prob_away = 1.0 - prob_home
        ml_home = latest_lookup.get((fixture_id, "game_moneyline", "home"))
        ml_away = latest_lookup.get((fixture_id, "game_moneyline", "away"))
        if ml_home and ml_away:
            home_edge = prob_home - american_to_prob(ml_home.get("price"))
            away_edge = prob_away - american_to_prob(ml_away.get("price"))
            if max(home_edge, away_edge) >= min_edge:
                if home_edge >= away_edge:
                    rows.append(
                        _recommendation_row(
                            feature_row=feature_row,
                            market="game_moneyline",
                            selection="home",
                            sportsbook_line=0.0,
                            sportsbook_odds=float(ml_home.get("price")),
                            fair_line=0.0,
                            selected_probability=prob_home,
                            edge=home_edge,
                            market_snapshot_at=snapshot_at,
                            model_version=_resolve_model_version(ml_bundle, "game_moneyline"),
                        )
                    )
                else:
                    rows.append(
                        _recommendation_row(
                            feature_row=feature_row,
                            market="game_moneyline",
                            selection="away",
                            sportsbook_line=0.0,
                            sportsbook_odds=float(ml_away.get("price")),
                            fair_line=0.0,
                            selected_probability=prob_away,
                            edge=away_edge,
                            market_snapshot_at=snapshot_at,
                            model_version=_resolve_model_version(ml_bundle, "game_moneyline"),
                        )
                    )

        spread_bundle = bundles["game_spread"]
        pred_margin = float(spread_bundle["model"].predict(X[spread_bundle["feature_cols"]])[0])
        sigma_spread = float(spread_bundle.get("sigma") or 1.0)
        spread_home = latest_lookup.get((fixture_id, "game_spread", "home"))
        spread_away = latest_lookup.get((fixture_id, "game_spread", "away"))
        if spread_home and spread_away:
            threshold = float(spread_home.get("line_value") or spread_away.get("line_value") or 0.0)
            prob_home_cover = float(1.0 - norm.cdf(threshold, loc=pred_margin, scale=sigma_spread))
            prob_away_cover = float(norm.cdf(threshold, loc=pred_margin, scale=sigma_spread))
            home_edge = prob_home_cover - american_to_prob(spread_home.get("price"))
            away_edge = prob_away_cover - american_to_prob(spread_away.get("price"))
            if max(home_edge, away_edge) >= min_edge:
                if home_edge >= away_edge:
                    rows.append(
                        _recommendation_row(
                            feature_row=feature_row,
                            market="game_spread",
                            selection="home",
                            sportsbook_line=threshold,
                            sportsbook_odds=float(spread_home.get("price")),
                            fair_line=pred_margin,
                            selected_probability=prob_home_cover,
                            edge=home_edge,
                            market_snapshot_at=snapshot_at,
                            model_version=_resolve_model_version(spread_bundle, "game_spread"),
                        )
                    )
                else:
                    rows.append(
                        _recommendation_row(
                            feature_row=feature_row,
                            market="game_spread",
                            selection="away",
                            sportsbook_line=threshold,
                            sportsbook_odds=float(spread_away.get("price")),
                            fair_line=pred_margin,
                            selected_probability=prob_away_cover,
                            edge=away_edge,
                            market_snapshot_at=snapshot_at,
                            model_version=_resolve_model_version(spread_bundle, "game_spread"),
                        )
                    )

        total_bundle = bundles["game_total"]
        pred_total = float(total_bundle["model"].predict(X[total_bundle["feature_cols"]])[0])
        sigma_total = float(total_bundle.get("sigma") or 1.0)
        total_over = latest_lookup.get((fixture_id, "game_total", "over"))
        total_under = latest_lookup.get((fixture_id, "game_total", "under"))
        if total_over and total_under:
            total_line = float(total_over.get("line_value") or total_under.get("line_value") or 0.0)
            prob_over = float(1.0 - norm.cdf(total_line, loc=pred_total, scale=sigma_total))
            prob_under = float(norm.cdf(total_line, loc=pred_total, scale=sigma_total))
            over_edge = prob_over - american_to_prob(total_over.get("price"))
            under_edge = prob_under - american_to_prob(total_under.get("price"))
            if max(over_edge, under_edge) >= min_edge:
                if over_edge >= under_edge:
                    rows.append(
                        _recommendation_row(
                            feature_row=feature_row,
                            market="game_total",
                            selection="over",
                            sportsbook_line=total_line,
                            sportsbook_odds=float(total_over.get("price")),
                            fair_line=pred_total,
                            selected_probability=prob_over,
                            edge=over_edge,
                            market_snapshot_at=snapshot_at,
                            model_version=_resolve_model_version(total_bundle, "game_total"),
                        )
                    )
                else:
                    rows.append(
                        _recommendation_row(
                            feature_row=feature_row,
                            market="game_total",
                            selection="under",
                            sportsbook_line=total_line,
                            sportsbook_odds=float(total_under.get("price")),
                            fair_line=pred_total,
                            selected_probability=prob_under,
                            edge=under_edge,
                            market_snapshot_at=snapshot_at,
                            model_version=_resolve_model_version(total_bundle, "game_total"),
                        )
                    )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["game_date", "edge"], ascending=[True, False]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score game-level NBA recommendations from stored odds snapshots.")
    parser.add_argument("--logs-csv", default=str(LOGS_CSV))
    parser.add_argument("--odds-snapshots-csv", default=str(ODDS_SNAPSHOTS_CSV))
    parser.add_argument("--injuries-csv", default=str(INJURIES_CSV))
    parser.add_argument("--lineups-csv", default=str(LINEUPS_CSV))
    parser.add_argument("--starters-csv", default=str(STARTERS_CSV))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--sportsbook", default=None)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--min-edge", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs = pd.read_csv(args.logs_csv)
    odds = pd.read_csv(args.odds_snapshots_csv)
    injuries = _load_optional_csv(Path(args.injuries_csv))
    lineups = _load_optional_csv(Path(args.lineups_csv))
    starters = _load_optional_csv(Path(args.starters_csv))

    output_df = build_game_market_recommendations(
        logs_df=logs,
        odds_snapshots_df=odds,
        models_dir=Path(args.models_dir),
        sportsbook=args.sportsbook,
        target_date=args.target_date,
        min_edge=float(args.min_edge),
        injuries_df=injuries if not injuries.empty else None,
        lineup_df=lineups if not lineups.empty else None,
        starter_history_df=starters if not starters.empty else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output, index=False)
    print(f"[INFO] Wrote {len(output_df)} game-market recommendation row(s): {output}")


if __name__ == "__main__":
    main()
