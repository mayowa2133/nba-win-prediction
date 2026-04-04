#!/usr/bin/env python
"""Settle persisted recommendations against final NBA results and closing lines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import delete, select

from src.utils.betting import clv_price_component, roi_for_result
from src.utils.game_market_features import build_team_games
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import ClosingLineRecord, RecommendationRecord, SettledBetOutcomeRecord


LOGS_CSV = Path("data/player_game_logs.csv")
RECOMMENDATIONS_CSV = Path("data/game_market_recommendations.csv")
CLOSING_LINES_CSV = Path("data/closing_lines.csv")
SETTLED_OUTPUT = Path("data/settled_recommendations.csv")


PLAYER_MARKET_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
}


def build_game_results(logs_df: pd.DataFrame) -> pd.DataFrame:
    team_games = build_team_games(logs_df)
    if team_games.empty:
        return pd.DataFrame()
    home = team_games[team_games["is_home"] == 1][["game_id", "game_date", "team_abbrev", "opp_abbrev", "points_for", "points_against", "margin", "total_points"]].copy()
    away = team_games[team_games["is_home"] == 0][["game_id", "team_abbrev", "points_for", "points_against"]].copy()
    home = home.rename(columns={"team_abbrev": "home_team_abbrev", "opp_abbrev": "away_team_abbrev", "points_for": "home_score", "points_against": "away_score"})
    away = away.rename(columns={"team_abbrev": "away_team_abbrev_confirm"})
    results = home.merge(away, left_on="game_id", right_on="game_id", how="left")
    results = results.rename(columns={"game_date": "game_date"})
    results["home_win"] = (results["margin"] > 0).astype(int)
    results["home_margin"] = pd.to_numeric(results["margin"], errors="coerce").fillna(0.0)
    results["game_total"] = pd.to_numeric(results["total_points"], errors="coerce").fillna(0.0)
    return results[["game_id", "game_date", "home_team_abbrev", "away_team_abbrev", "home_score", "away_score", "home_win", "home_margin", "game_total"]]


def build_player_results(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame()
    logs = logs_df.copy()
    logs["game_id"] = logs["game_id"].astype(str)
    logs["game_date"] = logs["game_date"].astype(str)
    logs = logs.sort_values(["game_id", "player_name", "player_id"])
    grouped = logs.groupby(["game_id", "game_date", "player_name"], as_index=False).agg(
        pts=("pts", "max"),
        reb=("reb", "max"),
        ast=("ast", "max"),
        fg3m=("fg3m", "max"),
    )
    return grouped


def _clv_line_component(market: str, selection: str, published_line: Optional[float], closing_line: Optional[float]) -> float:
    if published_line is None or closing_line is None or pd.isna(published_line) or pd.isna(closing_line):
        return 0.0
    sign = 1.0 if selection in {"home", "over"} else -1.0
    if market == "game_moneyline":
        return 0.0
    return sign * (float(closing_line) - float(published_line))


def _settle_game_market(row: pd.Series, result_row: pd.Series) -> Tuple[Optional[float], str]:
    market = str(row["market"])
    selection = str(row["selection"])
    line = float(row.get("published_line", row.get("sportsbook_line", 0.0)) or 0.0)

    if market == "game_moneyline":
        actual = float(result_row["home_win"])
        outcome = "win" if (selection == "home" and actual == 1.0) or (selection == "away" and actual == 0.0) else "loss"
        return actual, outcome

    if market == "game_spread":
        actual = float(result_row["home_margin"])
        if selection == "home":
            outcome = "win" if actual > line else "push" if actual == line else "loss"
        else:
            outcome = "win" if actual < line else "push" if actual == line else "loss"
        return actual, outcome

    actual = float(result_row["game_total"])
    if selection == "over":
        outcome = "win" if actual > line else "push" if actual == line else "loss"
    else:
        outcome = "win" if actual < line else "push" if actual == line else "loss"
    return actual, outcome


def _settle_prop_market(row: pd.Series, result_row: pd.Series) -> Tuple[Optional[float], str]:
    market = str(row["market"])
    stat = PLAYER_MARKET_TO_STAT.get(market)
    if stat is None:
        return None, "unsettled"
    actual = float(result_row[stat])
    line = float(row.get("published_line", row.get("sportsbook_line", 0.0)) or 0.0)
    selection = str(row["selection"])
    if selection == "over":
        outcome = "win" if actual > line else "push" if actual == line else "loss"
    else:
        outcome = "win" if actual < line else "push" if actual == line else "loss"
    return actual, outcome


def _closing_line_lookup(closing_lines_df: pd.DataFrame) -> Dict[tuple[str, str, str], pd.Series]:
    if closing_lines_df.empty:
        return {}
    df = closing_lines_df.copy()
    if "closing_captured_at" in df.columns and "captured_at" not in df.columns:
        df["captured_at"] = df["closing_captured_at"]
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce").fillna(0.0).abs()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    latest = df.sort_values("captured_at").groupby(["game_id", "market", "side"], as_index=False).tail(1)
    return {
        (str(row["game_id"]), str(row["market"]), str(row["side"])): row
        for _, row in latest.iterrows()
    }


def settle_recommendations_frame(
    recommendations_df: pd.DataFrame,
    *,
    logs_df: pd.DataFrame,
    closing_lines_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recommendations_df.empty:
        return recommendations_df.copy(), pd.DataFrame()

    recommendations = recommendations_df.copy()
    recommendations["game_id"] = recommendations["game_id"].astype(str)
    recommendations["market"] = recommendations["market"].astype(str)
    recommendations["selection"] = recommendations["selection"].astype(str)
    if "recommendation_origin" not in recommendations.columns:
        recommendations["recommendation_origin"] = "live_daily"
    recommendations["published_line"] = pd.to_numeric(
        recommendations.get("published_line", recommendations.get("sportsbook_line")), errors="coerce"
    )
    recommendations["published_odds"] = pd.to_numeric(
        recommendations.get("published_odds", recommendations.get("sportsbook_odds")), errors="coerce"
    )

    game_results = build_game_results(logs_df)
    player_results = build_player_results(logs_df)
    game_lookup = {str(row["game_id"]): row for _, row in game_results.iterrows()}
    player_lookup = {(str(row["game_id"]), str(row["player_name"])): row for _, row in player_results.iterrows()}
    player_lookup_by_date = {(str(row["game_date"]), str(row["player_name"])): row for _, row in player_results.iterrows()}
    closing_lookup = _closing_line_lookup(closing_lines_df if closing_lines_df is not None else pd.DataFrame())

    updated_rows = []
    settled_rows = []
    settled_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for _, row in recommendations.iterrows():
        market = str(row["market"])
        game_id = str(row["game_id"])
        actual_value = None
        result = "unsettled"
        if market.startswith("game_"):
            result_row = game_lookup.get(game_id)
            if result_row is not None:
                actual_value, result = _settle_game_market(row, result_row)
        else:
            player_name = str(row.get("player") or "")
            player_key = (game_id, player_name)
            result_row = player_lookup.get(player_key)
            if result_row is None:
                result_row = player_lookup_by_date.get((str(row["game_date"]), player_name))
            if result_row is not None:
                actual_value, result = _settle_prop_market(row, result_row)

        closing_match = closing_lookup.get((game_id, market, str(row["selection"])))
        closing_line = None
        closing_odds = None
        if closing_match is not None:
            closing_line = float(closing_match.get("line_value") or 0.0)
            closing_odds = float(closing_match.get("price")) if not pd.isna(closing_match.get("price")) else None
        elif not pd.isna(row.get("closing_line")):
            closing_line = float(row.get("closing_line"))
            closing_odds = float(row.get("closing_odds")) if not pd.isna(row.get("closing_odds")) else None

        published_line = float(row["published_line"]) if not pd.isna(row["published_line"]) else None
        published_odds = float(row["published_odds"]) if not pd.isna(row["published_odds"]) else None
        clv_line = _clv_line_component(market, str(row["selection"]), published_line, closing_line)
        clv_price = clv_price_component(published_odds, closing_odds)
        clv = clv_line + clv_price
        roi = roi_for_result(result, published_odds) if result != "unsettled" else None

        updated = row.to_dict()
        updated["actual_value"] = actual_value
        updated["result"] = result if result != "unsettled" else None
        updated["closing_line"] = closing_line
        updated["closing_odds"] = closing_odds
        updated["clv"] = clv if result != "unsettled" else None
        updated["roi"] = roi
        updated_rows.append(updated)

        if result != "unsettled":
            settled_rows.append(
                {
                    "recommendation_id": str(row["recommendation_id"]),
                    "game_id": game_id,
                    "game_date": str(row["game_date"]),
                    "market": market,
                    "selection": str(row["selection"]),
                    "recommendation_origin": str(row.get("recommendation_origin") or "live_daily"),
                    "published_line": published_line,
                    "published_odds": published_odds,
                    "closing_line": closing_line,
                    "closing_odds": closing_odds,
                    "actual_value": actual_value,
                    "result": result,
                    "clv": clv,
                    "clv_line": clv_line,
                    "clv_price": clv_price,
                    "roi": roi,
                    "settled_at": settled_at,
                }
            )

    return pd.DataFrame(updated_rows), pd.DataFrame(settled_rows)


def _load_recommendations_from_database(database_url: str) -> pd.DataFrame:
    with session_scope(database_url) as session:
        rows = session.execute(select(RecommendationRecord)).scalars().all()
    return pd.DataFrame(
        [
            {
                "recommendation_id": row.id,
                "game_id": row.game_id,
                "player": row.player,
                "game_date": row.game_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "market": row.market,
                "selection": row.selection,
                "recommendation_origin": row.recommendation_origin,
                "sportsbook_line": row.sportsbook_line,
                "sportsbook_odds": row.sportsbook_odds,
                "published_line": row.published_line,
                "published_odds": row.published_odds,
                "closing_line": row.closing_line,
                "closing_odds": row.closing_odds,
            }
            for row in rows
        ]
    )


def persist_settlements(
    updated_recommendations_df: pd.DataFrame,
    settled_df: pd.DataFrame,
    *,
    database_url: Optional[str] = None,
) -> int:
    if database_url is None or settled_df.empty:
        return 0

    init_database(database_url)
    with session_scope(database_url) as session:
        recommendation_ids = settled_df["recommendation_id"].astype(str).unique().tolist()
        if recommendation_ids:
            session.execute(delete(SettledBetOutcomeRecord).where(SettledBetOutcomeRecord.recommendation_id.in_(recommendation_ids)))

        for row in updated_recommendations_df.to_dict(orient="records"):
            record = session.get(RecommendationRecord, str(row["recommendation_id"]))
            if record is None:
                continue
            record.actual_value = row.get("actual_value")
            record.result = row.get("result")
            record.closing_line = row.get("closing_line")
            record.closing_odds = row.get("closing_odds")
            record.clv = row.get("clv")
            record.roi = row.get("roi")

        count = 0
        for row in settled_df.to_dict(orient="records"):
            session.add(
                SettledBetOutcomeRecord(
                    recommendation_id=str(row["recommendation_id"]),
                    game_id=str(row["game_id"]),
                    game_date=str(row["game_date"]),
                    market=str(row["market"]),
                    selection=str(row["selection"]),
                    recommendation_origin=str(row.get("recommendation_origin") or "live_daily"),
                    published_line=row.get("published_line"),
                    published_odds=row.get("published_odds"),
                    closing_line=row.get("closing_line"),
                    closing_odds=row.get("closing_odds"),
                    actual_value=row.get("actual_value"),
                    result=str(row["result"]),
                    clv=row.get("clv"),
                    clv_line=row.get("clv_line"),
                    clv_price=row.get("clv_price"),
                    roi=row.get("roi"),
                    settled_at=str(row["settled_at"]),
                )
            )
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Settle persisted recommendations against final results.")
    parser.add_argument("--logs-csv", default=str(LOGS_CSV))
    parser.add_argument("--recommendations-csv", default=str(RECOMMENDATIONS_CSV))
    parser.add_argument("--closing-lines-csv", default=str(CLOSING_LINES_CSV))
    parser.add_argument("--output", default=str(SETTLED_OUTPUT))
    parser.add_argument("--database-url", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs = pd.read_csv(args.logs_csv)
    closing_lines = pd.read_csv(args.closing_lines_csv) if Path(args.closing_lines_csv).exists() else pd.DataFrame()

    if args.database_url:
        recommendations = _load_recommendations_from_database(args.database_url)
    else:
        recommendations = pd.read_csv(args.recommendations_csv)

    updated_df, settled_df = settle_recommendations_frame(
        recommendations,
        logs_df=logs,
        closing_lines_df=closing_lines if not closing_lines.empty else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(output, index=False)
    persisted = persist_settlements(updated_df, settled_df, database_url=args.database_url)
    print(f"[INFO] Wrote {len(updated_df)} updated recommendation row(s): {output}")
    if args.database_url:
        print(f"[INFO] Persisted {persisted} settled outcome row(s) to the warehouse")


if __name__ == "__main__":
    main()
