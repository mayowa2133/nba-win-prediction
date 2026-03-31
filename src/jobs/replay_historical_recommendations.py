"""Historical walk-forward replay for readiness bootstrap."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from src.data.build_lineup_projections import build_lineup_projection_frame
from src.evaluation.settle_recommendations import (
    _load_recommendations_from_database,
    persist_settlements,
    settle_recommendations_frame,
)
from src.inference.score_game_markets import build_game_market_recommendations
from src.warehouse.db import get_database_url
from src.warehouse.materialize import materialize_edges


DEFAULT_CURSOR_PATH = Path("data/pipeline_state/historical_replay_cursor.json")
DEFAULT_FEATURES_CSV = Path("data/player_points_features_with_vegas.csv")
DEFAULT_LOGS_CSV = Path("data/player_game_logs.csv")
DEFAULT_OFFICIAL_INJURIES_CSV = Path("data/official_injuries.csv")
DEFAULT_STARTER_HISTORY_CSV = Path("data/starter_history.csv")
DEFAULT_PLAYER_POSITIONS_CSV = Path("data/player_positions.csv")
DEFAULT_GAME_ODDS_CSV = Path("data/game_odds_snapshots.csv")
DEFAULT_PROPS_MARKET_DIR = Path("data/props_market")
DEFAULT_GAME_MODEL_METRICS_CSV = Path("data/game_market_model_metrics.csv")


PROP_MODEL_SPECS = [
    ("target_pts", "points_regression.pkl", "points_val_preds.csv"),
    ("target_reb", "rebounds_regression.pkl", "rebounds_val_preds.csv"),
    ("target_ast", "assists_regression.pkl", "assists_val_preds.csv"),
    ("target_fg3m", "threes_regression.pkl", "threes_val_preds.csv"),
]


@dataclass(frozen=True)
class ReplaySummary:
    months_completed: int
    dates_scored: int
    recommendations_materialized: int
    settlements_persisted: int


def month_start_iter(start_date: date, end_date: date) -> Iterator[date]:
    current = start_date.replace(day=1)
    last = end_date.replace(day=1)
    while current <= last:
        yield current
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)


def month_end(month_start: date, hard_end: date) -> date:
    last_day = monthrange(month_start.year, month_start.month)[1]
    return min(month_start.replace(day=last_day), hard_end)


def load_cursor(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cursor(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def resume_months(start_date: date, end_date: date, cursor_path: Path, *, reset_cursor: bool) -> list[date]:
    months = list(month_start_iter(start_date, end_date))
    if reset_cursor:
        return months

    cursor = load_cursor(cursor_path)
    last_completed = cursor.get("last_completed_month")
    if not last_completed:
        return months
    completed = date.fromisoformat(last_completed)
    return [month for month in months if month >= next_month_start(completed)]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def train_prop_models_for_month(
    *,
    features_df: pd.DataFrame,
    cutoff_date: date,
    model_dir: Path,
    python_bin: str,
) -> None:
    filtered = features_df[pd.to_datetime(features_df["game_date"], errors="coerce") < pd.to_datetime(cutoff_date)].copy()
    if filtered.empty:
        raise RuntimeError(f"No prop features available before replay cutoff {cutoff_date.isoformat()}")

    filtered_path = model_dir / "features_before_cutoff.csv"
    filtered.to_csv(filtered_path, index=False)

    for target_col, model_name, val_name in PROP_MODEL_SPECS:
        command = [
            python_bin,
            "src/models/build_points_regression.py",
            "--features-csv",
            str(filtered_path),
            "--target-col",
            target_col,
            "--model-path",
            str(model_dir / model_name),
            "--val-preds-out",
            str(model_dir / val_name),
        ]
        run_command(command)


def train_game_models_for_month(
    *,
    logs_csv: Path,
    injuries_csv: Path,
    starter_history_csv: Path,
    model_dir: Path,
    metrics_out: Path,
    cutoff_date: date,
    python_bin: str,
) -> None:
    command = [
        python_bin,
        "src/models/build_game_market_models.py",
        "--logs-csv",
        str(logs_csv),
        "--injuries-csv",
        str(injuries_csv),
        "--starters-csv",
        str(starter_history_csv),
        "--models-dir",
        str(model_dir),
        "--metrics-out",
        str(metrics_out),
        "--train-cutoff-date",
        cutoff_date.isoformat(),
    ]
    run_command(command)


def score_prop_markets_for_date(
    *,
    target_date: date,
    model_dir: Path,
    features_csv: Path,
    market_lines_path: Path,
    python_bin: str,
    output_path: Path,
    min_edge: float,
) -> pd.DataFrame:
    if not market_lines_path.exists():
        return pd.DataFrame()

    command = [
        python_bin,
        "src/inference/scan_slate_with_model.py",
        "--model-paths",
        (
            f"player_points={model_dir / 'points_regression.pkl'},"
            f"player_rebounds={model_dir / 'rebounds_regression.pkl'},"
            f"player_assists={model_dir / 'assists_regression.pkl'},"
            f"player_threes={model_dir / 'threes_regression.pkl'}"
        ),
        "--features-csv",
        str(features_csv),
        "--market-lines",
        str(market_lines_path),
        "--output",
        str(output_path),
        "--min-edge",
        str(min_edge),
    ]
    run_command(command)
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def score_game_markets_for_date(
    *,
    target_date: date,
    logs_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    model_dir: Path,
    injuries_df: pd.DataFrame,
    starter_history_df: pd.DataFrame,
    player_positions_df: pd.DataFrame,
    sportsbook: Optional[str],
    min_edge: float,
) -> pd.DataFrame:
    date_str = target_date.isoformat()
    day_odds = odds_df[odds_df["game_date"].astype(str) == date_str].copy()
    if day_odds.empty:
        return pd.DataFrame()

    day_lineups = build_lineup_projection_frame(
        target_date=target_date,
        starter_history_df=starter_history_df,
        logs_df=logs_df,
        injuries_df=injuries_df,
        player_positions_df=player_positions_df,
    )
    return build_game_market_recommendations(
        logs_df=logs_df,
        odds_snapshots_df=day_odds,
        models_dir=model_dir,
        sportsbook=sportsbook,
        target_date=date_str,
        min_edge=min_edge,
        injuries_df=injuries_df,
        lineup_df=day_lineups,
        starter_history_df=starter_history_df,
    )


def materialize_replay_frame(frame: pd.DataFrame, *, database_url: str, temp_path: Path) -> int:
    if frame.empty:
        return 0
    frame = frame.copy()
    frame["recommendation_origin"] = "historical_replay"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temp_path, index=False)
    scored_count, _ = materialize_edges(
        temp_path,
        database_url=database_url,
        recommendation_origin="historical_replay",
        persist_readiness=False,
    )
    return scored_count


def settle_replay_rows(*, logs_df: pd.DataFrame, database_url: str, closing_lines_csv: Path) -> int:
    closing_df = pd.read_csv(closing_lines_csv) if closing_lines_csv.exists() else pd.DataFrame()
    recommendations = _load_recommendations_from_database(database_url)
    if recommendations.empty:
        return 0
    recommendations = recommendations[recommendations.get("recommendation_origin", "live_daily").astype(str) == "historical_replay"].copy()
    if recommendations.empty:
        return 0

    updated_df, settled_df = settle_recommendations_frame(
        recommendations,
        logs_df=logs_df,
        closing_lines_df=closing_df if not closing_df.empty else None,
    )
    return persist_settlements(updated_df, settled_df, database_url=database_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay historical recommendations month-by-month for readiness bootstrap.")
    parser.add_argument("--start-date", required=True, help="Replay start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Replay end date in YYYY-MM-DD.")
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--logs-csv", default=str(DEFAULT_LOGS_CSV))
    parser.add_argument("--official-injuries-csv", default=str(DEFAULT_OFFICIAL_INJURIES_CSV))
    parser.add_argument("--starter-history-csv", default=str(DEFAULT_STARTER_HISTORY_CSV))
    parser.add_argument("--player-positions-csv", default=str(DEFAULT_PLAYER_POSITIONS_CSV))
    parser.add_argument("--game-odds-csv", default=str(DEFAULT_GAME_ODDS_CSV))
    parser.add_argument("--props-market-dir", default=str(DEFAULT_PROPS_MARKET_DIR))
    parser.add_argument("--closing-lines-csv", default="data/closing_lines.csv")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--cursor-path", default=str(DEFAULT_CURSOR_PATH))
    parser.add_argument("--reset-cursor", action="store_true")
    parser.add_argument("--sportsbook", default=None)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    cursor_path = Path(args.cursor_path)
    database_url = get_database_url(args.database_url)

    features_df = pd.read_csv(args.features_csv)
    logs_df = pd.read_csv(args.logs_csv)
    injuries_df = pd.read_csv(args.official_injuries_csv) if Path(args.official_injuries_csv).exists() else pd.DataFrame()
    starter_history_df = pd.read_csv(args.starter_history_csv) if Path(args.starter_history_csv).exists() else pd.DataFrame()
    player_positions_df = pd.read_csv(args.player_positions_csv) if Path(args.player_positions_csv).exists() else pd.DataFrame()
    odds_df = pd.read_csv(args.game_odds_csv) if Path(args.game_odds_csv).exists() else pd.DataFrame()
    props_market_dir = Path(args.props_market_dir)

    months = resume_months(start_date, end_date, cursor_path, reset_cursor=bool(args.reset_cursor))
    if not months:
        print("[INFO] Historical replay cursor is already complete for the requested window")
        return

    months_completed = 0
    dates_scored = 0
    recommendations_materialized = 0

    for month in months:
        month_cutoff = month
        month_finish = month_end(month, end_date)
        with tempfile.TemporaryDirectory(prefix=f"replay_{month.isoformat()}_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            models_dir = temp_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            train_prop_models_for_month(
                features_df=features_df,
                cutoff_date=month_cutoff,
                model_dir=models_dir,
                python_bin=args.python_bin,
            )
            train_game_models_for_month(
                logs_csv=Path(args.logs_csv),
                injuries_csv=Path(args.official_injuries_csv),
                starter_history_csv=Path(args.starter_history_csv),
                model_dir=models_dir,
                metrics_out=temp_dir / "game_market_model_metrics.csv",
                cutoff_date=month_cutoff,
                python_bin=args.python_bin,
            )

            day = month
            while day <= month_finish:
                prop_market_file = props_market_dir / f"market_lines_{day.isoformat()}.csv"
                prop_output = temp_dir / f"props_{day.isoformat()}.csv"

                prop_df = score_prop_markets_for_date(
                    target_date=day,
                    model_dir=models_dir,
                    features_csv=Path(args.features_csv),
                    market_lines_path=prop_market_file,
                    python_bin=args.python_bin,
                    output_path=prop_output,
                    min_edge=float(args.min_edge),
                )
                game_df = score_game_markets_for_date(
                    target_date=day,
                    logs_df=logs_df,
                    odds_df=odds_df,
                    model_dir=models_dir,
                    injuries_df=injuries_df,
                    starter_history_df=starter_history_df,
                    player_positions_df=player_positions_df,
                    sportsbook=args.sportsbook,
                    min_edge=float(args.min_edge),
                )
                combined = pd.concat([prop_df, game_df], ignore_index=True) if not prop_df.empty or not game_df.empty else pd.DataFrame()
                if not combined.empty:
                    combined["recommendation_origin"] = "historical_replay"
                    materialized = materialize_replay_frame(
                        combined,
                        database_url=database_url,
                        temp_path=temp_dir / f"combined_{day.isoformat()}.csv",
                    )
                    recommendations_materialized += materialized
                    dates_scored += 1
                day += timedelta(days=1)

        months_completed += 1
        save_cursor(
            cursor_path,
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "last_completed_month": month.isoformat(),
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            },
        )

    settlements_persisted = settle_replay_rows(
        logs_df=logs_df,
        database_url=database_url,
        closing_lines_csv=Path(args.closing_lines_csv),
    )
    summary = ReplaySummary(
        months_completed=months_completed,
        dates_scored=dates_scored,
        recommendations_materialized=recommendations_materialized,
        settlements_persisted=settlements_persisted,
    )
    print(
        "[INFO] Historical replay complete: "
        f"months={summary.months_completed}, "
        f"dates_scored={summary.dates_scored}, "
        f"materialized={summary.recommendations_materialized}, "
        f"settlements={summary.settlements_persisted}"
    )


if __name__ == "__main__":
    main()
