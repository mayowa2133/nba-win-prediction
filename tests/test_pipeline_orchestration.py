from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from src.jobs import replay_historical_recommendations as replay_job
from src.pipeline import run_full_slate_pipeline as pipeline


def test_backfill_official_injuries_resumes_from_cursor(tmp_path, monkeypatch):
    cursor_path = tmp_path / "injury_cursor.json"
    cursor_path.write_text(
        json.dumps({"last_completed_date": "2026-01-02"}),
        encoding="utf-8",
    )

    seen_dates: list[str] = []

    def fake_fetch(*, report_date, latest_only):
        seen_dates.append(report_date.isoformat())
        return pd.DataFrame([{"report_date": report_date.isoformat()}])

    monkeypatch.setattr(pipeline, "fetch_official_injury_reports", fake_fetch)
    monkeypatch.setattr(
        pipeline,
        "persist_official_injury_reports",
        lambda report_df, output_path, database_url: len(report_df),
    )

    result = pipeline.backfill_official_injuries(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 3),
        output_path=tmp_path / "official_injuries.csv",
        database_url="sqlite:///ignored.db",
        cursor_path=cursor_path,
        reset_cursor=False,
    )

    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert seen_dates == ["2026-01-03"]
    assert result["days_completed"] == 1
    assert result["rows_persisted"] == 1
    assert cursor["last_completed_date"] == "2026-01-03"


def test_backfill_game_odds_processes_seven_day_chunks(tmp_path, monkeypatch):
    cursor_path = tmp_path / "game_odds_cursor.json"
    requested_chunks: list[tuple[str, str]] = []

    monkeypatch.setattr(pipeline, "get_odds_papi_api_key", lambda _: "free-key")

    def fake_fetch(*, start_date, end_date, api_key, bookmakers):
        requested_chunks.append((start_date.isoformat(), end_date.isoformat()))
        return [{"fixtureId": f"{start_date.isoformat()}_{end_date.isoformat()}"}]

    monkeypatch.setattr(pipeline, "fetch_historical_game_odds_snapshots", fake_fetch)
    monkeypatch.setattr(
        pipeline,
        "persist_game_odds",
        lambda snapshots, snapshots_output_path, closing_output_path, database_url: (len(snapshots), len(snapshots)),
    )

    result = pipeline.backfill_game_odds(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 10),
        output_path=tmp_path / "game_odds_snapshots.csv",
        closing_output_path=tmp_path / "closing_lines.csv",
        database_url="sqlite:///ignored.db",
        bookmakers=["pinnacle"],
        cursor_path=cursor_path,
        reset_cursor=False,
    )

    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert requested_chunks == [("2026-01-01", "2026-01-07"), ("2026-01-08", "2026-01-10")]
    assert result["chunks_completed"] == 2
    assert result["snapshot_rows"] == 2
    assert cursor["last_completed_chunk_end"] == "2026-01-10"


def test_resume_months_skips_completed_replay_months(tmp_path):
    cursor_path = tmp_path / "historical_replay_cursor.json"
    cursor_path.write_text(
        json.dumps({"last_completed_month": "2025-11-01"}),
        encoding="utf-8",
    )

    months = replay_job.resume_months(
        start_date=date(2025, 10, 1),
        end_date=date(2026, 1, 31),
        cursor_path=cursor_path,
        reset_cursor=False,
    )

    assert [month.isoformat() for month in months] == ["2025-12-01", "2026-01-01"]


def test_replay_historical_range_forwards_reset_cursor_flag(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda command: commands.append(command))

    pipeline.replay_historical_range(
        start_date=date(2025, 10, 1),
        end_date=date(2025, 10, 31),
        database_url="sqlite:///ignored.db",
        reset_cursor=True,
    )

    assert commands
    assert "--reset-cursor" in commands[0]


def test_daily_mode_aborts_when_same_day_official_injuries_are_missing(monkeypatch):
    events: list[str] = []

    def mark(name):
        def _inner(*args, **kwargs):
            events.append(name)
            return {}

        return _inner

    def fail_require_rows(path, date_value, *date_columns):
        if path == pipeline.OFFICIAL_INJURIES_CSV:
            raise RuntimeError("Expected same-day official injuries")
        return 1

    monkeypatch.setattr(pipeline, "build_prop_feature_stack", mark("build_prop_feature_stack"))
    monkeypatch.setattr(pipeline, "ingest_official_injuries_for_day", mark("ingest_official_injuries"))
    monkeypatch.setattr(pipeline, "require_rows", fail_require_rows)
    monkeypatch.setattr(pipeline, "refresh_starter_history", mark("refresh_starter_history"))

    run_log = {"steps": [], "warnings": []}
    with pytest.raises(RuntimeError, match="official injuries"):
        pipeline.run_daily_mode(
            date(2026, 3, 31),
            "sqlite:///ignored.db",
            ["pinnacle"],
            run_log,
            skip_star_screener=True,
        )

    assert events == ["build_prop_feature_stack", "ingest_official_injuries"]
    assert [step["name"] for step in run_log["steps"]] == [
        "build_prop_feature_stack",
        "ingest_official_injuries",
    ]


def test_bootstrap_mode_continues_after_historical_backfill_failure(monkeypatch):
    events: list[str] = []

    def mark(name, *, error: Exception | None = None):
        def _inner(*args, **kwargs):
            events.append(name)
            if error is not None:
                raise error
            return {"step": name}

        return _inner

    monkeypatch.setattr(pipeline, "init_database", lambda url: SimpleNamespace(url=url))
    monkeypatch.setattr(
        pipeline,
        "backfill_official_injuries",
        mark("backfill_official_injuries", error=RuntimeError("injury backfill failed")),
    )
    monkeypatch.setattr(pipeline, "refresh_starter_history", mark("refresh_starter_history"))
    monkeypatch.setattr(pipeline, "backfill_game_odds", mark("backfill_game_odds"))
    monkeypatch.setattr(pipeline, "build_prop_feature_stack", mark("build_prop_feature_stack"))
    monkeypatch.setattr(pipeline, "train_prop_models", mark("train_prop_models"))
    monkeypatch.setattr(pipeline, "train_game_models", mark("train_game_models"))
    monkeypatch.setattr(pipeline, "replay_historical_range", mark("replay_historical_range"))
    monkeypatch.setattr(pipeline, "settle_and_refresh_readiness", mark("settle_and_refresh_readiness"))

    run_log = {"steps": [], "warnings": []}
    pipeline.run_bootstrap_or_backfill_mode(
        mode="bootstrap",
        report_day=date(2026, 3, 31),
        backfill_start=date(2025, 10, 1),
        backfill_end=date(2026, 3, 30),
        database_url="sqlite:///ignored.db",
        bookmakers=["pinnacle"],
        run_log=run_log,
        skip_prop_training=False,
        skip_game_training=False,
        publish_current_day_at_end=False,
        reset_cursors=True,
        skip_star_screener=True,
    )

    statuses = {step["name"]: step["status"] for step in run_log["steps"]}
    assert statuses["backfill_official_injuries"] == "failed"
    assert statuses["settle_refresh_readiness"] == "completed"
    assert any("backfill_official_injuries" in warning for warning in run_log["warnings"])
    assert events == [
        "backfill_official_injuries",
        "refresh_starter_history",
        "backfill_game_odds",
        "build_prop_feature_stack",
        "train_prop_models",
        "train_game_models",
        "replay_historical_range",
        "settle_and_refresh_readiness",
    ]
