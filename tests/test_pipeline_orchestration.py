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


def test_import_historical_game_odds_writes_canonical_artifacts(tmp_path, monkeypatch):
    manifest = tmp_path / "source_manifest.json"
    canonical = tmp_path / "canonical.csv"
    conflicts = tmp_path / "conflicts.csv"
    manifest.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(pipeline, "CANONICAL_HISTORICAL_ODDS_CSV", canonical)
    monkeypatch.setattr(pipeline, "HISTORICAL_ODDS_CONFLICTS_CSV", conflicts)
    monkeypatch.setattr(
        pipeline,
        "import_historical_odds_sources",
        lambda manifest_path: pd.DataFrame([{"source_name": "a"}]),
    )
    monkeypatch.setattr(
        pipeline,
        "reconcile_historical_odds",
        lambda source_rows: (
            pd.DataFrame([{"game_date": "2026-01-10", "market": "spread"}]),
            pd.DataFrame([{"game_date": "2026-01-10", "market": "spread"}]),
        ),
    )
    monkeypatch.setattr(pipeline, "write_historical_odds_artifacts", lambda *args, **kwargs: canonical.write_text("ok", encoding="utf-8"))
    monkeypatch.setattr(pipeline, "persist_historical_odds", lambda canonical_df, conflicts_df, database_url: (len(canonical_df), len(conflicts_df)))

    result = pipeline.import_historical_game_odds(
        manifest_path=manifest,
        database_url="sqlite:///ignored.db",
    )

    assert result["source_rows"] == 1
    assert result["canonical_rows"] == 1
    assert result["conflict_rows"] == 1
    assert result["persisted_odds_rows"] == 1


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
            game_odds_source="scoresandodds",
            props_source="scoresandodds",
            skip_star_screener=True,
        )

    assert events == ["build_prop_feature_stack", "ingest_official_injuries"]
    assert [step["name"] for step in run_log["steps"]] == [
        "build_prop_feature_stack",
        "ingest_official_injuries",
    ]


def test_active_teams_for_report_day_uses_same_day_injury_rows(tmp_path, monkeypatch):
    injuries_csv = tmp_path / "official_injuries.csv"
    pd.DataFrame(
        [
            {"report_date": "2026-04-04", "game_date": "2026-04-04", "team_abbrev": "mia"},
            {"report_date": "2026-04-04", "game_date": "2026-04-04", "team_abbrev": "atl"},
            {"report_date": "2026-04-03", "game_date": "2026-04-03", "team_abbrev": "bos"},
        ]
    ).to_csv(injuries_csv, index=False)

    teams = pipeline.active_teams_for_report_day(date(2026, 4, 4), injuries_csv=injuries_csv)

    assert teams == ["ATL", "MIA"]


def test_refresh_starter_history_limits_cold_start_to_active_teams(monkeypatch, tmp_path):
    original_read_csv = pd.read_csv
    logs_df = pd.DataFrame(
        [
            {
                "game_id": f"22501{i:03d}",
                "game_date": f"2026-04-{i:02d}",
                "team_abbrev": "MIA",
                "opp_abbrev": "ATL",
            }
            for i in range(1, 13)
        ]
        + [
            {
                "game_id": f"22502{i:03d}",
                "game_date": f"2026-04-{i:02d}",
                "team_abbrev": "ATL",
                "opp_abbrev": "MIA",
            }
            for i in range(1, 5)
        ]
        + [
            {
                "game_id": f"22503{i:03d}",
                "game_date": f"2026-04-{i:02d}",
                "team_abbrev": "BOS",
                "opp_abbrev": "NYK",
            }
            for i in range(1, 9)
        ]
    )
    injuries_csv = tmp_path / "official_injuries.csv"
    pd.DataFrame(
        [
            {"report_date": "2026-04-12", "game_date": "2026-04-12", "team_abbrev": "MIA"},
            {"report_date": "2026-04-12", "game_date": "2026-04-12", "team_abbrev": "ATL"},
        ]
    ).to_csv(injuries_csv, index=False)

    captured: dict[str, object] = {}

    def fake_read_csv(path, *args, **kwargs):
        if path == "data/player_game_logs.csv":
            return logs_df.copy()
        return original_read_csv(path, *args, **kwargs)

    def fake_build(logs_df, *, existing_game_ids=None, max_games=None, fetch_timeout_seconds=10):
        captured["logs_df"] = logs_df.copy()
        captured["max_games"] = max_games
        return pd.DataFrame()

    monkeypatch.setattr(pipeline.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pipeline, "OFFICIAL_INJURIES_CSV", injuries_csv)
    monkeypatch.setattr(pipeline, "STARTER_HISTORY_CSV", tmp_path / "starter_history.csv")
    monkeypatch.setattr(pipeline, "build_starter_history_frame", fake_build)
    monkeypatch.setattr(pipeline, "persist_starter_history", lambda frame, output_path, database_url: 0)

    result = pipeline.refresh_starter_history("sqlite:///ignored.db", report_day=date(2026, 4, 12))

    scoped_logs = captured["logs_df"]
    assert set(scoped_logs["team_abbrev"].unique()) == {"ATL", "MIA"}
    assert scoped_logs[scoped_logs["team_abbrev"] == "MIA"]["game_id"].nunique() == 10
    assert scoped_logs[scoped_logs["team_abbrev"] == "ATL"]["game_id"].nunique() == 4
    assert captured["max_games"] == 30
    assert result["rows_persisted"] == 0


def test_score_and_materialize_live_game_markets_skips_when_no_models(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = pipeline.score_and_materialize_live_game_markets(
        date(2026, 4, 4),
        "sqlite:///ignored.db",
        ["draftkings"],
    )

    assert result == {"rows_materialized": 0, "status": "skipped_no_game_market_models"}


def test_bootstrap_mode_continues_after_historical_backfill_failure(tmp_path, monkeypatch):
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
    monkeypatch.setattr(pipeline, "import_historical_game_odds", mark("import_historical_game_odds"))
    monkeypatch.setattr(pipeline, "backfill_historical_market_data", mark("backfill_historical_market_data"))
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
        historical_manifest_path=tmp_path / "source_manifest.json",
        bookmakers=["pinnacle"],
        game_odds_source="scoresandodds",
        props_source="scoresandodds",
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
        "import_historical_game_odds",
        "backfill_historical_market_data",
        "build_prop_feature_stack",
        "train_prop_models",
        "train_game_models",
        "replay_historical_range",
        "settle_and_refresh_readiness",
    ]
