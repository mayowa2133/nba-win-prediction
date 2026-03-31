"""Database helpers for the local warehouse and beta API."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.warehouse.models import Base


DEFAULT_DATABASE_URL = "sqlite:///data/nba_betting_beta.db"


def get_database_url(database_url: Optional[str] = None) -> str:
    return database_url or os.getenv("NBA_BETTING_DATABASE_URL") or DEFAULT_DATABASE_URL


def database_path_from_url(database_url: str) -> Optional[Path]:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    raw_path = database_url[len(prefix):]
    if raw_path == ":memory:":
        return None
    return Path(raw_path)


def default_sqlite_database_path() -> Path:
    path = database_path_from_url(DEFAULT_DATABASE_URL)
    assert path is not None
    return path


@lru_cache(maxsize=8)
def _build_engine(database_url: str) -> Engine:
    db_path = database_path_from_url(database_url)
    connect_args = {}
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args["check_same_thread"] = False
    return create_engine(database_url, future=True, connect_args=connect_args)


def get_engine(database_url: Optional[str] = None) -> Engine:
    return _build_engine(get_database_url(database_url))


def _ensure_sqlite_compatibility(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    add_columns = {
        "injury_reports": {
            "report_date": "TEXT",
            "report_time_et": "TEXT",
            "matchup": "TEXT",
            "row_kind": "TEXT DEFAULT 'player_status'",
            "raw_status": "TEXT",
            "raw_reason": "TEXT",
            "normalized_status": "TEXT",
            "source_url": "TEXT",
            "schema_version": "TEXT DEFAULT '1.0.0'",
            "pulled_at": "TEXT",
        },
        "recommendations": {
            "recommendation_origin": "TEXT DEFAULT 'live_daily'",
            "selected_probability": "REAL",
            "market_implied_probability": "REAL",
            "published_line": "REAL",
            "published_odds": "REAL",
            "published_at": "TEXT",
            "closing_line": "REAL",
            "closing_odds": "REAL",
            "actual_value": "REAL",
            "result": "TEXT",
            "clv": "REAL",
            "roi": "REAL",
            "lineup_context_json": "JSON",
            "injury_context_json": "JSON",
        },
        "settled_bet_outcomes": {
            "recommendation_origin": "TEXT DEFAULT 'live_daily'",
        },
    }

    with engine.begin() as connection:
        for table_name, columns in add_columns.items():
            if not inspector.has_table(table_name):
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, ddl in columns.items():
                if column_name in existing:
                    continue
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))


def init_database(database_url: Optional[str] = None) -> Engine:
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    _ensure_sqlite_compatibility(engine)
    return engine


@contextmanager
def session_scope(database_url: Optional[str] = None) -> Iterator[Session]:
    engine = get_engine(database_url)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
