"""Free-tier game odds ingestion and historical backfill via OddsPapi."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from sqlalchemy import delete

from src.utils.artifact_metadata import stable_id
from src.utils.nba_teams import canonical_team_abbrev, canonical_team_name_from_abbrev
from src.warehouse.db import init_database, session_scope
from src.warehouse.models import ClosingLineRecord, GameOddsSnapshotRecord


DEFAULT_BASE_URL = "https://api.oddspapi.io"
DEFAULT_BOOKMAKERS = ("pinnacle", "bet365")


@dataclass(frozen=True)
class ResolvedTournament:
    sport_id: int
    tournament_id: int
    tournament_name: str


class OddsPapiClient:
    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http = requests.Session()

    def _get(self, endpoint: str, **params):
        payload = {key: value for key, value in params.items() if value is not None}
        payload["apiKey"] = self.api_key
        response = self.http.get(
            f"{self.base_url}{endpoint}",
            params=payload,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        return response.json()

    def get_sports(self):
        return self._get("/v4/sports")

    def get_tournaments(self, *, sport_id: int):
        return self._get("/v4/tournaments", sportId=sport_id, language="en")

    def get_markets(self, *, sport_id: int):
        return self._get("/v4/markets", sportId=sport_id, language="en")

    def get_fixtures(
        self,
        *,
        sport_id: int,
        tournament_id: int,
        from_iso: str,
        to_iso: str,
        status_id: int,
        has_odds: bool = True,
    ):
        return self._get(
            "/v4/fixtures",
            sportId=sport_id,
            tournamentId=tournament_id,
            statusId=status_id,
            hasOdds=str(has_odds).lower(),
            language="en",
            **{"from": from_iso, "to": to_iso},
        )

    def get_odds(self, *, fixture_id: str, bookmakers: str, odds_format: str = "american", verbosity: int = 3):
        return self._get(
            "/v4/odds",
            fixtureId=fixture_id,
            bookmakers=bookmakers,
            oddsFormat=odds_format,
            verbosity=verbosity,
            language="en",
        )

    def get_historical_odds(self, *, fixture_id: str, bookmakers: str):
        return self._get(
            "/v4/historical-odds",
            fixtureId=fixture_id,
            bookmakers=bookmakers,
        )


def get_odds_papi_api_key(cli_key: Optional[str]) -> str:
    api_key = cli_key or os.getenv("ODDSPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("ODDSPAPI_API_KEY is required for game odds ingestion")
    return api_key


def resolve_nba_tournament(client: OddsPapiClient) -> ResolvedTournament:
    sports = client.get_sports()
    basketball = next(
        (item for item in sports if "basketball" in str(item.get("sportName", "")).lower()),
        None,
    )
    if basketball is None:
        raise RuntimeError("Could not resolve Basketball sportId from OddsPapi")

    sport_id = int(basketball["sportId"])
    tournaments = client.get_tournaments(sport_id=sport_id)
    nba = next(
        (
            item
            for item in tournaments
            if str(item.get("tournamentName", "")).strip().lower() == "nba"
            or str(item.get("tournamentSlug", "")).strip().lower() == "nba"
        ),
        None,
    )
    if nba is None:
        raise RuntimeError("Could not resolve NBA tournamentId from OddsPapi")

    return ResolvedTournament(
        sport_id=sport_id,
        tournament_id=int(nba["tournamentId"]),
        tournament_name=str(nba["tournamentName"]),
    )


def market_catalog_by_id(client: OddsPapiClient, *, sport_id: int) -> Dict[str, dict]:
    catalog = {}
    for market in client.get_markets(sport_id=sport_id):
        catalog[str(market.get("marketId"))] = market
    return catalog


def _parse_market_kind(bookmaker_outcome_id: str) -> Optional[str]:
    token = str(bookmaker_outcome_id or "").lower()
    if token in {"home", "away"}:
        return "game_moneyline"
    if "/over" in token or "/under" in token:
        return "game_total"
    if "/home" in token or "/away" in token:
        return "game_spread"
    return None


def _parse_side_and_line(bookmaker_outcome_id: str, *, market_kind: str) -> tuple[str, float]:
    token = str(bookmaker_outcome_id or "").lower()
    if market_kind == "game_moneyline":
        return token, 0.0

    raw_line, _, raw_side = token.partition("/")
    side = raw_side or token
    line_value = abs(float(raw_line or 0.0))

    if market_kind == "game_total":
        return side, line_value
    return side, line_value


def _normalize_odds_rows(
    payload: dict,
    *,
    market_catalog: Dict[str, dict],
    source_url: str,
    is_historical: bool,
) -> List[dict]:
    fixture_id = str(payload.get("fixtureId") or "")
    start_time = str(payload.get("startTime") or "")
    captured_at_default = str(payload.get("updatedAt") or start_time or datetime.now(timezone.utc).isoformat())
    home_team = str(payload.get("participant1Name") or "")
    away_team = str(payload.get("participant2Name") or "")
    home_abbrev = canonical_team_abbrev(home_team) or home_team
    away_abbrev = canonical_team_abbrev(away_team) or away_team
    game_date = start_time[:10] if start_time else ""
    game_id = stable_id(game_date, home_abbrev, away_abbrev, prefix="game")

    rows: List[dict] = []
    bookmaker_odds = payload.get("bookmakerOdds") or payload.get("bookmakers") or {}

    for bookmaker_slug, bookmaker_payload in bookmaker_odds.items():
        markets = bookmaker_payload.get("markets", {})
        for market_id, market_payload in markets.items():
            outcomes = market_payload.get("outcomes", {})
            market_meta = market_catalog.get(str(market_id), {})
            market_name = str(market_meta.get("marketName") or market_payload.get("bookmakerMarketId") or "")
            for outcome_id, outcome_payload in outcomes.items():
                players = outcome_payload.get("players", {})
                player_bucket = players.get("0")
                if player_bucket is None:
                    continue

                entries = player_bucket if isinstance(player_bucket, list) else [player_bucket]
                for entry in entries:
                    bookmaker_outcome_id = str(entry.get("bookmakerOutcomeId") or "")
                    market_kind = _parse_market_kind(bookmaker_outcome_id)
                    if market_kind is None:
                        continue
                    side, line_value = _parse_side_and_line(bookmaker_outcome_id, market_kind=market_kind)
                    captured_at = str(entry.get("createdAt") or entry.get("changedAt") or captured_at_default)

                    rows.append(
                        {
                            "fixture_id": fixture_id,
                            "game_id": game_id,
                            "game_date": game_date,
                            "commence_time": start_time,
                            "home_team": canonical_team_name_from_abbrev(home_abbrev) or home_team,
                            "away_team": canonical_team_name_from_abbrev(away_abbrev) or away_team,
                            "market": market_kind,
                            "side": side,
                            "sportsbook": str(bookmaker_slug),
                            "bookmaker_id": None,
                            "line_value": line_value,
                            "price": float(entry.get("price")) if entry.get("price") is not None else None,
                            "market_id": int(market_id),
                            "market_name": market_name,
                            "is_historical": int(is_historical),
                            "source_url": source_url,
                            "snapshot_type": "historical" if is_historical else "intraday",
                            "captured_at": captured_at,
                        }
                    )

    return rows


def build_game_odds_snapshot_frame(
    payloads: Iterable[dict],
    *,
    market_catalog: Dict[str, dict],
    source_url_prefix: str,
    is_historical: bool,
) -> pd.DataFrame:
    rows: List[dict] = []
    for payload in payloads:
        fixture_id = str(payload.get("fixtureId") or "")
        rows.extend(
            _normalize_odds_rows(
                payload,
                market_catalog=market_catalog,
                source_url=f"{source_url_prefix}{fixture_id}",
                is_historical=is_historical,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "game_id",
                "game_date",
                "commence_time",
                "home_team",
                "away_team",
                "market",
                "side",
                "sportsbook",
                "line_value",
                "price",
                "market_id",
                "market_name",
                "is_historical",
                "source_url",
                "snapshot_type",
                "captured_at",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True).astype(str)
    return frame.drop_duplicates(
        subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
        keep="last",
    )


def build_closing_lines_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()

    df = snapshot_df.copy()
    df["captured_at_ts"] = pd.to_datetime(df["captured_at"], utc=True)
    df["commence_time_ts"] = pd.to_datetime(df["commence_time"], utc=True)
    pre_tip = df[df["captured_at_ts"] <= df["commence_time_ts"]].copy()
    if pre_tip.empty:
        pre_tip = df.copy()

    pre_tip = pre_tip.sort_values("captured_at_ts")
    closing = (
        pre_tip.groupby(["fixture_id", "market", "side", "sportsbook"], as_index=False)
        .tail(1)
        .drop(columns=["captured_at_ts", "commence_time_ts"], errors="ignore")
        .rename(columns={"captured_at": "closing_captured_at"})
    )
    return closing


def persist_game_odds(
    snapshot_df: pd.DataFrame,
    *,
    snapshots_output_path: Optional[Path] = None,
    closing_output_path: Optional[Path] = None,
    database_url: Optional[str] = None,
) -> tuple[int, int]:
    closing_df = build_closing_lines_frame(snapshot_df)

    if snapshots_output_path is not None:
        snapshots_output_path.parent.mkdir(parents=True, exist_ok=True)
        if snapshots_output_path.exists():
            existing = pd.read_csv(snapshots_output_path)
            snapshot_out = pd.concat([existing, snapshot_df], ignore_index=True)
            snapshot_out = snapshot_out.drop_duplicates(
                subset=["fixture_id", "market", "side", "sportsbook", "captured_at", "line_value", "price"],
                keep="last",
            )
        else:
            snapshot_out = snapshot_df.copy()
        snapshot_out.to_csv(snapshots_output_path, index=False)

    if closing_output_path is not None:
        closing_output_path.parent.mkdir(parents=True, exist_ok=True)
        if closing_output_path.exists():
            existing_closing = pd.read_csv(closing_output_path)
            closing_out = pd.concat([existing_closing, closing_df], ignore_index=True)
            closing_out = closing_out.drop_duplicates(
                subset=["fixture_id", "market", "side", "sportsbook"],
                keep="last",
            )
        else:
            closing_out = closing_df.copy()
        closing_out.to_csv(closing_output_path, index=False)

    init_database(database_url)
    with session_scope(database_url) as session:
        fixture_ids = sorted({str(value) for value in snapshot_df.get("fixture_id", pd.Series(dtype=str)).dropna().unique().tolist()})
        if fixture_ids:
            session.execute(delete(GameOddsSnapshotRecord).where(GameOddsSnapshotRecord.fixture_id.in_(fixture_ids)))
            session.execute(delete(ClosingLineRecord).where(ClosingLineRecord.fixture_id.in_(fixture_ids)))

        snapshot_count = 0
        for row in snapshot_df.fillna("").to_dict(orient="records"):
            session.add(
                GameOddsSnapshotRecord(
                    fixture_id=str(row["fixture_id"]),
                    game_id=str(row.get("game_id") or ""),
                    game_date=str(row["game_date"]),
                    commence_time=str(row.get("commence_time") or ""),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    market=str(row["market"]),
                    side=str(row["side"]),
                    sportsbook=str(row["sportsbook"]),
                    bookmaker_id=int(row["bookmaker_id"]) if row.get("bookmaker_id") not in {"", None} else None,
                    line_value=float(row.get("line_value") or 0.0),
                    price=float(row["price"]) if row.get("price") not in {"", None} else None,
                    market_id=int(row["market_id"]) if row.get("market_id") not in {"", None} else None,
                    market_name=str(row.get("market_name") or ""),
                    is_historical=int(row.get("is_historical") or 0),
                    source_url=str(row.get("source_url") or ""),
                    snapshot_type=str(row.get("snapshot_type") or "intraday"),
                    captured_at=str(row["captured_at"]),
                )
            )
            snapshot_count += 1

        closing_count = 0
        for row in closing_df.fillna("").to_dict(orient="records"):
            session.add(
                ClosingLineRecord(
                    fixture_id=str(row["fixture_id"]),
                    game_id=str(row.get("game_id") or ""),
                    game_date=str(row["game_date"]),
                    commence_time=str(row.get("commence_time") or ""),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    market=str(row["market"]),
                    side=str(row["side"]),
                    sportsbook=str(row["sportsbook"]),
                    line_value=float(row.get("line_value") or 0.0),
                    price=float(row["price"]) if row.get("price") not in {"", None} else None,
                    captured_at=str(row["closing_captured_at"]),
                )
            )
            closing_count += 1

    return snapshot_count, closing_count


def fetch_current_game_odds_snapshots(
    *,
    report_date: date,
    api_key: str,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
) -> pd.DataFrame:
    client = OddsPapiClient(api_key)
    tournament = resolve_nba_tournament(client)
    markets = market_catalog_by_id(client, sport_id=tournament.sport_id)
    start = datetime.combine(report_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    fixtures = client.get_fixtures(
        sport_id=tournament.sport_id,
        tournament_id=tournament.tournament_id,
        from_iso=start.isoformat().replace("+00:00", "Z"),
        to_iso=end.isoformat().replace("+00:00", "Z"),
        status_id=0,
        has_odds=True,
    )
    payloads = [
        client.get_odds(
            fixture_id=str(fixture["fixtureId"]),
            bookmakers=",".join(bookmakers),
        )
        for fixture in fixtures
    ]
    return build_game_odds_snapshot_frame(
        payloads,
        market_catalog=markets,
        source_url_prefix=f"{client.base_url}/v4/odds?fixtureId=",
        is_historical=False,
    )


def fetch_historical_game_odds_snapshots(
    *,
    start_date: date,
    end_date: date,
    api_key: str,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
    max_fixtures: Optional[int] = None,
) -> pd.DataFrame:
    client = OddsPapiClient(api_key)
    tournament = resolve_nba_tournament(client)
    markets = market_catalog_by_id(client, sport_id=tournament.sport_id)
    fixtures = client.get_fixtures(
        sport_id=tournament.sport_id,
        tournament_id=tournament.tournament_id,
        from_iso=datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        to_iso=datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        status_id=2,
        has_odds=True,
    )
    if max_fixtures is not None:
        fixtures = fixtures[:max_fixtures]

    payloads = [
        client.get_historical_odds(
            fixture_id=str(fixture["fixtureId"]),
            bookmakers=",".join(bookmakers),
        )
        for fixture in fixtures
    ]
    return build_game_odds_snapshot_frame(
        payloads,
        market_catalog=markets,
        source_url_prefix=f"{client.base_url}/v4/historical-odds?fixtureId=",
        is_historical=True,
    )
