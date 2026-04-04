# NBA Betting Beta

This repository is now structured as a reproducible research-and-product bridge for an iOS-first NBA betting beta. The current production candidate remains `player_points`; rebounds, assists, and threes are supported in the scoring path but still marked `experimental` until they clear real readiness gates.

## Current State

- Python runtime is standardized on `3.11.6` via [`.python-version`](.python-version).
- Dependencies are managed through [`pyproject.toml`](pyproject.toml), not an ad hoc `requirements.txt`.
- A reproducible environment snapshot is checked in as [`requirements.lock.txt`](requirements.lock.txt).
- The inference path now injects the live slate's current prop line and odds into model features instead of relying on stale historical joins.
- The main pipeline retrains the live-scored prop markets and materializes scored recommendations into a local warehouse for the beta API.
- The API serves precomputed recommendation artifacts from a database when available, with CSV fallback for local development.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

Environment variables:

- `ODDS_API_KEY`: optional for live current-day odds ingestion. Required for `daily` mode and current prop/game snapshot collection, but not for historical bootstrap/backfill.
- `NBA_BETTING_DATABASE_URL`: optional SQLAlchemy URL. Defaults to `sqlite:///data/nba_betting_beta.db`.
- `NBA_BETTING_EDGES_PATH`: optional override for the precomputed recommendation CSV path.

## Common Commands

```bash
make install-dev
make test
make score-slate
make ingest-official-injuries
make build-starter-history
make build-lineup-projections
make ingest-game-odds
make import-historical-odds
make backfill-historical-market-data
make train-game-models
make score-game-markets
make settle-recommendations
make build-market-readiness
make pipeline-daily
make pipeline-bootstrap
make pipeline-backfill
make replay-historical
make warehouse-init
make materialize-recommendations
make api
```

Useful direct entrypoints:

```bash
python run_pipeline.py --mode daily
python run_pipeline.py --mode bootstrap --publish-current-day-at-end
python run_pipeline.py --mode backfill-only --backfill-start-date 2025-10-01 --backfill-end-date 2026-03-30
python src/pipeline/run_full_slate_pipeline.py
python src/jobs/materialize_recommendations.py --edges-path data/edges_with_market.csv
python src/jobs/ingest_official_injuries.py --report-date 2026-03-31
python src/jobs/build_starter_history.py --max-games 50
python src/jobs/build_lineup_projections.py --target-date 2026-03-31
python src/jobs/ingest_game_odds.py --report-date 2026-03-31
python src/jobs/import_historical_game_odds.py --manifest data/historical_odds/source_manifest.json
python src/jobs/backfill_historical_market_data.py --canonical-odds-csv data/historical_odds/canonical_historical_odds.csv
python src/jobs/backfill_game_odds_history.py
python src/jobs/train_game_market_models.py
python src/jobs/score_game_markets.py --target-date 2026-03-31
python src/jobs/replay_historical_recommendations.py --start-date 2025-10-01 --end-date 2026-03-30
python src/jobs/settle_recommendations.py --database-url sqlite:///data/nba_betting_beta.db
python src/jobs/build_market_readiness_snapshot.py --database-url sqlite:///data/nba_betting_beta.db
python -m uvicorn src.api.app:app --reload
```

## Pipeline Output

The orchestration CLI now supports three operating modes:

1. `daily`
   - updates logs and features
   - ingests same-day official injuries
   - refreshes starter history and projected lineups
   - ingests current game odds and current props
   - scores props and game markets
   - materializes live recommendations
   - settles prior recommendations and rebuilds readiness
2. `bootstrap`
   - initializes the warehouse
   - imports local historical odds sources and backfills historical market artifacts for the last two seasons by default
   - rebuilds starter history
   - retrains prop and game-market models
   - runs monthly walk-forward historical replay tagged as `historical_replay`
   - settles replay rows and rebuilds readiness
   - can optionally publish the current day at the end
3. `backfill-only`
   - resumes injury backfill, historical odds reconciliation/backfill, and historical replay
   - settles replay rows and rebuilds readiness
   - never publishes current-day recommendations

Runtime state is persisted under `data/pipeline_state/`:

- `injury_backfill_cursor.json`
- `historical_replay_cursor.json`

Historical odds are local-first. The canonical import lives under `data/historical_odds/` and can be driven by `data/historical_odds/source_manifest.json`, with a fallback to `data/historical_vegas_lines.csv` if no manifest is present. Live API feeds default to `live_daily` recommendations only. Historical replay rows are persisted for readiness/bootstrap purposes and are excluded from the mobile-facing feed unless explicitly queried.

Primary artifacts:

- `data/edges_with_market.csv`: scored recommendations with schema/version metadata.
- `data/nba_betting_beta.db`: default local warehouse for the API.
- `data/historical_odds/canonical_historical_odds.csv`: reconciled local historical odds table used for backfill and replay.
- `data/historical_odds/historical_odds_conflicts.csv`: unresolved or conflicting historical odds candidates for manual review.
- `models/*.pkl`: model bundles with embedded metadata about target, training window, and readiness status.
- `data/run_logs/pipeline_<date>.json`: structured per-run execution logs with step status, warnings, and counters.

## Beta API

The FastAPI service is defined in [`src/api/app.py`](src/api/app.py).

Endpoints:

- `GET /v1/recommendations`
- `GET /v1/recommendations/{id}`
- `GET /v1/games/{game_id}`
- `GET /v1/slates/{date}`
- `GET /v1/markets/readiness`

Recommendation responses include:

- `id`
- `game_id`
- `market`
- `selection`
- `sportsbook_line`
- `sportsbook_odds`
- `fair_line`
- `fair_odds`
- `edge`
- `confidence`
- `status`
- `model_version`
- `data_timestamp`
- `reasons[]`

## Warehouse Scope

The local warehouse schema lives under [`src/warehouse`](src/warehouse) and currently includes tables for:

- player logs
- team/game context
- injury reports
- game odds snapshots
- closing lines
- starter history
- lineup projections
- scored recommendations
- settled bet outcomes
- market readiness snapshots

This is the persistence layer the current beta API can grow into. It is compatible with local SQLite out of the box and intended to move to Postgres in deployment environments through `NBA_BETTING_DATABASE_URL`.

## Testing And CI

- Unit and regression tests live in [`tests`](tests).
- CI is defined in [`ci.yml`](.github/workflows/ci.yml).
- The current suite covers import smoke, target-column regression, official injury parsing, lineup projection replacement logic, odds normalization/closing lines, game-market scoring, settlement/readiness logic, current prop override behavior, a frozen-slate scoring fixture, API contract behavior, and database-backed recommendation loading.

## Known Gaps

- Historical bootstrap and replay no longer depend on an external historical odds API, but they still depend on the quality and coverage of the locally downloaded historical datasets you import into `data/historical_odds/`.
- Live game-market publishing and true CLV evidence still depend on fresh current-day snapshots from The Odds API and the locally accumulated sample size from those snapshots.
- Game-market models currently use free historical logs plus market consensus features; they still need larger historical odds coverage and more settled live sample before readiness can promote them to `production`.
- No iOS client is in this repository yet; the current work is the backend and data-contract foundation that the SwiftUI client will consume.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
