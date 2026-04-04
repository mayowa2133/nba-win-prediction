PYTHON ?= .venv/bin/python

.PHONY: install install-dev test api score-slate warehouse-init materialize-recommendations freeze-lock ingest-official-injuries build-starter-history build-lineup-projections ingest-game-odds ingest-props import-historical-odds backfill-historical-market-data backfill-game-odds train-game-models score-game-markets settle-recommendations build-market-readiness run-pipeline pipeline-daily pipeline-bootstrap pipeline-backfill replay-historical

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn src.api.app:app --reload

run-pipeline:
	$(PYTHON) run_pipeline.py

pipeline-daily:
	$(PYTHON) run_pipeline.py --mode daily

pipeline-bootstrap:
	$(PYTHON) run_pipeline.py --mode bootstrap --publish-current-day-at-end

pipeline-backfill:
	$(PYTHON) run_pipeline.py --mode backfill-only

score-slate:
	$(PYTHON) src/inference/scan_slate_with_model.py

warehouse-init:
	$(PYTHON) src/jobs/init_warehouse.py

materialize-recommendations:
	$(PYTHON) src/jobs/materialize_recommendations.py

ingest-official-injuries:
	$(PYTHON) src/jobs/ingest_official_injuries.py

build-starter-history:
	$(PYTHON) src/jobs/build_starter_history.py

build-lineup-projections:
	$(PYTHON) src/jobs/build_lineup_projections.py

ingest-game-odds:
	$(PYTHON) src/jobs/ingest_game_odds.py

ingest-props:
	$(PYTHON) src/jobs/ingest_props.py

import-historical-odds:
	$(PYTHON) src/jobs/import_historical_game_odds.py

backfill-historical-market-data:
	$(PYTHON) src/jobs/backfill_historical_market_data.py

backfill-game-odds:
	$(PYTHON) src/jobs/backfill_game_odds_history.py

train-game-models:
	$(PYTHON) src/jobs/train_game_market_models.py

score-game-markets:
	$(PYTHON) src/jobs/score_game_markets.py

settle-recommendations:
	$(PYTHON) src/jobs/settle_recommendations.py

build-market-readiness:
	$(PYTHON) src/jobs/build_market_readiness_snapshot.py

replay-historical:
	$(PYTHON) src/jobs/replay_historical_recommendations.py

freeze-lock:
	$(PYTHON) -m pip freeze > requirements.lock.txt
