# Archive Directory

This directory contains scripts and files that are **not actively used** by the current NBA player props prediction pipeline. They have been moved here to keep the main codebase clean for GitHub, but are preserved for reference.

## Directory Structure

### `legacy/`
Legacy/experimental scripts that were replaced by the current pipeline:
- `backfill_historical_vegas_lines.py` - Attempted historical Vegas lines backfill (not integrated)
- `derive_vegas_from_scores.py` - Synthetic Vegas lines derivation (not used)
- `join_market_lines_to_features.py` - Superseded by pipeline logic
- `log_today_props.py` - Logging now handled in pipeline
- `scan_slate.py` - Old version, replaced by `scan_slate_with_model.py`
- `build_points_model.py` - Old model building approach
- `build_points_regression_stars.py` - Star-specific model (not used)
- `tune_points_regression.py` - Hyperparameter tuning (now integrated into `build_points_regression.py`)
- `build_player_features_rolling.py` - Old feature building approach
- `build_player_game_logs_from_balldontlie.py` - Old data source
- `build_player_game_logs_from_nba_api.py` - Initial setup script (now use incremental updates)
- `build_player_positions_from_nba_api.py` - One-time setup script
- `build_prev_season_teamstats_features.py` - Old feature building
- `build_rolling_team_features.py` - Old feature building
- `build_rolling_team_features_last10.py` - Old feature building
- `build_basic_features.py` - Old feature building
- `build_bbref_team_talent.py` - Old feature building

### `analysis/`
One-off analysis and debugging scripts:
- `analyze_points_model_errors.py` - Model error analysis
- `analyze_points_residuals.py` - Residual analysis
- `analyze_market_buckets.py` - Market bucket analysis
- `analyze_model_buckets_prevstats_with_odds_scaled.py` - Model bucket analysis
- `compare_star_mae.py` - Star player MAE comparison
- `eval_single_bet.py` - Single bet evaluation
- `backtest_over_line.py` - Standalone backtesting script

### `prediction/`
Standalone prediction scripts (not part of the main pipeline):
- `predict_points_over_line.py`
- `predict_points_over_line_next_game.py`
- `predict_points_over_15_5.py`
- `predict_player_props_from_features.py`
- `predict_upcoming_value_props.py`
- `predict_upcoming_value_bets.py`
- `predict_upcoming_from_balldontlie.py`
- `predict_upcoming_from_balldontlie_prevstats.py`
- `predict_upcoming_from_snapshot.py`
- `predict_today_template.py`
- `predict_day_from_history.py`
- `star_ladder_preview.py` - Star ladder preview
- `star_ladder_vs_market.py` - Star ladder vs market comparison

### `game-level/`
Game-level prediction scripts (team/game outcomes, not player props):
- `download_games.py` - Game data download
- `download_team_season_stats.py` - Team stats download
- `merge_team_stats_into_games.py` - Team stats merging
- `merge_odds_into_games_with_market.py` - Odds merging
- `merge_bbref_talent_into_games.py` - Basketball Reference talent merging
- `scrape_bbref_player_advanced.py` - Basketball Reference scraping
- `team_state_snapshot.py` - Team state snapshots
- `elo_multi.py` - Elo rating system
- `evaluate_calibration.py` - Calibration evaluation (game-level)
- `evaluate_by_season.py` - Season-by-season evaluation (game-level)
- `evaluate_market_baseline.py` - Market baseline evaluation (game-level)

### `training/`
Legacy training scripts (old model training approaches):
- `train_final_model.py` - Final model training (old approach)
- `train_logistic_basic.py` - Basic logistic regression
- `train_logistic_prevstats.py` - Logistic with previous stats
- `train_logistic_prevstats_scaled.py` - Scaled logistic with previous stats
- `train_logistic_prevstats_with_odds.py` - Logistic with odds
- `train_logistic_prevstats_with_odds_scaled.py` - Scaled logistic with odds
- `train_logistic_rolling.py` - Rolling logistic regression
- `train_logistic_rolling_last10.py` - Rolling logistic (last 10)
- `train_logistic_teamstats.py` - Team stats logistic
- `train_xgboost_prevstats.py` - XGBoost with previous stats
- `train_xgboost_rolling.py` - Rolling XGBoost

### `simulation/`
Backtesting and simulation scripts:
- `simulate_value_bets_pure_model.py` - Pure model simulation
- `simulate_value_bets_pure_model_multiseason.py` - Multi-season pure model
- `simulate_value_bets_pure_model_bbref.py` - Pure model with Basketball Reference
- `simulate_value_bets_pure_model_bbref_multiseason.py` - Multi-season BBR pure model
- `simulate_value_bets_xgb_bbref_multiseason.py` - XGBoost BBR multi-season
- `simulate_value_bets_xgb_bbref_vs_nobbref_multiseason.py` - XGBoost BBR comparison
- `simulate_roi_high_confidence.py` - High confidence ROI simulation

### `utilities/`
One-time setup or deprecated utility scripts:
- `fetch_odds_for_date.py` - Old odds fetching approach

### `data/`
Legacy CSV data files from old game-level prediction system:
- `games_*.csv` - Historical game data files (2015-2025)
- `games_all_2015_2025*.csv` - Aggregated game data with various feature sets
- `team_stats_*.csv` - Historical team statistics (2015-2025)
- `bbref_*.csv` - Basketball Reference advanced stats and talent data
- `oddsData.csv` - Old odds data format

**Note:** The current pipeline uses `data/player_game_logs.csv` (incremental updates) and does not depend on these archived files.

## Active Pipeline Files

The following files are **actively used** by the pipeline and remain in the root directory:

**Main Pipeline:**
- `run_full_slate_pipeline.py` - Main orchestrator
- `update_player_game_logs_incremental.py` - Incremental game log updates
- `build_player_points_features.py` - Feature engineering
- `fetch_injury_data.py` - Injury data inference
- `build_minutes_regression.py` - Minutes prediction model
- `build_points_regression.py` - Points regression model (and rebounds/assists/threes)
- `build_points_regression_quantile.py` - Quantile regression models
- `build_points_regression_tiered.py` - Tiered regression models
- `build_points_sigma_model.py` - Heteroscedastic sigma model
- `build_over_prob_calibrator.py` - Probability calibrator
- `fetch_props_from_the_odds_api.py` - Live odds fetching
- `props_to_market_lines.py` - Market lines aggregation
- `scan_slate_with_model.py` - Slate scanning with models
- `star_best_bets_screener.py` - Star bets screener

**Helper Modules:**
- `load_quantile_model.py` - Quantile model loader
- `load_tiered_model.py` - Tiered model loader
- `sigma_features.py` - Sigma feature helpers
- `minutes_utils.py` - Minutes prediction utilities

**Downstream Tools:**
- `build_optimal_parlay.py` - Optimal parlay builder
- `evaluate_over_prob_holdout.py` - Holdout evaluation
- `evaluate_sigma_blend_holdout.py` - Sigma blend evaluation
- `evaluate_tiered_unified_ensemble.py` - Ensemble evaluation
- `validate_quantile_models.py` - Quantile model validation

## Notes

- These archived files are preserved for historical reference and potential future use
- They are not imported or referenced by any active pipeline scripts
- If you need to reference old approaches or restore functionality, check here first
- The archive can be safely excluded from GitHub if desired (add to `.gitignore`)

