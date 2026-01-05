# NBA Win Prediction - Player Props Pipeline

A machine learning pipeline for predicting NBA player prop bets (points, rebounds, assists, 3-pointers made) and identifying betting edges.

## Project Structure

```
nba-win-prediction/
├── src/                          # Source code
│   ├── pipeline/                 # Main pipeline orchestrator
│   │   └── run_full_slate_pipeline.py
│   ├── data/                     # Data processing scripts
│   │   ├── build_player_points_features.py
│   │   ├── fetch_injury_data.py
│   │   ├── fetch_props_from_the_odds_api.py
│   │   └── props_to_market_lines.py
│   ├── models/                   # Model training scripts
│   │   ├── build_minutes_regression.py
│   │   ├── build_points_regression.py
│   │   ├── build_points_regression_quantile.py
│   │   ├── build_points_regression_tiered.py
│   │   ├── build_points_sigma_model.py
│   │   └── build_over_prob_calibrator.py
│   ├── inference/                # Inference and evaluation scripts
│   │   ├── scan_slate_with_model.py
│   │   ├── star_best_bets_screener.py
│   │   └── build_optimal_parlay.py
│   ├── evaluation/               # Model evaluation scripts
│   │   ├── evaluate_over_prob_holdout.py
│   │   ├── evaluate_sigma_blend_holdout.py
│   │   ├── evaluate_tiered_unified_ensemble.py
│   │   └── validate_quantile_models.py
│   └── utils/                    # Helper modules
│       ├── load_quantile_model.py
│       ├── load_tiered_model.py
│       ├── sigma_features.py
│       └── minutes_utils.py
├── data/                         # Data files (CSVs, logs)
├── models/                       # Trained model files (.pkl)
├── archive/                      # Archived/unused scripts
├── run_pipeline.py               # Convenience wrapper script
└── README.md                     # This file
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key (for The Odds API)
export ODDS_API_KEY="your_api_key_here"
```

### 2. Run Full Pipeline

```bash
# From project root
python run_pipeline.py

# Or directly
python src/pipeline/run_full_slate_pipeline.py
```

The pipeline will:
1. Update player game logs incrementally
2. Rebuild player features
3. Train/retrain all models
4. Fetch fresh odds from The Odds API
5. Scan the slate and compute betting edges
6. Generate star player recommendations

### 3. Individual Scripts

You can also run individual scripts:

```bash
# Update game logs
python src/pipeline/update_player_game_logs_incremental.py

# Build features
python src/data/build_player_points_features.py

# Train models
python src/models/build_points_regression.py

# Scan slate
python src/inference/scan_slate_with_model.py --market-lines data/market_lines.csv --output data/edges.csv

# Build parlays
python src/inference/build_optimal_parlay.py --edges-csv data/edges_with_market.csv
```

## Key Features

- **Multi-market support**: Points, rebounds, assists, 3-pointers made
- **Advanced modeling**: Tiered models, quantile regression, heteroscedastic variance, probability calibration
- **Feature engineering**: Rolling stats, Vegas lines, injury data, lineup context, player vs opponent history
- **Edge detection**: Identifies betting edges by comparing model predictions to market odds
- **Parlay optimization**: Builds optimal multi-leg parlays with correlation constraints

## Data Files

- `data/player_game_logs.csv` - Historical player game logs
- `data/player_points_features.csv` - Engineered features
- `data/odds_slate.csv` - Latest prop odds from The Odds API
- `data/market_lines.csv` - Aggregated market lines
- `data/edges_with_market.csv` - Computed betting edges

## Model Files

- `models/points_regression.pkl` - Main points regression model
- `models/rebounds_regression.pkl` - Rebounds model
- `models/assists_regression.pkl` - Assists model
- `models/threes_regression.pkl` - 3-pointers model
- `models/minutes_regression.pkl` - Minutes prediction model
- `models/points_sigma_model.pkl` - Heteroscedastic variance model
- `models/over_prob_calibrator.pkl` - Probability calibrator

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

