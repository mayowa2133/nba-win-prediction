#!/usr/bin/env python
"""
predict_upcoming_value_bets.py

1. Runs your model-based upcoming-game predictor:
     python predict_upcoming_from_balldontlie_prevstats.py [YYYY-MM-DD]

2. Fetches market odds for that same date via The Odds API:
     python fetch_odds_for_date.py [YYYY-MM-DD]

3. Joins model probs with market odds, computes implied probabilities
   from moneylines, and calculates "edge" = model_prob - market_prob
   for both home and away.

4. Prints all sides sorted by edge, prints suggested value bets
   above a threshold, saves a detailed per-date CSV to:
     predictions/value_bets_YYYY-MM-DD.csv

5. Appends all suggested value bets to a master log:
     logs/value_bets_master.csv
"""

import os
import re
import sys
import subprocess
from typing import Optional, List

import pandas as pd

# --- Config -------------------------------------------------------------

PREDICTION_SCRIPT = "predict_upcoming_from_balldontlie_prevstats.py"
FETCH_ODDS_SCRIPT = "fetch_odds_for_date.py"

ODDS_DIR = "odds"
PREDICTIONS_DIR = "predictions"
LOGS_DIR = "logs"
MASTER_LOG_FILENAME = "value_bets_master.csv"

# How big an edge we require to flag a bet
VALUE_EDGE_THRESHOLD = 0.03  # 3%


# --- Team name ↔ abbrev mapping ----------------------------------------

# Keys are lowercase normalized team names, values are your abbreviations
TEAM_NAME_TO_ABBREV = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "cleveland cavs": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "gs warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "philadelphia sixers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


def normalize_team_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    return s or None


def name_to_abbrev(name: Optional[str]) -> Optional[str]:
    key = normalize_team_name(name)
    if key is None:
        return None
    return TEAM_NAME_TO_ABBREV.get(key)


# --- Helpers ------------------------------------------------------------

def american_to_prob(ml: float) -> Optional[float]:
    """
    Convert American moneyline odds to implied probability (no vig removed).
    Returns None if ml is NaN/None/0.
    """
    if ml is None:
        return None
    try:
        if pd.isna(ml):
            return None
    except TypeError:
        pass

    if ml == 0:
        return None

    if ml > 0:
        # +100 => 0.5, +200 => 0.333...
        return 100.0 / (ml + 100.0)
    else:
        # -110 => ~0.524, -200 => ~0.667
        return -ml / (-ml + 100.0)


def parse_target_date_from_stdout(stdout: str) -> Optional[str]:
    """
    Try to find 'Predicting games for YYYY-MM-DD' in predictor stdout.
    Returns the date string or None if not found.
    """
    for line in stdout.splitlines():
        m = re.search(r"Predicting games for (\d{4}-\d{2}-\d{2})", line)
        if m:
            return m.group(1)
    return None


def parse_predictions_from_stdout(stdout: str, date_str: str) -> pd.DataFrame:
    """
    Parse lines like:
      Season 2025 | BKN @ TOR | P(home win) = 0.805

    into a DataFrame with:
      date_utc, season, home_team_abbrev, away_team_abbrev, home_win_prob
    """
    rows: List[dict] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("Season"):
            continue

        # Expected format: Season 2025 | BKN @ TOR | P(home win) = 0.805
        try:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue

            # Part 0: "Season 2025"
            season_tokens = parts[0].split()
            season = int(season_tokens[1])

            # Part 1: "BKN @ TOR"
            teams_part = parts[1]
            away_abbrev, _, home_abbrev = teams_part.partition(" @ ")
            away_abbrev = away_abbrev.strip()
            home_abbrev = home_abbrev.strip()

            # Part 2: "P(home win) = 0.805"
            prob_part = parts[2]
            prob_str = prob_part.split("=")[1].strip()
            home_prob = float(prob_str)

            rows.append(
                {
                    "date_utc": date_str,
                    "season": season,
                    "home_team_abbrev": home_abbrev,
                    "away_team_abbrev": away_abbrev,
                    "home_win_prob": home_prob,
                }
            )
        except Exception:
            # If parsing one line fails, just skip it
            continue

    if not rows:
        raise RuntimeError(
            "Could not parse any prediction lines from predictor stdout. "
            "Check the output format of predict_upcoming_from_balldontlie_prevstats.py."
        )

    df = pd.DataFrame(rows)
    return df


def load_odds_csv_for_date(date_str: str) -> pd.DataFrame:
    """
    Load odds/odds_nba_YYYY-MM-DD.csv and ensure we have:
      date_utc, commence_time,
      home_team_name, away_team_name,
      home_team_abbrev, away_team_abbrev,
      home_ml, away_ml

    If the CSV only has 'home_team' / 'away_team', we derive the name and
    abbreviation columns from those.
    """
    odds_path = os.path.join(ODDS_DIR, f"odds_nba_{date_str}.csv")
    if not os.path.isfile(odds_path):
        raise FileNotFoundError(
            f"Odds CSV not found at {odds_path}. "
            "Make sure fetch_odds_for_date.py ran successfully for this date."
        )

    df = pd.read_csv(odds_path)

    # Minimal required columns that fetch_odds_for_date.py should output
    base_required = {
        "date_utc",
        "commence_time",
        "home_team",
        "away_team",
        "home_ml",
        "away_ml",
    }
    missing_base = base_required - set(df.columns)
    if missing_base:
        raise KeyError(
            f"Odds CSV is missing base required columns: {missing_base}. "
            f"Present columns: {list(df.columns)}"
        )

    # Create name columns if not already present
    if "home_team_name" not in df.columns:
        df["home_team_name"] = df["home_team"]
    if "away_team_name" not in df.columns:
        df["away_team_name"] = df["away_team"]

    # Create abbrev columns if not already present
    if "home_team_abbrev" not in df.columns:
        df["home_team_abbrev"] = df["home_team_name"].apply(name_to_abbrev)
    if "away_team_abbrev" not in df.columns:
        df["away_team_abbrev"] = df["away_team_name"].apply(name_to_abbrev)

    # Warn if any team names couldn't be mapped
    missing_home = df[df["home_team_abbrev"].isna()]["home_team_name"].unique()
    missing_away = df[df["away_team_abbrev"].isna()]["away_team_name"].unique()

    missing_names = set(missing_home) | set(missing_away)
    missing_names = {m for m in missing_names if isinstance(m, str)}
    if missing_names:
        print(
            "WARNING: Could not map some team names to abbreviations in odds CSV:\n"
            + "\n".join(f"  - {name}" for name in missing_names)
        )

    return df


def build_sides_table(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Join predictions + odds on (date_utc, home_team_abbrev, away_team_abbrev),
    then expand each game into two rows (home side & away side) with:
      - team_abbrev, team_name
      - opponent_abbrev, opponent_name
      - moneyline
      - model_prob
      - market_prob (implied from ML)
      - edge = model_prob - market_prob
    """
    merged = preds.merge(
        odds,
        on=["date_utc", "home_team_abbrev", "away_team_abbrev"],
        how="inner",
        suffixes=("", "_odds"),
    )

    if merged.empty:
        raise RuntimeError(
            "No overlap between predictions and odds after merge. "
            "Check team abbreviations or the mapping between odds team names and your abbreviations."
        )

    rows = []
    for _, row in merged.iterrows():
        date_utc = row["date_utc"]
        commence_time = row.get("commence_time", "")

        home_abbrev = row["home_team_abbrev"]
        away_abbrev = row["away_team_abbrev"]
        home_name = row["home_team_name"]
        away_name = row["away_team_name"]

        home_ml = row["home_ml"]
        away_ml = row["away_ml"]

        home_model_prob = row["home_win_prob"]
        away_model_prob = 1.0 - home_model_prob

        # Home side
        home_mark_p = american_to_prob(home_ml)
        if home_mark_p is not None:
            rows.append(
                {
                    "date_utc": date_utc,
                    "commence_time": commence_time,
                    "side": "home",
                    "team_abbrev": home_abbrev,
                    "team_name": home_name,
                    "opponent_abbrev": away_abbrev,
                    "opponent_name": away_name,
                    "moneyline": home_ml,
                    "model_prob": home_model_prob,
                    "market_prob": home_mark_p,
                }
            )

        # Away side
        away_mark_p = american_to_prob(away_ml)
        if away_mark_p is not None:
            rows.append(
                {
                    "date_utc": date_utc,
                    "commence_time": commence_time,
                    "side": "away",
                    "team_abbrev": away_abbrev,
                    "team_name": away_name,
                    "opponent_abbrev": home_abbrev,
                    "opponent_name": home_name,
                    "moneyline": away_ml,
                    "model_prob": away_model_prob,
                    "market_prob": away_mark_p,
                }
            )

    if not rows:
        raise RuntimeError(
            "After filtering for valid moneylines, no sides remained."
        )

    sides_df = pd.DataFrame(rows)
    sides_df["edge"] = sides_df["model_prob"] - sides_df["market_prob"]
    return sides_df


def append_to_master_log(suggested: pd.DataFrame) -> None:
    """
    Append suggested bets to logs/value_bets_master.csv.
    Creates directory/file if needed.
    """
    if suggested.empty:
        return

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, MASTER_LOG_FILENAME)

    cols = [
        "date_utc",
        "commence_time",
        "side",
        "team_abbrev",
        "team_name",
        "opponent_abbrev",
        "opponent_name",
        "moneyline",
        "model_prob",
        "market_prob",
        "edge",
    ]
    suggested_to_write = suggested[cols]

    if not os.path.exists(log_path):
        suggested_to_write.to_csv(log_path, index=False)
    else:
        suggested_to_write.to_csv(log_path, mode="a", header=False, index=False)


# --- Main ---------------------------------------------------------------

def main(argv: List[str]) -> None:
    # 1) Determine target date from CLI or from predictor stdout
    cli_date: Optional[str] = argv[1] if len(argv) > 1 else None

    if cli_date:
        print(f"\n=== Value bet scan for {cli_date} ===\n")
    else:
        print("\n=== Value bet scan (auto date from predictor) ===\n")

    # 2) Run prediction script
    if cli_date:
        pred_cmd = [sys.executable, PREDICTION_SCRIPT, cli_date]
    else:
        pred_cmd = [sys.executable, PREDICTION_SCRIPT]

    print("Running:", " ".join(pred_cmd))
    pred_res = subprocess.run(
        pred_cmd, capture_output=True, text=True, check=True
    )
    pred_stdout = pred_res.stdout

    # Derive actual date_str
    if cli_date:
        date_str = cli_date
    else:
        date_str = parse_target_date_from_stdout(pred_stdout)
        if not date_str:
            raise RuntimeError(
                "Could not infer date from predictor stdout. "
                "Run with an explicit YYYY-MM-DD argument."
            )

    # Parse predictions into DataFrame
    preds_df = parse_predictions_from_stdout(pred_stdout, date_str)
    print("\nParsed predictions:")
    print(preds_df.to_string(index=False))

    # 3) Run odds fetch script for same date
    print(f"\nRunning: {sys.executable} {FETCH_ODDS_SCRIPT} {date_str}")
    odds_cmd = [sys.executable, FETCH_ODDS_SCRIPT, date_str]
    odds_res = subprocess.run(
        odds_cmd, capture_output=True, text=True, check=True
    )
    odds_stdout = odds_res.stdout
    if odds_stdout:
        # You can uncomment this if you want to debug odds fetch output:
        # print(odds_stdout)
        pass

    # 4) Load odds CSV and build side-level table
    odds_df = load_odds_csv_for_date(date_str)
    sides_df = build_sides_table(preds_df, odds_df)

    # Sort by edge descending
    sides_sorted = sides_df.sort_values("edge", ascending=False).reset_index(
        drop=True
    )

    print("\nAll sides sorted by edge (model_prob - market_prob):\n")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.precision", 3,
    ):
        print(sides_sorted.to_string(index=False))

    # 5) Suggested value bets above threshold
    suggested = sides_sorted[sides_sorted["edge"] >= VALUE_EDGE_THRESHOLD].copy()

    print(
        f"\nSuggested value bets (edge ≥ {VALUE_EDGE_THRESHOLD:.0%}) "
        f"for {date_str} (1-unit flat stakes conceptually):\n"
    )
    if suggested.empty:
        print("No bets meet the edge threshold.")
    else:
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 200,
            "display.precision", 3,
        ):
            print(suggested.to_string(index=False))

    # 6) Save per-day detailed CSV
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    daily_path = os.path.join(
        PREDICTIONS_DIR, f"value_bets_{date_str}.csv"
    )
    sides_sorted.to_csv(daily_path, index=False)
    print(f"\nSaved detailed bets table to {daily_path}")

    # 7) Append suggested bets to master log
    append_to_master_log(suggested)
    print(
        f"Appended {len(suggested)} suggested bets "
        f"to {os.path.join(LOGS_DIR, MASTER_LOG_FILENAME)}\n"
    )


if __name__ == "__main__":
    main(sys.argv)
