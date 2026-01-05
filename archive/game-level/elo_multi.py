import pandas as pd
import math

# ---------- ELO CONFIG ----------

INITIAL_RATING = 1500.0   # starting rating for all teams
K = 20.0                  # sensitivity of rating updates
HOME_ADV = 60.0           # home court advantage in Elo points

# we'll treat everything from 2015–2025 as one continuous timeline
RECENT_SEASON_CUTOFF = 2022  # for reporting recent vs overall performance


def build_elo_predictions(csv_path: str):
    # 1. Load all games
    df = pd.read_csv(csv_path)

    # Make sure date is datetime & sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Keep only finished games (current season will have future/scheduled ones)
    if "status" in df.columns:
        df = df[df["status"] == "Final"].copy()

    # 2. Elo state
    ratings = {}  # {team_abbrev: current_elo}

    preds = []     # P(home win)
    actuals = []   # 1 if home wins else 0
    elo_home_pre = []
    elo_away_pre = []

    # 3. Loop over all games in chronological order
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # current ratings with default
        home_rating = ratings.get(home, INITIAL_RATING)
        away_rating = ratings.get(away, INITIAL_RATING)

        # store pre-game ratings
        elo_home_pre.append(home_rating)
        elo_away_pre.append(away_rating)

        # expected prob home wins
        rating_diff = (home_rating + HOME_ADV) - away_rating
        p_home = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))

        # actual result
        home_win = 1 if row["home_score"] > row["away_score"] else 0

        preds.append(p_home)
        actuals.append(home_win)

        # update ratings
        home_new = home_rating + K * (home_win - p_home)
        away_new = away_rating - K * (home_win - p_home)

        ratings[home] = home_new
        ratings[away] = away_new

    # 4. Attach predictions back to dataframe
    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_p_home"] = preds
    df["home_win"] = actuals

    # 5. Evaluate performance

    def compute_metrics(mask, label: str):
        df_sub = df[mask]
        y = df_sub["home_win"].tolist()
        p = df_sub["elo_p_home"].tolist()

        # accuracy
        yhat = [1 if prob >= 0.5 else 0 for prob in p]
        acc = sum(int(a == b) for a, b in zip(y, yhat)) / len(y)

        # log loss
        eps = 1e-15
        ll_terms = []
        for yi, pi in zip(y, p):
            pi = max(min(pi, 1 - eps), eps)
            if yi == 1:
                ll_terms.append(-math.log(pi))
            else:
                ll_terms.append(-math.log(1 - pi))
        log_loss = sum(ll_terms) / len(ll_terms)

        print(f"\n=== {label} ===")
        print(f"Games:    {len(df_sub)}")
        print(f"Accuracy: {acc:.3f}")
        print(f"LogLoss:  {log_loss:.3f}")

    # overall
    compute_metrics(mask=df["season"] >= 2015, label="Overall (2015–2025)")

    # recent seasons only (e.g. 2022+)
    compute_metrics(mask=df["season"] >= RECENT_SEASON_CUTOFF,
                    label=f"Recent (>= {RECENT_SEASON_CUTOFF})")

    return df, ratings


if __name__ == "__main__":
    df_with_elo, final_ratings = build_elo_predictions("games_all_2015_2025.csv")

    # save enriched games file
    out_path = "games_all_2015_2025_with_elo.csv"
    df_with_elo.to_csv(out_path, index=False)
    print(f"\nSaved Elo predictions to {out_path}")

    # show top 10 teams by final Elo
    final_ratings_sorted = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 final Elo ratings:")
    for team, r in final_ratings_sorted[:10]:
        print(f"{team}: {r:.1f}")
