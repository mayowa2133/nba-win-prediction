import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb


CSV_PATH = "games_all_2015_2025_features_rolling_last10_prevstats_odds_bbref.csv"


EDGE_THRESHOLDS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]


def moneyline_to_prob(ml):
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)


def odds_to_profit(ml, outcome):
    """
    Profit for 1u stake at moneyline ml, given outcome (1 = win, 0 = lose).
    """
    if outcome not in (0, 1):
        raise ValueError("Outcome must be 0 or 1")
    if outcome == 0:
        return -1.0
    ml = float(ml)
    if ml < 0:
        return 100.0 / (-ml)
    else:
        return ml / 100.0


def simulate_value_bets(
    df,
    proba_home,
    thresholds,
    market_home_ml_col,
    market_away_ml_col,
    label_col="home_win",
):
    """
    For each edge threshold, simulate flat 1u betting both sides
    whenever model edge >= threshold.
    """
    y = df[label_col].values.astype(int)
    mh = df[market_home_ml_col].values
    ma = df[market_away_ml_col].values

    implied_home = np.vectorize(moneyline_to_prob)(mh)
    implied_away = np.vectorize(moneyline_to_prob)(ma)

    edge_home = proba_home - implied_home
    edge_away = (1.0 - proba_home) - implied_away

    rows = []

    for thr in thresholds:
        bets = 0
        wins = 0
        profit = 0.0

        # We will also track edges for avg_edge on bets
        bet_edges = []

        for i in range(len(df)):
            # Home side
            if edge_home[i] >= thr:
                bets += 1
                outcome = y[i]  # 1 if home wins
                if outcome == 1:
                    wins += 1
                profit += odds_to_profit(mh[i], outcome)
                bet_edges.append(edge_home[i])

            # Away side
            if edge_away[i] >= thr:
                bets += 1
                outcome = 1 - y[i]  # 1 if away wins
                if outcome == 1:
                    wins += 1
                profit += odds_to_profit(ma[i], outcome)
                bet_edges.append(edge_away[i])

        if bets == 0:
            hitrate = np.nan
            roi = np.nan
            avg_edge = np.nan
        else:
            hitrate = wins / bets
            roi = profit / bets
            avg_edge = float(np.mean(bet_edges)) if bet_edges else np.nan

        rows.append(
            {
                "EdgeThr": thr,
                "Bets": bets,
                "HitRate": hitrate,
                "AvgEdge": avg_edge,
                "Profit": profit,
                "ROI": roi,
            }
        )

    return pd.DataFrame(rows)


def fit_xgb_with_val(X_train, y_train, X_val, y_val, verbose=False):
    """
    Small hyperparam sweep over max_depth and eta, pick best by val logloss.
    """
    param_grid = [
        {"max_depth": 3, "eta": 0.05},
        {"max_depth": 4, "eta": 0.05},
        {"max_depth": 5, "eta": 0.05},
        {"max_depth": 4, "eta": 0.10},
    ]

    best_model = None
    best_params = None
    best_logloss = float("inf")

    for params in param_grid:
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            learning_rate=params["eta"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, val_pred)
        if verbose:
            print(
                f"    max_depth={params['max_depth']}, eta={params['eta']:.3f} "
                f"-> Val LogLoss={ll:.3f}"
            )
        if ll < best_logloss:
            best_logloss = ll
            best_model = model
            best_params = params

    return best_model, best_params, best_logloss


def weighted_avg(group, weight_col, value_col):
    w = group[weight_col].values
    v = group[value_col].values
    if np.nansum(w) == 0:
        return np.nan
    return float(np.nansum(w * v) / np.nansum(w))


def aggregate_value_results(all_bets_df):
    if all_bets_df.empty:
        return all_bets_df

    grouped = (
        all_bets_df.groupby("EdgeThr")
        .apply(
            lambda g: pd.Series(
                {
                    "Bets": g["Bets"].sum(),
                    "Profit": g["Profit"].sum(),
                    "HitRate": weighted_avg(g, "Bets", "HitRate"),
                    "AvgEdge": weighted_avg(g, "Bets", "AvgEdge"),
                }
            )
        )
        .reset_index()
    )
    grouped["ROI"] = grouped["Profit"] / grouped["Bets"]
    return grouped


def run_walk_forward(
    df,
    feature_cols,
    seasons_with_odds,
    label_col="home_win",
    market_home_ml_col="home_ml",
    market_away_ml_col="away_ml",
    edge_thresholds=None,
):
    """
    Walk-forward:
      train on seasons < (test-1),
      val on (test-1),
      test on test season.
    """
    if edge_thresholds is None:
        edge_thresholds = EDGE_THRESHOLDS

    test_seasons = [s for s in seasons_with_odds if s >= 2018]

    season_summaries = []
    all_bets = []

    for test_season in test_seasons:
        val_season = test_season - 1
        train_seasons = [s for s in seasons_with_odds if s < val_season]
        if not train_seasons:
            continue

        df_train = df[df["season"].isin(train_seasons)].copy()
        df_val = df[df["season"] == val_season].copy()
        df_test = df[df["season"] == test_season].copy()

        X_train = df_train[feature_cols]
        y_train = df_train[label_col].astype(int)
        X_val = df_val[feature_cols]
        y_val = df_val[label_col].astype(int)
        X_test = df_test[feature_cols]
        y_test = df_test[label_col].astype(int)

        # tune on val
        model_val, best_params, best_ll = fit_xgb_with_val(
            X_train, y_train, X_val, y_val, verbose=False
        )

        # refit best model on train+val
        X_trainval = pd.concat([X_train, X_val], axis=0)
        y_trainval = pd.concat([y_train, y_val], axis=0)

        final_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=best_params["max_depth"],
            learning_rate=best_params["eta"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        final_model.fit(X_trainval, y_trainval)

        proba_test = final_model.predict_proba(X_test)[:, 1]
        preds_bin = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds_bin)
        ll = log_loss(y_test, proba_test)

        season_summaries.append((test_season, len(df_test), acc, ll))

        valres = simulate_value_bets(
            df_test,
            proba_test,
            thresholds=edge_thresholds,
            market_home_ml_col=market_home_ml_col,
            market_away_ml_col=market_away_ml_col,
            label_col=label_col,
        )
        valres["season"] = test_season
        all_bets.append(valres)

    summary_df = pd.DataFrame(
        season_summaries, columns=["Season", "Games", "Accuracy", "LogLoss"]
    )
    all_bets_df = (
        pd.concat(all_bets, ignore_index=True) if len(all_bets) > 0 else pd.DataFrame()
    )
    agg_df = aggregate_value_results(all_bets_df) if not all_bets_df.empty else None

    return summary_df, all_bets_df, agg_df


def print_classification_summary(label, summary_df):
    print(f"\n=== Classification summary ({label}) ===")
    print("Season   Games   Accuracy   LogLoss")
    print("------------------------------------")
    overall_games = summary_df["Games"].sum()
    overall_acc = (
        (summary_df["Accuracy"] * summary_df["Games"]).sum() / overall_games
        if overall_games > 0
        else np.nan
    )
    overall_ll = (
        (summary_df["LogLoss"] * summary_df["Games"]).sum() / overall_games
        if overall_games > 0
        else np.nan
    )

    for _, row in summary_df.iterrows():
        print(
            f" {int(row['Season'])}   {int(row['Games']):5d}   "
            f"{row['Accuracy']:.3f}     {row['LogLoss']:.3f}"
        )
    print("------------------------------------")
    print(f"Overall  {overall_games:5d}   {overall_acc:.3f}     {overall_ll:.3f}")


def print_value_summary(label, agg_df):
    print(f"\n=== Aggregated VALUE betting ({label}) ===")
    print("EdgeThr   Bets   HitRate   AvgEdge   Profit    ROI")
    print("--------------------------------------------------")
    for _, row in agg_df.iterrows():
        print(
            f"{row['EdgeThr']:7.2f}  "
            f"{int(row['Bets']):5d}   "
            f"{(row['HitRate'] if not np.isnan(row['HitRate']) else 0):7.3f}   "
            f"{(row['AvgEdge'] if not np.isnan(row['AvgEdge']) else 0):7.3f}   "
            f"{row['Profit']:7.2f}   "
            f"{row['ROI']:7.3f}"
        )


def main():
    print(f"Loading games from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} rows.")
    # Figure out market odds columns
    market_home_ml_col = "market_home_ml" if "market_home_ml" in df.columns else "home_ml"
    market_away_ml_col = "market_away_ml" if "market_away_ml" in df.columns else "away_ml"

    if "market_home_ml" not in df.columns:
        print("INFO: 'market_home_ml' not in CSV, using 'home_ml' instead.")
    if "market_away_ml" not in df.columns:
        print("INFO: 'market_away_ml' not in CSV, using 'away_ml' instead.")

    # Filter to rows with odds and seasons 2016-2022 (since odds start in 2016)
    mask_odds = df[market_home_ml_col].notna() & df[market_away_ml_col].notna()
    df = df[mask_odds & df["season"].between(2016, 2022)].copy()

    print("\nGames with odds by season:")
    print(df.groupby("season")["home_win"].count())

    seasons_with_odds = sorted(df["season"].unique())
    print(f"\nSeasons available with odds: {seasons_with_odds}")
    print(
        "We will run walk-forward value simulation on test seasons "
        f"{[s for s in seasons_with_odds if s >= 2018]}."
    )

    # Build feature sets
    # Drop obvious leak and odds columns from features
    drop_cols = [
        "game_id",
        "date",
        "home_score",
        "away_score",
        "home_win",
        "home_ml",
        "away_ml",
        "home_prob_raw",
        "away_prob_raw",
        "market_home_prob",
        "market_away_prob",
        "market_spread",
        "market_total",
    ]

    # Also drop non-numeric features for XGB (object dtype: team abbreviations, status, etc.)
    object_cols = [c for c in df.columns if df[c].dtype == "object"]

    base_feature_cols = [
        c for c in df.columns if c not in drop_cols and c not in object_cols
    ]
    bbref_cols = [c for c in base_feature_cols if c.startswith("bbref_")]
    feature_cols_with_bbref = base_feature_cols
    feature_cols_no_bbref = [c for c in base_feature_cols if not c.startswith("bbref_")]

    print(
        f"\nTotal numeric feature columns (no odds/leaks): {len(base_feature_cols)} "
        f"(including {len(bbref_cols)} BBRef talent columns)"
    )
    print(
        f"Feature count WITH BBRef: {len(feature_cols_with_bbref)}, "
        f"WITHOUT BBRef: {len(feature_cols_no_bbref)}"
    )

    # Run walk-forward with BBRef
    print("\n===== RUN 1: XGBoost WITH BBRef team talent features =====")
    summary_with, bets_with, agg_with = run_walk_forward(
        df,
        feature_cols_with_bbref,
        seasons_with_odds,
        label_col="home_win",
        market_home_ml_col=market_home_ml_col,
        market_away_ml_col=market_away_ml_col,
        edge_thresholds=EDGE_THRESHOLDS,
    )

    print_classification_summary("WITH BBRef", summary_with)
    if agg_with is not None:
        print_value_summary("WITH BBRef", agg_with)

    # Run walk-forward without BBRef
    print("\n\n===== RUN 2: XGBoost WITHOUT BBRef team talent features =====")
    summary_no, bets_no, agg_no = run_walk_forward(
        df,
        feature_cols_no_bbref,
        seasons_with_odds,
        label_col="home_win",
        market_home_ml_col=market_home_ml_col,
        market_away_ml_col=market_away_ml_col,
        edge_thresholds=EDGE_THRESHOLDS,
    )

    print_classification_summary("WITHOUT BBRef", summary_no)
    if agg_no is not None:
        print_value_summary("WITHOUT BBRef", agg_no)

    print("\nDone. Compare the two runs to see how much BBRef talent is adding.")


if __name__ == "__main__":
    main()
