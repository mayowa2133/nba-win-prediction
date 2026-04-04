#!/usr/bin/env python
"""
build_player_points_features.py

Create rolling features for player points modeling from per-game logs.

Takes:
  data/player_game_logs.csv  (from build_player_game_logs_from_nba_api.py)

Produces:
  data/player_points_features.csv

New (2025-11):
  - Player baseline features:
      * player_pts_career_mean
      * player_pts_season_mean
      * player_minutes_career_mean
      * player_minutes_season_mean
  - Form / trend features:
      * pts_trend_5_15
      * minutes_trend_5_15
      * fga_trend_5_15
  - Volatility features:
      * pts_std5
      * minutes_std5
      * fga_std5
  - Usage-style ratios (roll5):
      * pts_per_min_roll5
      * fga_per_min_roll5
      * fta_per_min_roll5
  - Rest flags:
      * is_b2b
      * is_long_rest
  - Environment / matchup:
      * opp_pts_allowed_roll5 / roll15
      * team_pace_roll5 / roll15 (estimated possessions)
      * team_margin_roll5 / roll15 (recent team scoring margin)
      * opp_dvp_pos_pts_roll5 / roll15 (points allowed vs position, if positions available)
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_CSV = Path("data") / "player_game_logs.csv"
OUT_CSV = Path("data") / "player_points_features.csv"
PLAYER_POSITIONS_CSV = Path("data") / "player_positions.csv"

ROLL_SHORT = 5
ROLL_LONG = 15


def main() -> None:
    print(f"Loading raw player logs from {IN_CSV} ...")
    df = pd.read_csv(IN_CSV)

    # Ensure date is datetime
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ------------------------------------------------------------------
    # Optionally merge in player positions (for DvP by position)
    # ------------------------------------------------------------------
    if PLAYER_POSITIONS_CSV.exists():
        print(f"Merging player positions from {PLAYER_POSITIONS_CSV} ...")
        df_pos = pd.read_csv(PLAYER_POSITIONS_CSV)

        # Expect at least: player_id, position
        if "player_id" in df_pos.columns and "position" in df_pos.columns:
            df_pos_small = (
                df_pos[["player_id", "position"]]
                .drop_duplicates(subset=["player_id"])
                .rename(columns={"position": "player_position"})
            )
            df = df.merge(df_pos_small, on="player_id", how="left", validate="many_to_one")
        else:
            print(
                "WARNING: player_positions.csv missing expected columns "
                "['player_id', 'position']; skipping position merge."
            )
    else:
        print(
            "No player_positions.csv found; you can build it with "
            "build_player_positions_from_nba_api.py to enable DvP-vs-position features."
        )

    # ------------------------------------------------------------------
    # 1) Days since last game (rest) per player
    # ------------------------------------------------------------------
    df = df.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["days_since_last_game"] = (df["game_date"] - df["prev_game_date"]).dt.days

    # ------------------------------------------------------------------
    # 2) Opponent defensive strength: points allowed (reconstructed)
    # ------------------------------------------------------------------
    # We keep multiple targets at the end (points, rebounds, assists, 3PM),
    # even though points is the primary modeled stat today.
    required_cols = ["season", "game_id", "game_date", "team_abbrev", "opp_abbrev", "pts", "reb", "ast", "fg3m"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed to build opponent features: {missing}")

    # Aggregate to team-level scores per game
    team_game_totals = (
        df.groupby(
            ["season", "game_id", "game_date", "team_abbrev", "opp_abbrev"],
            as_index=False,
        )["pts"]
        .sum()
        .rename(columns={"pts": "team_pts"})
    )

    # Build a table with opponent team points for each game
    opp_totals = (
        team_game_totals[["season", "game_id", "team_abbrev", "team_pts"]]
        .rename(columns={"team_abbrev": "opp_abbrev", "team_pts": "opp_pts"})
    )

    # Merge so each row has team_pts (their own score) + opp_pts (points they allowed)
    team_games = team_game_totals.merge(
        opp_totals,
        on=["season", "game_id", "opp_abbrev"],
        how="left",
        validate="many_to_one",
    )

    # For defensive strength, rolling avg of opp_pts (shifted so it's only prior games).
    def _rolling_allowed(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window=window, min_periods=1).mean()

    group_team_season = team_games.groupby(["season", "team_abbrev"], group_keys=False)
    team_games["opp_pts_allowed_roll5"] = group_team_season["opp_pts"].apply(
        lambda s: _rolling_allowed(s, ROLL_SHORT)
    )
    team_games["opp_pts_allowed_roll15"] = group_team_season["opp_pts"].apply(
        lambda s: _rolling_allowed(s, ROLL_LONG)
    )

    # ------------------------------------------------------------------
    # 2a) Team margin features (team_pts - opp_pts, rolling)
    # ------------------------------------------------------------------
    team_games["team_margin"] = team_games["team_pts"] - team_games["opp_pts"]
    team_games["team_margin_roll5"] = group_team_season["team_margin"].apply(
        lambda s: _rolling_allowed(s, ROLL_SHORT)
    )
    team_games["team_margin_roll15"] = group_team_season["team_margin"].apply(
        lambda s: _rolling_allowed(s, ROLL_LONG)
    )

    # Keep just the features we need to merge back to player logs
    team_feats = team_games[
        [
            "season",
            "game_id",
            "team_abbrev",
            "opp_pts_allowed_roll5",
            "opp_pts_allowed_roll15",
            "team_margin_roll5",
            "team_margin_roll15",
        ]
    ]

    df = df.merge(
        team_feats,
        on=["season", "game_id", "team_abbrev"],
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # 2b) Team pace features (estimated possessions) per team/game
    # ------------------------------------------------------------------
    # Need FGA, FTA, OREB, TOV to get a crude possessions estimate.
    pace_required = ["fga", "fta", "oreb", "tov"]
    if all(c in df.columns for c in pace_required):
        print("Building team pace features (estimated possessions)...")

        # Estimate team possessions at team-game level
        team_pace_df = (
            df.groupby(
                ["season", "game_id", "game_date", "team_abbrev", "opp_abbrev"],
                as_index=False,
            )
            .agg(
                fga=("fga", "sum"),
                fta=("fta", "sum"),
                oreb=("oreb", "sum"),
                tov=("tov", "sum"),
            )
        )

        # Simple Bball-Ref style possession estimate:
        # poss ≈ FGA + 0.44*FTA - OREB + TOV
        team_pace_df["team_possessions_est"] = (
            team_pace_df["fga"]
            + 0.44 * team_pace_df["fta"]
            - team_pace_df["oreb"]
            + team_pace_df["tov"]
        )

        # Avoid divide-by-zero downstream
        team_pace_df["team_possessions_est"].replace(0, np.nan, inplace=True)

        group_team_season_pace = team_pace_df.groupby(
            ["season", "team_abbrev"], group_keys=False
        )
        team_pace_df["team_pace_roll5"] = group_team_season_pace["team_possessions_est"].apply(
            lambda s: s.shift(1).rolling(ROLL_SHORT, min_periods=1).mean()
        )
        team_pace_df["team_pace_roll15"] = group_team_season_pace[
            "team_possessions_est"
        ].apply(
            lambda s: s.shift(1).rolling(ROLL_LONG, min_periods=1).mean()
        )

        team_pace_feats = team_pace_df[
            [
                "season",
                "game_id",
                "team_abbrev",
                "team_pace_roll5",
                "team_pace_roll15",
            ]
        ]

        df = df.merge(
            team_pace_feats,
            on=["season", "game_id", "team_abbrev"],
            how="left",
            validate="many_to_one",
        )
    else:
        print(
            f"WARNING: Missing one or more of {pace_required}; skipping pace features."
        )

    # ------------------------------------------------------------------
    # 2c) Usage events per game (FGA + 0.44*FTA [+ TOV])
    # ------------------------------------------------------------------
    usage_required = ["fga", "fta"]
    if all(c in df.columns for c in usage_required):
        print("Building usage events feature (usg_events)...")
        has_tov = "tov" in df.columns
        if has_tov:
            df["usg_events"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
        else:
            df["usg_events"] = df["fga"] + 0.44 * df["fta"]
    else:
        print(
            f"WARNING: Missing one or more of {usage_required}; "
            "skipping usage events feature."
        )

    # ------------------------------------------------------------------
    # 3) Rolling player stats + new advanced features
    # ------------------------------------------------------------------
    # Sort consistently for all rolling operations
    df = df.sort_values(["season", "player_id", "game_date"]).reset_index(drop=True)

    roll_stats = ["minutes", "pts", "reb", "ast", "fg3m", "fg3a", "fga", "fta"]
    if "usg_events" in df.columns:
        roll_stats.append("usg_events")

    by = ["season", "player_id"]

    # Classic rolling means (shifted so current game isn't included)
    for stat in roll_stats:
        short_col = f"{stat}_roll{ROLL_SHORT}"
        long_col = f"{stat}_roll{ROLL_LONG}"

        df[short_col] = (
            df.groupby(by)[stat]
            .transform(lambda s: s.shift(1).rolling(window=ROLL_SHORT, min_periods=1).mean())
        )
        df[long_col] = (
            df.groupby(by)[stat]
            .transform(lambda s: s.shift(1).rolling(window=ROLL_LONG, min_periods=1).mean())
        )

    # ------------------------------------------------------------------
    # 3a) Player baselines (career + per-season means up to before this game)
    # ------------------------------------------------------------------
    group_player = df.groupby("player_id", group_keys=False)
    df["player_pts_career_mean"] = group_player["pts"].apply(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )
    df["player_minutes_career_mean"] = group_player["minutes"].apply(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )

    group_player_season = df.groupby(["player_id", "season"], group_keys=False)
    df["player_pts_season_mean"] = group_player_season["pts"].apply(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )
    df["player_minutes_season_mean"] = group_player_season["minutes"].apply(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )

    # ------------------------------------------------------------------
    # 3ab) PHASE 4A: Player vs Opponent History
    # ------------------------------------------------------------------
    print("Building player vs opponent history features...")
    # Sort by player, opponent, game_date for rolling calculations
    df = df.sort_values(["player_id", "opp_abbrev", "game_date"]).reset_index(drop=True)
    
    # Group by player and opponent to calculate head-to-head stats
    group_player_opp = df.groupby(["player_id", "opp_abbrev"], group_keys=False)
    
    # Career average vs this opponent (all previous games vs this team)
    df["player_vs_opp_pts_avg_career"] = group_player_opp["pts"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_vs_opp_minutes_avg_career"] = group_player_opp["minutes"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    
    # Last 5 games vs this opponent
    df["player_vs_opp_pts_avg_last_5"] = group_player_opp["pts"].apply(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["player_vs_opp_minutes_avg_last_5"] = group_player_opp["minutes"].apply(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Count of games vs this opponent (career)
    df["player_vs_opp_games_count"] = group_player_opp["game_id"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )
    
    # Fill NaN for players who haven't faced this opponent yet
    df["player_vs_opp_pts_avg_career"] = df["player_vs_opp_pts_avg_career"].fillna(0.0)
    df["player_vs_opp_minutes_avg_career"] = df["player_vs_opp_minutes_avg_career"].fillna(0.0)
    df["player_vs_opp_pts_avg_last_5"] = df["player_vs_opp_pts_avg_last_5"].fillna(0.0)
    df["player_vs_opp_minutes_avg_last_5"] = df["player_vs_opp_minutes_avg_last_5"].fillna(0.0)
    df["player_vs_opp_games_count"] = df["player_vs_opp_games_count"].fillna(0.0)
    
    # Re-sort by season, player, game_date for remaining features
    df = df.sort_values(["season", "player_id", "game_date"]).reset_index(drop=True)
    
    print("[INFO] Added player vs opponent history features")

    # ------------------------------------------------------------------
    # 3aa) Extra features: role vs career + star tiers
    # ------------------------------------------------------------------
    extra_required = [
        "minutes_roll5",
        "pts_roll5",
        "player_pts_career_mean",
        "player_minutes_career_mean",
    ]
    extra_missing = [c for c in extra_required if c not in df.columns]
    if extra_missing:
        print(f"[WARN] Skipping extra star/role features; missing columns: {extra_missing}")
    else:
        # Relative minutes vs career baseline
        df["rel_minutes_vs_career"] = (
            df["minutes_roll5"] - df["player_minutes_career_mean"]
        )

        # Relative points vs career baseline
        df["rel_pts_vs_career"] = (
            df["pts_roll5"] - df["player_pts_career_mean"]
        )

        # Star tier by career scoring
        # 0 = low-usage / bench
        # 1 = rotation scorer
        # 2 = primary / secondary option
        # 3 = star / elite scorer
        def _bucket_star_tier_pts(pts: float) -> int:
            if pd.isna(pts):
                return 0
            if pts < 8:
                return 0
            elif pts < 15:
                return 1
            elif pts < 22:
                return 2
            else:
                return 3

        df["star_tier_pts"] = df["player_pts_career_mean"].apply(_bucket_star_tier_pts)

        # Star tier by career minutes
        # 0 = deep bench
        # 1 = rotation (15–24 min)
        # 2 = strong starter (24–30)
        # 3 = heavy-minutes (30+)
        def _bucket_star_tier_minutes(m: float) -> int:
            if pd.isna(m):
                return 0
            if m < 15:
                return 0
            elif m < 24:
                return 1
            elif m < 30:
                return 2
            else:
                return 3

        df["star_tier_minutes"] = df["player_minutes_career_mean"].apply(
            _bucket_star_tier_minutes
        )

        print(
            "[INFO] Added extra features: rel_minutes_vs_career, rel_pts_vs_career, "
            "star_tier_pts, star_tier_minutes"
        )

    # ------------------------------------------------------------------
    # 3b) Trend features: short vs long window
    # ------------------------------------------------------------------
    df["pts_trend_5_15"] = df["pts_roll5"] - df["pts_roll15"]
    df["minutes_trend_5_15"] = df["minutes_roll5"] - df["minutes_roll15"]
    df["fga_trend_5_15"] = df["fga_roll5"] - df["fga_roll15"]

    # ------------------------------------------------------------------
    # 3c) Volatility features (std over last 5 games, per season+player)
    # ------------------------------------------------------------------
    for stat in ["pts", "minutes", "fga"]:
        df[f"{stat}_std5"] = (
            df.groupby(by)[stat]
            .transform(lambda s: s.shift(1).rolling(window=ROLL_SHORT, min_periods=3).std())
        )

    # ------------------------------------------------------------------
    # 3d) Usage-style ratios using roll5
    # ------------------------------------------------------------------
    eps = 1e-3
    df["pts_per_min_roll5"] = df["pts_roll5"] / (df["minutes_roll5"] + eps)
    df["fga_per_min_roll5"] = df["fga_roll5"] / (df["minutes_roll5"] + eps)
    df["fta_per_min_roll5"] = df["fta_roll5"] / (df["minutes_roll5"] + eps)

    # ------------------------------------------------------------------
    # 3e) Rest flags
    # ------------------------------------------------------------------
    df["is_b2b"] = (df["days_since_last_game"] <= 1).astype("Int64")
    df["is_long_rest"] = (df["days_since_last_game"] >= 3).astype("Int64")

    # ------------------------------------------------------------------
    # 3f) DvP by position: how many points does the OPPONENT allow
    #     to this position, on a rolling basis?
    # ------------------------------------------------------------------
    # We need a position column on the *offensive* player rows.
    pos_col = None
    for cand in ["player_position", "position", "pos"]:
        if cand in df.columns:
            pos_col = cand
            break

    if pos_col is None:
        print(
            "WARNING: No player position column found (expected one of "
            "['player_position', 'position', 'pos']); skipping DvP-vs-position features."
        )
    else:
        print(f"Building DvP-by-position features using '{pos_col}' ...")

        # Defensive perspective: each row is "points allowed by def_team vs pos in this game"
        dvp_cols = ["season", "game_id", "game_date", "opp_abbrev", pos_col, "pts"]
        # PHASE 4A: Add shooting stats for enhanced DvP
        if "fgm" in df.columns and "fga" in df.columns:
            dvp_cols.extend(["fgm", "fga"])
        if "fg3m" in df.columns and "fg3a" in df.columns:
            dvp_cols.extend(["fg3m", "fg3a"])
        
        dvp_df = (
            df[dvp_cols]
            .rename(
                columns={
                    "opp_abbrev": "def_team",
                    pos_col: "pos",
                    "pts": "pts_allowed",
                }
            )
        )

        # Drop rows without a position
        dvp_df = dvp_df.dropna(subset=["pos"]).copy()

        # Aggregate by defensive team / position per game
        agg_dict = {"pts_allowed": "sum"}
        if "fgm" in dvp_df.columns:
            agg_dict["fgm"] = "sum"
            agg_dict["fga"] = "sum"
        if "fg3m" in dvp_df.columns:
            agg_dict["fg3m"] = "sum"
            agg_dict["fg3a"] = "sum"
        
        dvp_game = (
            dvp_df.groupby(
                ["season", "def_team", "pos", "game_date", "game_id"],
                as_index=False,
            ).agg(agg_dict)
        )
        
        # Rename for clarity
        if "fgm" in dvp_game.columns:
            dvp_game = dvp_game.rename(columns={"fgm": "fgm_allowed", "fga": "fga_allowed"})
        if "fg3m" in dvp_game.columns:
            dvp_game = dvp_game.rename(columns={"fg3m": "fg3m_allowed", "fg3a": "fg3a_allowed"})

        # Sort before rolling
        dvp_game = dvp_game.sort_values(
            ["season", "def_team", "pos", "game_date", "game_id"]
        ).reset_index(drop=True)

        group_def = dvp_game.groupby(["season", "def_team", "pos"], group_keys=False)
        dvp_game["opp_dvp_pos_pts_roll5"] = group_def["pts_allowed"].apply(
            lambda s: s.shift(1).rolling(ROLL_SHORT, min_periods=1).mean()
        )
        dvp_game["opp_dvp_pos_pts_roll15"] = group_def["pts_allowed"].apply(
            lambda s: s.shift(1).rolling(ROLL_LONG, min_periods=1).mean()
        )
        
        # PHASE 4A: Enhanced DvP - FG% and 3PT% allowed vs position
        if "fgm_allowed" in dvp_game.columns and "fga_allowed" in dvp_game.columns:
            # Rolling FG% allowed
            fgm_roll5 = group_def["fgm_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_SHORT, min_periods=1).sum()
            )
            fga_roll5 = group_def["fga_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_SHORT, min_periods=1).sum()
            )
            dvp_game["opp_fg_pct_allowed_vs_pos_roll5"] = (
                fgm_roll5 / fga_roll5.replace(0, np.nan)
            ).fillna(0.45)  # Default to league average if no data
            
            fgm_roll15 = group_def["fgm_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_LONG, min_periods=1).sum()
            )
            fga_roll15 = group_def["fga_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_LONG, min_periods=1).sum()
            )
            dvp_game["opp_fg_pct_allowed_vs_pos_roll15"] = (
                fgm_roll15 / fga_roll15.replace(0, np.nan)
            ).fillna(0.45)
        
        if "fg3m_allowed" in dvp_game.columns and "fg3a_allowed" in dvp_game.columns:
            # Rolling 3PT% allowed
            fg3m_roll5 = group_def["fg3m_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_SHORT, min_periods=1).sum()
            )
            fg3a_roll5 = group_def["fg3a_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_SHORT, min_periods=1).sum()
            )
            dvp_game["opp_3pt_pct_allowed_vs_pos_roll5"] = (
                fg3m_roll5 / fg3a_roll5.replace(0, np.nan)
            ).fillna(0.35)  # Default to league average
            
            fg3m_roll15 = group_def["fg3m_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_LONG, min_periods=1).sum()
            )
            fg3a_roll15 = group_def["fg3a_allowed"].apply(
                lambda s: s.shift(1).rolling(ROLL_LONG, min_periods=1).sum()
            )
            dvp_game["opp_3pt_pct_allowed_vs_pos_roll15"] = (
                fg3m_roll15 / fg3a_roll15.replace(0, np.nan)
            ).fillna(0.35)

        # Build feature list (include enhanced DvP if available)
        dvp_feat_cols = [
            "season",
            "game_id",
            "def_team",
            "pos",
            "opp_dvp_pos_pts_roll5",
            "opp_dvp_pos_pts_roll15",
        ]
        if "opp_fg_pct_allowed_vs_pos_roll5" in dvp_game.columns:
            dvp_feat_cols.extend([
                "opp_fg_pct_allowed_vs_pos_roll5",
                "opp_fg_pct_allowed_vs_pos_roll15",
            ])
        if "opp_3pt_pct_allowed_vs_pos_roll5" in dvp_game.columns:
            dvp_feat_cols.extend([
                "opp_3pt_pct_allowed_vs_pos_roll5",
                "opp_3pt_pct_allowed_vs_pos_roll15",
            ])
        
        dvp_feats = dvp_game[dvp_feat_cols]

        # Merge back to the original df (offensive perspective)
        df = df.merge(
            dvp_feats,
            left_on=["season", "game_id", "opp_abbrev", pos_col],
            right_on=["season", "game_id", "def_team", "pos"],
            how="left",
        )

        # Clean up helper keys
        df = df.drop(columns=["def_team", "pos"])

    # ------------------------------------------------------------------
    # 4) Keep the columns we care about and drop rows without enough history
    # ------------------------------------------------------------------
    cols_order = [
        "game_id",
        "season",
        "game_date",
        "player_id",
        "player_name",
        "team_abbrev",
        "opp_abbrev",
        "is_home",
        "days_since_last_game",

        # classic roll features
        "minutes_roll5",
        "minutes_roll15",
        "pts_roll5",
        "pts_roll15",
        "reb_roll5",
        "reb_roll15",
        "ast_roll5",
        "ast_roll15",
        "fg3m_roll5",
        "fg3m_roll15",
        "fg3a_roll5",
        "fg3a_roll15",
        "fga_roll5",
        "fga_roll15",
        "fta_roll5",
        "fta_roll15",
        # NEW: rolling usage volume (if usg_events was present)
        "usg_events_roll5",
        "usg_events_roll15",
        "opp_pts_allowed_roll5",
        "opp_pts_allowed_roll15",
        # NEW: team margin env features
        "team_margin_roll5",
        "team_margin_roll15",

        # matchup-by-position (may be NaN if no positions)
        "opp_dvp_pos_pts_roll5",
        "opp_dvp_pos_pts_roll15",
        # PHASE 4A: Enhanced DvP (if available)
        "opp_fg_pct_allowed_vs_pos_roll5",
        "opp_fg_pct_allowed_vs_pos_roll15",
        "opp_3pt_pct_allowed_vs_pos_roll5",
        "opp_3pt_pct_allowed_vs_pos_roll15",

        # player baselines
        "player_pts_career_mean",
        "player_pts_season_mean",
        "player_minutes_career_mean",
        "player_minutes_season_mean",

        # PHASE 4A: Player vs Opponent History
        "player_vs_opp_pts_avg_career",
        "player_vs_opp_pts_avg_last_5",
        "player_vs_opp_minutes_avg_career",
        "player_vs_opp_minutes_avg_last_5",
        "player_vs_opp_games_count",

        # NEW: role vs career & star tiers
        "rel_minutes_vs_career",
        "rel_pts_vs_career",
        "star_tier_pts",
        "star_tier_minutes",

        # trends
        "pts_trend_5_15",
        "minutes_trend_5_15",
        "fga_trend_5_15",

        # volatility
        "pts_std5",
        "minutes_std5",
        "fga_std5",

        # usage ratios
        "pts_per_min_roll5",
        "fga_per_min_roll5",
        "fta_per_min_roll5",

        # rest flags
        "is_b2b",
        "is_long_rest",

        # team pace env features (may be NaN for older games)
        "team_pace_roll5",
        "team_pace_roll15",

        # raw targets we keep at the end
        "minutes",
        "pts",
        "reb",
        "ast",
        "fg3m",
    ]

    missing_cols = [c for c in cols_order if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected columns not found after feature engineering: {missing_cols}")

    # Drop rows where any feature/target is NaN (mostly early career / first games)
    feat_df = df[cols_order].dropna().reset_index(drop=True)

    # Rename targets explicitly for downstream scripts
    feat_df = feat_df.rename(
        columns={
            "pts": "target_pts",
            "minutes": "target_min",
            "reb": "target_reb",
            "ast": "target_ast",
            "fg3m": "target_fg3m",
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(feat_df):,} rows to {OUT_CSV}")

    print("\nSample rows:")
    print(feat_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()