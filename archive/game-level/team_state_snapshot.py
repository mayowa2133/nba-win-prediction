# team_state_snapshot.py

import pandas as pd
from datetime import date

LAST_N = 10

# Elo hyperparameters (you can tweak to match elo_multi.py)
INITIAL_ELO = 1500.0
K = 20.0
HOME_ADV = 65.0  # home-court advantage in Elo points


class TeamStateSnapshot:
    """
    Maintains Elo, season-to-date stats, last-N stats, and rest info
    for each (season, team) pair as of a given cutoff date.
    """

    def __init__(self):
        # Elo: (season, team) -> current Elo
        self.elo = {}

        # Season-to-date rolling stats:
        # (season, team) -> {gp, pts_for, pts_against, wins}
        self.full_state = {}

        # Last-N stats:
        # (season, team) -> {pts_for: [], pts_against: [], wins: []}
        self.last_state = {}

        # Last game date:
        # (season, team) -> date
        self.last_game_date = {}

    @staticmethod
    def _expected_home_prob(home_elo: float, away_elo: float) -> float:
        """
        Standard Elo expected score formula with home advantage.
        """
        diff = (home_elo + HOME_ADV) - away_elo
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def build_from_games(self, games_df: pd.DataFrame, cutoff_date: date):
        """
        Build internal state using all games with date < cutoff_date.
        Expect columns: season, date, home_team, away_team, home_score, away_score.
        """
        df = games_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Only games strictly before the target date
        df = df[df["date"] < cutoff_date].sort_values("date").reset_index(drop=True)

        for _, row in df.iterrows():
            season = row["season"]
            d = row["date"]
            home = row["home_team"]
            away = row["away_team"]
            hs = row["home_score"]
            as_ = row["away_score"]

            key_home = (season, home)
            key_away = (season, away)

            # --- Elo ---
            h_elo = self.elo.get(key_home, INITIAL_ELO)
            a_elo = self.elo.get(key_away, INITIAL_ELO)

            expected_home = self._expected_home_prob(h_elo, a_elo)
            result_home = 1.0 if hs > as_ else 0.0

            delta = K * (result_home - expected_home)
            h_elo_new = h_elo + delta
            a_elo_new = a_elo - delta

            self.elo[key_home] = h_elo_new
            self.elo[key_away] = a_elo_new

            # --- Season-to-date rolling ---
            if key_home not in self.full_state:
                self.full_state[key_home] = {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0}
            if key_away not in self.full_state:
                self.full_state[key_away] = {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0}

            fs_h = self.full_state[key_home]
            fs_a = self.full_state[key_away]

            fs_h["gp"] += 1
            fs_h["pts_for"] += hs
            fs_h["pts_against"] += as_
            if hs > as_:
                fs_h["wins"] += 1

            fs_a["gp"] += 1
            fs_a["pts_for"] += as_
            fs_a["pts_against"] += hs
            if as_ > hs:
                fs_a["wins"] += 1

            # --- Last-N rolling ---
            if key_home not in self.last_state:
                self.last_state[key_home] = {"pts_for": [], "pts_against": [], "wins": []}
            if key_away not in self.last_state:
                self.last_state[key_away] = {"pts_for": [], "pts_against": [], "wins": []}

            lst_h = self.last_state[key_home]
            lst_a = self.last_state[key_away]

            # home
            lst_h["pts_for"].append(hs)
            lst_h["pts_against"].append(as_)
            lst_h["wins"].append(1 if hs > as_ else 0)
            if len(lst_h["pts_for"]) > LAST_N:
                lst_h["pts_for"].pop(0)
                lst_h["pts_against"].pop(0)
                lst_h["wins"].pop(0)

            # away
            lst_a["pts_for"].append(as_)
            lst_a["pts_against"].append(hs)
            lst_a["wins"].append(1 if as_ > hs else 0)
            if len(lst_a["pts_for"]) > LAST_N:
                lst_a["pts_for"].pop(0)
                lst_a["pts_against"].pop(0)
                lst_a["wins"].pop(0)

            # --- Last game date ---
            self.last_game_date[key_home] = d
            self.last_game_date[key_away] = d

    def get_team_features(self, season: int, team: str, game_date: date):
        """
        Return a dict of team-level features for (season, team) as of just
        before a game on game_date.
        """
        key = (season, team)

        # Elo
        elo = self.elo.get(key, INITIAL_ELO)

        # Season-to-date
        fs = self.full_state.get(key, {"gp": 0, "pts_for": 0.0, "pts_against": 0.0, "wins": 0})
        gp = fs["gp"]
        if gp > 0:
            simple_off = fs["pts_for"] / gp
            simple_def = fs["pts_against"] / gp
            simple_net = (fs["pts_for"] - fs["pts_against"]) / gp
            simple_win_pct = fs["wins"] / gp
        else:
            simple_off = simple_def = simple_net = simple_win_pct = None

        # Last-N
        lst = self.last_state.get(key, {"pts_for": [], "pts_against": [], "wins": []})
        n = len(lst["pts_for"])
        if n > 0:
            lastN_off = sum(lst["pts_for"]) / n
            lastN_def = sum(lst["pts_against"]) / n
            lastN_net = sum(pf - pa for pf, pa in zip(lst["pts_for"], lst["pts_against"])) / n
            lastN_win_pct = sum(lst["wins"]) / n
        else:
            lastN_off = lastN_def = lastN_net = lastN_win_pct = None

        # Rest & B2B
        last_d = self.last_game_date.get(key, None)
        if last_d is None:
            rest_days = None
            b2b = 0
        else:
            rest_days = (game_date - last_d).days
            b2b = 1 if rest_days == 1 else 0

        return {
            "elo": elo,
            "games_played": gp,
            "simple_off": simple_off,
            "simple_def": simple_def,
            "simple_net": simple_net,
            "simple_win_pct": simple_win_pct,
            "lastN_off": lastN_off,
            "lastN_def": lastN_def,
            "lastN_net": lastN_net,
            "lastN_win_pct": lastN_win_pct,
            "lastN_games": n,
            "rest_days": rest_days,
            "b2b": b2b,
        }
