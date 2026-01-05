import requests
import pandas as pd
import time

# ---------- CONFIG ----------

BASE_URL = "https://api.balldontlie.io/v1/games"
API_KEY = "82389ab9-bb51-4f74-9631-4d1c88ba7407"  # <-- put your real key here

# Seasons you want (regular season year, e.g. 2015 = 2015-16, 2025 = 2025-26)
START_SEASON = 2015
END_SEASON = 2025  # inclusive


def get_page_with_retry(params, max_retries: int = 10):
    headers = {"Authorization": API_KEY}

    for attempt in range(max_retries):
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=10)

        # Handle rate limiting
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait_seconds = int(retry_after)
                except ValueError:
                    wait_seconds = 5
            else:
                wait_seconds = 5

            print(f"Hit rate limit (429). Waiting {wait_seconds} seconds then retrying...")
            time.sleep(wait_seconds)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError("Too many retries due to rate limiting.")


def fetch_games_for_season(season: int) -> pd.DataFrame:
    """
    Fetch all NBA games for a given season from BallDontLie
    and return them as a pandas DataFrame.
    """
    games = []
    cursor = None

    while True:
        params = {
            "per_page": 100,
            "seasons[]": season,
        }
        if cursor is not None:
            params["cursor"] = cursor

        print(f"Fetching season {season}, cursor={cursor}...")
        resp = get_page_with_retry(params)
        data = resp.json()

        for g in data["data"]:
            games.append({
                "game_id": g["id"],
                "date": g["date"],              # YYYY-MM-DD
                "season": g["season"],
                "status": g["status"],
                "postseason": g["postseason"],
                "home_team": g["home_team"]["abbreviation"],
                "away_team": g["visitor_team"]["abbreviation"],
                "home_score": g["home_team_score"],
                "away_score": g["visitor_team_score"],
            })

        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")

        if not cursor:
            break

    df = pd.DataFrame(games)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    all_seasons = []
    for season in range(START_SEASON, END_SEASON + 1):
        df_season = fetch_games_for_season(season)
        out_path = f"games_{season}.csv"
        df_season.to_csv(out_path, index=False)
        print(f"Saved {len(df_season)} games to {out_path}")
        all_seasons.append(df_season)

        # tiny pause between seasons just to be polite to the API
        time.sleep(1)

    # Combine all seasons into one big CSV
    all_games = pd.concat(all_seasons, ignore_index=True)
    all_games = all_games.sort_values("date").reset_index(drop=True)
    all_games.to_csv(f"games_all_{START_SEASON}_{END_SEASON}.csv", index=False)
    print(f"Saved {len(all_games)} total games to games_all_{START_SEASON}_{END_SEASON}.csv")
