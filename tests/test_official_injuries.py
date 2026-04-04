from __future__ import annotations

from src.data.official_injuries import normalize_report_status, parse_official_injury_report_text


def test_official_injury_report_parser_normalizes_statuses_and_team_rows():
    text = """
    NBA Injury Report
    03/31/2026 6:30 (ET) ATL@BOS Atlanta Hawks Trae Young Questionable - Left ankle soreness
    Atlanta Hawks Clint Capela Out - Rest
    Boston Celtics NOT YET SUBMITTED
    03/31/2026 8:00 (ET) LAL@GSW Los Angeles Lakers LeBron James Probable - Illness
    additional treatment before warmups
    """

    df = parse_official_injury_report_text(
        text,
        source_url="https://official.nba.com/test.pdf",
        pulled_at="2026-03-31T10:00:00Z",
    )

    assert len(df) == 4
    trae = df[df["player_name"] == "Trae Young"].iloc[0]
    assert trae["team_abbrev"] == "ATL"
    assert trae["normalized_status"] == "questionable"

    capela = df[df["player_name"] == "Clint Capela"].iloc[0]
    assert capela["normalized_status"] == "out"

    team_row = df[df["row_kind"] == "team_status"].iloc[0]
    assert team_row["team_abbrev"] == "BOS"
    assert team_row["normalized_status"] == "inactive_other"

    lebron = df[df["player_name"] == "LeBron James"].iloc[0]
    assert "additional treatment" in lebron["raw_reason"]
    assert normalize_report_status("NOT YET SUBMITTED") == "inactive_other"


def test_official_injury_report_parser_handles_tokenized_pdf_text():
    text = """
    Injury Report: 04/04/26 12:45 PM Page 1 of 2
    Game Date Game Time Matchup Team Player Name Current Status Reason
    04/04/2026 03:00 (ET) WAS@MIA
    Washington Wizards
    Champagnie, Justin Questionable Injury/Illness - Right Knee; Contusion
    Davis, Anthony Out Injury/Illness - Left Finger; Sprain
    Miami Heat
    Herro, Tyler Probable Injury/Illness - Right Foot; Soreness
    Wiggins, Andrew Available Injury/Illness - Left Big Toe; Sesamoiditis
    04/04/2026 07:00 (ET) DET@PHI
    Detroit Pistons
    Cunningham, Cade Out Injury/Illness - Left Lung; Pneumothorax
    Philadelphia 76ers
    NOT YET SUBMITTED
    """

    df = parse_official_injury_report_text(
        text,
        source_url="https://official.nba.com/tokenized.pdf",
        pulled_at="2026-04-04T12:45:00Z",
    )

    assert len(df) == 6
    justin = df[df["player_name"] == "Champagnie, Justin"].iloc[0]
    assert justin["team_abbrev"] == "WAS"
    assert justin["normalized_status"] == "questionable"

    herro = df[df["player_name"] == "Herro, Tyler"].iloc[0]
    assert herro["team_abbrev"] == "MIA"
    assert herro["normalized_status"] == "probable"

    team_row = df[df["row_kind"] == "team_status"].iloc[0]
    assert team_row["team_abbrev"] == "PHI"
    assert team_row["normalized_status"] == "inactive_other"
