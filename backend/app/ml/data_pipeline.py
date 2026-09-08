"""Parses the raw UFC CSVs into cleaned, typed parquet tables.

Two outputs:
- fighters_clean.parquet: one row per fighter, physical attributes as numbers.
- fights_clean.parquet: one row per fight, normalized Method (3 classes) and
  Weight_Class (canonical categories), sorted by Event_Date. Fights with an
  ambiguous outcome (Draw/NC, Overturned, DQ, Could Not Continue, Other) are
  dropped here since they carry no clean method-of-victory signal.
"""

import re

import pandas as pd

from app.config import (
    FIGHTERS_CLEAN_PARQUET,
    FIGHTS_CLEAN_PARQUET,
    PROCESSED_DIR,
    RAW_FIGHTERS_CSV,
    RAW_FIGHTS_CSV,
)

HEIGHT_RE = re.compile(r"(\d+)'\s*(\d+)\"")

# Order matters: more specific labels (Women's divisions, Light Heavyweight)
# must be matched before their substrings (Bantamweight, Heavyweight).
WEIGHT_CLASS_KEYWORDS = [
    ("Women's Strawweight", "Women's Strawweight"),
    ("Women's Flyweight", "Women's Flyweight"),
    ("Women's Bantamweight", "Women's Bantamweight"),
    ("Women's Featherweight", "Women's Featherweight"),
    ("Light Heavyweight", "Light Heavyweight"),
    ("Flyweight", "Flyweight"),
    ("Bantamweight", "Bantamweight"),
    ("Featherweight", "Featherweight"),
    ("Lightweight", "Lightweight"),
    ("Welterweight", "Welterweight"),
    ("Middleweight", "Middleweight"),
    ("Heavyweight", "Heavyweight"),
    ("Catch Weight", "Catch Weight"),
    ("Open Weight", "Open Weight"),
]

METHOD_MAP = {
    "KO/TKO": "KO_TKO",
    "TKO - Doctor's Stoppage": "KO_TKO",
    "Submission": "Submission",
    "Decision - Unanimous": "Decision",
    "Decision - Split": "Decision",
    "Decision - Majority": "Decision",
}


def parse_height_to_inches(value):
    if not isinstance(value, str) or not value.strip():
        return None
    m = HEIGHT_RE.search(value)
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return float(feet * 12 + inches)


def parse_reach_to_inches(value):
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.replace('"', "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_weight_lbs(value):
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.replace("lbs.", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_percent(value):
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.replace("%", "").strip()
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return None


def normalize_weight_class(raw):
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    for keyword, canonical in WEIGHT_CLASS_KEYWORDS:
        if keyword in raw:
            return canonical
    return "Unknown"


def is_title_fight(raw):
    return isinstance(raw, str) and "Title" in raw


def normalize_method(method):
    """Returns the canonical 3-class method, or None if the fight should be dropped."""
    return METHOD_MAP.get(method)


def clean_fighters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["height_in"] = out["Height"].apply(parse_height_to_inches)
    out["reach_in"] = out["Reach"].apply(parse_reach_to_inches)
    out["weight_lbs"] = out["Weight"].apply(parse_weight_lbs)
    out["stance"] = out["Stance"].apply(lambda s: s.strip() if isinstance(s, str) and s.strip() else "Unknown")
    out["dob"] = pd.to_datetime(out["DOB"], errors="coerce")
    out["wins"] = pd.to_numeric(out["Wins"], errors="coerce").fillna(0).astype(int)
    out["losses"] = pd.to_numeric(out["Losses"], errors="coerce").fillna(0).astype(int)
    out["draws"] = pd.to_numeric(out["Draws"], errors="coerce").fillna(0).astype(int)
    out["slpm"] = pd.to_numeric(out["SLpM"], errors="coerce")
    out["str_acc"] = out["Str_Acc"].apply(parse_percent)
    out["sapm"] = pd.to_numeric(out["SApM"], errors="coerce")
    out["str_def"] = out["Str_Def"].apply(parse_percent)
    out["td_avg"] = pd.to_numeric(out["TD_Avg"], errors="coerce")
    out["td_acc"] = out["TD_Acc"].apply(parse_percent)
    out["td_def"] = out["TD_Def"].apply(parse_percent)
    out["sub_avg"] = pd.to_numeric(out["Sub_Avg"], errors="coerce")

    result = out[
        [
            "Fighter_Name",
            "Fighter_URL",
            "height_in",
            "reach_in",
            "weight_lbs",
            "stance",
            "dob",
            "wins",
            "losses",
            "draws",
            "slpm",
            "str_acc",
            "sapm",
            "str_def",
            "td_avg",
            "td_acc",
            "td_def",
            "sub_avg",
        ]
    ].rename(columns={"Fighter_Name": "fighter_name", "Fighter_URL": "fighter_url"})

    # A handful of names are reused by different fighters (different URL, same
    # name) - keep fighter_url as the true unique key throughout the pipeline.
    return result


def clean_fights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_class"] = out["Method"].apply(normalize_method)
    out["weight_class"] = out["Weight_Class"].apply(normalize_weight_class)
    out["is_title_fight"] = out["Weight_Class"].apply(is_title_fight)
    out["event_date"] = pd.to_datetime(out["Event_Date"], errors="coerce")

    numeric_cols = [
        "F1_KD", "F2_KD",
        "F1_Sig_Landed", "F1_Sig_Att", "F2_Sig_Landed", "F2_Sig_Att",
        "F1_TD_Landed", "F2_TD_Landed", "F1_TD_Att", "F2_TD_Att",
        "F1_Sub_Att", "F2_Sub_Att",
        "F1_Ctrl_Sec", "F2_Ctrl_Sec",
        "F1_Head", "F2_Head", "F1_Body", "F2_Body", "F1_Leg", "F2_Leg",
        "F1_Distance", "F2_Distance", "F1_Clinch", "F2_Clinch",
        "F1_Ground", "F2_Ground",
        "End_Round", "Total_Fight_Time_Sec",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop fights with no valid winner (Draw/NC) or an ambiguous method
    # (Overturned/DQ/Could Not Continue/Other) - not a clean skill-based outcome.
    valid_winner = (out["Winner"] == out["Fighter_1"]) | (out["Winner"] == out["Fighter_2"])
    out = out[valid_winner & out["method_class"].notna()].copy()

    out["winner_is_f1"] = out["Winner"] == out["Fighter_1"]

    out = out.rename(columns={
        "Fight_URL": "fight_url",
        "Fighter_1": "fighter_1",
        "Fighter_2": "fighter_2",
    })

    keep_cols = [
        "fight_url", "fighter_1", "fighter_2", "winner_is_f1",
        "method_class", "weight_class", "is_title_fight", "event_date",
        "End_Round", "Total_Fight_Time_Sec",
    ] + numeric_cols[:-2]
    out = out[keep_cols].rename(columns={"End_Round": "end_round", "Total_Fight_Time_Sec": "total_fight_time_sec"})
    out = out.sort_values("event_date").reset_index(drop=True)
    return out


def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fighters_raw = pd.read_csv(RAW_FIGHTERS_CSV)
    fights_raw = pd.read_csv(RAW_FIGHTS_CSV)

    fighters_clean = clean_fighters(fighters_raw)
    fights_clean = clean_fights(fights_raw)

    fighters_clean.to_parquet(FIGHTERS_CLEAN_PARQUET, index=False)
    fights_clean.to_parquet(FIGHTS_CLEAN_PARQUET, index=False)

    print(f"fighters: {len(fighters_raw)} -> {len(fighters_clean)} cleaned")
    print(f"fights: {len(fights_raw)} -> {len(fights_clean)} cleaned (dropped {len(fights_raw) - len(fights_clean)})")
    print(f"method_class distribution:\n{fights_clean['method_class'].value_counts()}")
    print(f"weight_class distribution:\n{fights_clean['weight_class'].value_counts()}")


if __name__ == "__main__":
    run()
