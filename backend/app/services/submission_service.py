"""Manages the pending-fight review queue: scraped or manually-added fight
results wait here (status="pending") until an admin approves them, at which
point they're converted into a row matching the original raw dataset's exact
column schema and appended to data/raw/user_submitted_fights.csv (and, for
any newly-discovered fighter, user_submitted_fighters.csv) so the existing
data_pipeline.py can pick them up unchanged on the next retrain.
"""

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from app.config import (
    PENDING_FIGHTS_CSV,
    USER_SUBMITTED_FIGHTERS_CSV,
    USER_SUBMITTED_FIGHTS_CSV,
)
from app.schemas import PendingFight, PendingFightCreate, PendingFightUpdate

PENDING_COLUMNS = [
    "id", "source", "status", "fighter1_id", "fighter2_id",
    "fighter1_name", "fighter2_name", "weight_class", "method", "winner_name",
    "event_date", "is_title_fight", "source_url",
    "f1_sig_landed", "f1_sig_att", "f2_sig_landed", "f2_sig_att",
    "f1_td_landed", "f1_td_att", "f2_td_landed", "f2_td_att",
    "f1_ctrl_sec", "f2_ctrl_sec", "f1_kd", "f2_kd", "f1_sub_att", "f2_sub_att",
    "new_fighter_1_json", "new_fighter_2_json",
    "submitted_at", "reviewed_at",
]

GRANULAR_FIELDS = [
    "f1_sig_landed", "f1_sig_att", "f2_sig_landed", "f2_sig_att",
    "f1_td_landed", "f1_td_att", "f2_td_landed", "f2_td_att",
    "f1_ctrl_sec", "f2_ctrl_sec", "f1_kd", "f2_kd", "f1_sub_att", "f2_sub_att",
]

_RAW_FIGHTS_COLUMNS = [
    "Fight_URL", "Fighter_1", "Fighter_2", "Winner", "Weight_Class", "Method",
    "End_Round", "End_Time", "Total_Fight_Time_Sec", "Time_Format",
    "F1_KD", "F2_KD", "F1_Sig_Landed", "F1_Sig_Att", "F2_Sig_Landed", "F2_Sig_Att",
    "F1_TD_Landed", "F2_TD_Landed", "F1_TD_Att", "F2_TD_Att",
    "F1_Sub_Att", "F2_Sub_Att", "F1_Ctrl_Sec", "F2_Ctrl_Sec",
    "F1_Head", "F2_Head", "F1_Body", "F2_Body", "F1_Leg", "F2_Leg",
    "F1_Distance", "F2_Distance", "F1_Clinch", "F2_Clinch",
    "F1_Ground", "F2_Ground", "Event_Date",
]

_RAW_FIGHTERS_COLUMNS = [
    "Fighter_Name", "Height", "Weight", "Reach", "Stance", "DOB",
    "Wins", "Losses", "Draws", "SLpM", "Str_Acc", "SApM", "Str_Def",
    "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg", "Fighter_URL",
]

_METHOD_TO_RAW = {
    "KO_TKO": "KO/TKO",
    "Submission": "Submission",
    "Decision": "Decision - Unanimous",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_df() -> pd.DataFrame:
    if not PENDING_FIGHTS_CSV.exists():
        return pd.DataFrame(columns=PENDING_COLUMNS)
    df = pd.read_csv(PENDING_FIGHTS_CSV, dtype=str)
    for col in PENDING_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")


def _save_df(df: pd.DataFrame) -> None:
    PENDING_FIGHTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[PENDING_COLUMNS].to_csv(PENDING_FIGHTS_CSV, index=False)


def _row_to_model(row: pd.Series) -> PendingFight:
    data = row.to_dict()
    for field in GRANULAR_FIELDS:
        data[field] = float(data[field]) if data.get(field) not in (None, "", "nan") else None
    data["is_title_fight"] = str(data.get("is_title_fight")).lower() == "true"
    for key in ("new_fighter_1", "new_fighter_2"):
        raw = data.pop(f"{key}_json", "") or ""
        data[key] = json.loads(raw) if raw else None
    data["reviewed_at"] = data.get("reviewed_at") or None
    data["source_url"] = data.get("source_url") or None
    data["fighter1_id"] = data.get("fighter1_id") or None
    data["fighter2_id"] = data.get("fighter2_id") or None
    return PendingFight(**data)


def _model_to_row(model: PendingFight) -> dict:
    row = model.model_dump()
    row["new_fighter_1_json"] = json.dumps(row.pop("new_fighter_1")) if row.get("new_fighter_1") else ""
    row["new_fighter_2_json"] = json.dumps(row.pop("new_fighter_2")) if row.get("new_fighter_2") else ""
    row["reviewed_at"] = row.get("reviewed_at") or ""
    row["source_url"] = row.get("source_url") or ""
    row["fighter1_id"] = row.get("fighter1_id") or ""
    row["fighter2_id"] = row.get("fighter2_id") or ""
    return row


def list_pending() -> list[PendingFight]:
    df = _load_df()
    return [_row_to_model(row) for _, row in df.iterrows()]


def get(pending_id: str) -> PendingFight:
    df = _load_df()
    match = df[df["id"] == pending_id]
    if match.empty:
        raise KeyError(pending_id)
    return _row_to_model(match.iloc[0])


def create_manual(payload: PendingFightCreate) -> PendingFight:
    model = PendingFight(
        id=str(uuid.uuid4()),
        source="manual",
        status="pending",
        submitted_at=_now_iso(),
        **payload.model_dump(),
    )
    df = _load_df()
    df = pd.concat([df, pd.DataFrame([_model_to_row(model)])], ignore_index=True)
    _save_df(df)
    return model


def add_scraped_batch(fights: list[dict]) -> int:
    """Adds scraper results as pending rows, skipping any fight already seen
    (by source_url) in a previous scrape run."""
    df = _load_df()
    known_urls = set(df["source_url"]) if "source_url" in df.columns else set()
    new_rows = []
    for fight in fights:
        if fight.get("source_url") in known_urls:
            continue
        model = PendingFight(
            id=str(uuid.uuid4()),
            source="scraped",
            status="pending",
            submitted_at=_now_iso(),
            **fight,
        )
        new_rows.append(_model_to_row(model))
        known_urls.add(fight.get("source_url"))
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        _save_df(df)
    return len(new_rows)


def update(pending_id: str, patch: PendingFightUpdate) -> PendingFight:
    df = _load_df()
    mask = df["id"] == pending_id
    if not mask.any():
        raise KeyError(pending_id)
    model = _row_to_model(df[mask].iloc[0])
    updated = model.model_copy(update={k: v for k, v in patch.model_dump().items() if v is not None})
    df.loc[mask, list(_model_to_row(updated).keys())] = list(_model_to_row(updated).values())
    _save_df(df)
    return updated


def reject(pending_id: str) -> PendingFight:
    df = _load_df()
    mask = df["id"] == pending_id
    if not mask.any():
        raise KeyError(pending_id)
    df.loc[mask, "status"] = "rejected"
    df.loc[mask, "reviewed_at"] = _now_iso()
    _save_df(df)
    return _row_to_model(df[mask].iloc[0])


def _height_in_to_raw(inches):
    if inches is None:
        return ""
    feet = int(inches // 12)
    remainder = round(inches - feet * 12)
    return f"{feet}' {remainder}\""


def _fighter_url_for(fighter_id: str) -> str:
    return f"http://ufcstats.com/fighter-details/{fighter_id}"


def _new_fighter_row(fighter_id: str, info: dict) -> dict:
    return {
        "Fighter_Name": info.get("name") or "",
        "Height": _height_in_to_raw(info.get("height_in")),
        "Weight": f"{info['weight_lbs']} lbs." if info.get("weight_lbs") is not None else "",
        "Reach": f"{info['reach_in']}\"" if info.get("reach_in") is not None else "",
        "Stance": info.get("stance") or "",
        "DOB": info.get("dob") or "",
        "Wins": "", "Losses": "", "Draws": "",
        "SLpM": "", "Str_Acc": "", "SApM": "", "Str_Def": "",
        "TD_Avg": "", "TD_Acc": "", "TD_Def": "", "Sub_Avg": "",
        "Fighter_URL": _fighter_url_for(fighter_id),
    }


def _pending_to_raw_fight_row(model: PendingFight) -> dict:
    row = {col: "" for col in _RAW_FIGHTS_COLUMNS}
    row["Fight_URL"] = model.source_url or f"user-submission:{model.id}"
    row["Fighter_1"] = model.fighter1_name
    row["Fighter_2"] = model.fighter2_name
    row["Winner"] = model.winner_name
    suffix = "Title Bout" if model.is_title_fight else "Bout"
    row["Weight_Class"] = f"{model.weight_class} {suffix}"
    row["Method"] = _METHOD_TO_RAW[model.method]
    row["Event_Date"] = model.event_date
    for field in GRANULAR_FIELDS:
        value = getattr(model, field)
        if value is None:
            continue
        column = {
            "f1_sig_landed": "F1_Sig_Landed", "f1_sig_att": "F1_Sig_Att",
            "f2_sig_landed": "F2_Sig_Landed", "f2_sig_att": "F2_Sig_Att",
            "f1_td_landed": "F1_TD_Landed", "f1_td_att": "F1_TD_Att",
            "f2_td_landed": "F2_TD_Landed", "f2_td_att": "F2_TD_Att",
            "f1_ctrl_sec": "F1_Ctrl_Sec", "f2_ctrl_sec": "F2_Ctrl_Sec",
            "f1_kd": "F1_KD", "f2_kd": "F2_KD",
            "f1_sub_att": "F1_Sub_Att", "f2_sub_att": "F2_Sub_Att",
        }[field]
        row[column] = value
    return row


def approve_selected(pending_ids: list[str]) -> int:
    """Converts the given pending fights into raw-schema rows appended to
    user_submitted_fights.csv (and user_submitted_fighters.csv for any new
    fighter), marks them approved. Returns the number approved."""
    df = _load_df()
    fight_rows = []
    fighter_rows = []
    seen_new_fighter_ids = set()

    for pending_id in pending_ids:
        mask = df["id"] == pending_id
        if not mask.any():
            continue
        model = _row_to_model(df[mask].iloc[0])
        if model.status != "pending":
            continue

        fight_rows.append(_pending_to_raw_fight_row(model))

        for side, fighter_id, info in (
            ("1", model.fighter1_id, model.new_fighter_1),
            ("2", model.fighter2_id, model.new_fighter_2),
        ):
            if info and fighter_id and fighter_id not in seen_new_fighter_ids:
                fighter_rows.append(_new_fighter_row(fighter_id, info.model_dump() if hasattr(info, "model_dump") else info))
                seen_new_fighter_ids.add(fighter_id)

        df.loc[mask, "status"] = "approved"
        df.loc[mask, "reviewed_at"] = _now_iso()

    if fight_rows:
        _append_csv(USER_SUBMITTED_FIGHTS_CSV, fight_rows, _RAW_FIGHTS_COLUMNS)
    if fighter_rows:
        _append_csv(USER_SUBMITTED_FIGHTERS_CSV, fighter_rows, _RAW_FIGHTERS_COLUMNS)

    _save_df(df)
    return len(fight_rows)


def _append_csv(path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=columns)
    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        combined = pd.concat([existing, new_df.astype(str)], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)
