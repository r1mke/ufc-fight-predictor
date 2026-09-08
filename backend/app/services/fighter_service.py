from datetime import datetime

import pandas as pd

from app.config import FIGHTERS_CLEAN_PARQUET, FIGHTS_CLEAN_PARQUET, IMPUTATION_JSON
from app.ml.features import (
    age_years_at,
    aggregate_current_stats,
    apply_static_imputation,
    build_history_long,
    compute_primary_weight_class,
)
from app.schemas import FighterCareerStats, FighterDetail, FighterSummary

import json


def _fighter_id(fighter_url: str) -> str:
    return fighter_url.rstrip("/").split("/")[-1]


class FighterService:
    def __init__(self):
        fighters = pd.read_parquet(FIGHTERS_CLEAN_PARQUET)
        fights = pd.read_parquet(FIGHTS_CLEAN_PARQUET)

        self.history_long = build_history_long(fights)
        primary_wc = compute_primary_weight_class(self.history_long)
        imputation_params = json.loads(IMPUTATION_JSON.read_text())

        fighters = apply_static_imputation(fighters, primary_wc, imputation_params)
        fighters["primary_weight_class"] = fighters["fighter_name"].map(primary_wc).fillna("Unknown")
        fighters["id"] = fighters["fighter_url"].apply(_fighter_id)

        self.imputation_params = imputation_params
        self.fighters = fighters.set_index("id", drop=False)

    def search(self, query: str, limit: int = 20) -> list[FighterSummary]:
        if query:
            mask = self.fighters["fighter_name"].str.contains(query, case=False, na=False)
            matches = self.fighters[mask]
        else:
            matches = self.fighters
        matches = matches.sort_values(["wins"], ascending=False).head(limit)
        return [self._to_summary(row) for _, row in matches.iterrows()]

    def get_detail(self, fighter_id: str) -> FighterDetail:
        if fighter_id not in self.fighters.index:
            raise KeyError(fighter_id)
        row = self.fighters.loc[fighter_id]
        return self._to_detail(row)

    def primary_weight_class(self, fighter_id: str) -> str:
        return self.fighters.loc[fighter_id, "primary_weight_class"]

    def current_stats_dict(self, fighter_id: str) -> tuple[dict, str]:
        """Raw stat dict in the exact shape features.build_pairwise_row expects."""
        if fighter_id not in self.fighters.index:
            raise KeyError(fighter_id)
        row = self.fighters.loc[fighter_id]
        name = row["fighter_name"]
        history = self.history_long[self.history_long["fighter_name"] == name]
        agg = aggregate_current_stats(history)

        stats = {
            "height_in": row["height_in"],
            "reach_in": row["reach_in"],
            "weight_lbs": row["weight_lbs"],
            "age_years": age_years_at(row["dob"], pd.Timestamp(datetime.now()), self.imputation_params["global_age_median_years"]),
            "stance": row["stance"],
            "reach_missing": bool(row["reach_missing"]),
            "height_missing": bool(row["height_missing"]),
            "weight_missing": bool(row["weight_missing"]),
            **agg,
        }
        return stats, name

    def _to_summary(self, row) -> FighterSummary:
        return FighterSummary(
            id=row["id"],
            name=row["fighter_name"],
            weight_class=row["primary_weight_class"],
            wins=int(row["wins"]),
            losses=int(row["losses"]),
            draws=int(row["draws"]),
            height_in=row["height_in"],
            reach_in=row["reach_in"],
        )

    def _to_detail(self, row) -> FighterDetail:
        stats, name = self.current_stats_dict(row["id"])
        summary = self._to_summary(row)
        return FighterDetail(
            **summary.model_dump(),
            stance=row["stance"],
            age_years=stats["age_years"],
            weight_lbs=row["weight_lbs"],
            reach_missing=bool(row["reach_missing"]),
            height_missing=bool(row["height_missing"]),
            career_stats=FighterCareerStats(
                slpm=row["slpm"], str_acc=row["str_acc"], sapm=row["sapm"], str_def=row["str_def"],
                td_avg=row["td_avg"], td_acc=row["td_acc"], td_def=row["td_def"], sub_avg=row["sub_avg"],
            ),
            win_streak=int(stats["win_streak"]),
            fight_count=int(stats["prior_fight_count"]),
        )


_instance: FighterService | None = None


def get_fighter_service() -> FighterService:
    global _instance
    if _instance is None:
        _instance = FighterService()
    return _instance


def reset_instance() -> None:
    """Drops the cached singleton so the next call to get_fighter_service()
    rebuilds it from whatever is currently on disk - used after a retrain."""
    global _instance
    _instance = None
