from typing import Literal, Optional

from pydantic import BaseModel


class FighterSummary(BaseModel):
    id: str
    name: str
    weight_class: str
    wins: int
    losses: int
    draws: int
    height_in: Optional[float] = None
    reach_in: Optional[float] = None


class FighterCareerStats(BaseModel):
    slpm: Optional[float] = None
    str_acc: Optional[float] = None
    sapm: Optional[float] = None
    str_def: Optional[float] = None
    td_avg: Optional[float] = None
    td_acc: Optional[float] = None
    td_def: Optional[float] = None
    sub_avg: Optional[float] = None


class FighterDetail(FighterSummary):
    stance: str
    age_years: Optional[float] = None
    weight_lbs: Optional[float] = None
    reach_missing: bool
    height_missing: bool
    career_stats: FighterCareerStats
    win_streak: int
    fight_count: int


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    macro_f1: float
    log_loss: float
    cv_mean: float
    cv_std: float
    test_size: int


class TargetMetrics(BaseModel):
    labels: list[str]
    models: list[ModelMetrics]
    anova_feature_importance: list[dict]


class ModelsResponse(BaseModel):
    winner: TargetMetrics
    method: TargetMetrics
    correlation_pairs: list[dict]


class PredictRequest(BaseModel):
    fighter1_id: str
    fighter2_id: str
    weight_class: str
    model_name: str
    is_title_fight: bool = False


class WinnerPrediction(BaseModel):
    fighter_id: str
    fighter_name: str
    probability: float


class MethodPrediction(BaseModel):
    KO_TKO: float
    Submission: float
    Decision: float


class PredictResponse(BaseModel):
    winner: WinnerPrediction
    method: MethodPrediction
    model_name: str
    winner_model_metrics: ModelMetrics
    method_model_metrics: ModelMetrics
    top_features: list[dict]
    fighter1: FighterDetail
    fighter2: FighterDetail


# --- Data acquisition (scraping / manual submission / review) ---------------

GRANULAR_STAT_FIELDS = [
    "f1_sig_landed", "f1_sig_att", "f2_sig_landed", "f2_sig_att",
    "f1_td_landed", "f1_td_att", "f2_td_landed", "f2_td_att",
    "f1_ctrl_sec", "f2_ctrl_sec", "f1_kd", "f2_kd", "f1_sub_att", "f2_sub_att",
]


class NewFighterInfo(BaseModel):
    """Physical/career info for a fighter discovered by the scraper who isn't
    in fighters_clean yet (a debutant)."""
    name: str
    height_in: Optional[float] = None
    reach_in: Optional[float] = None
    weight_lbs: Optional[float] = None
    stance: Optional[str] = None
    dob: Optional[str] = None


class GranularStats(BaseModel):
    f1_sig_landed: Optional[float] = None
    f1_sig_att: Optional[float] = None
    f2_sig_landed: Optional[float] = None
    f2_sig_att: Optional[float] = None
    f1_td_landed: Optional[float] = None
    f1_td_att: Optional[float] = None
    f2_td_landed: Optional[float] = None
    f2_td_att: Optional[float] = None
    f1_ctrl_sec: Optional[float] = None
    f2_ctrl_sec: Optional[float] = None
    f1_kd: Optional[float] = None
    f2_kd: Optional[float] = None
    f1_sub_att: Optional[float] = None
    f2_sub_att: Optional[float] = None


class PendingFightCreate(GranularStats):
    """Fields an admin fills in manually, or that the scraper produces."""
    fighter1_id: Optional[str] = None
    fighter2_id: Optional[str] = None
    fighter1_name: str
    fighter2_name: str
    weight_class: str
    method: Literal["KO_TKO", "Submission", "Decision"]
    winner_name: str
    event_date: str
    is_title_fight: bool = False
    new_fighter_1: Optional[NewFighterInfo] = None
    new_fighter_2: Optional[NewFighterInfo] = None
    source_url: Optional[str] = None  # scraped fight-details URL, used to dedupe re-scrapes


class PendingFightUpdate(GranularStats):
    """PATCH payload - every field optional, only provided ones are changed."""
    fighter1_name: Optional[str] = None
    fighter2_name: Optional[str] = None
    weight_class: Optional[str] = None
    method: Optional[Literal["KO_TKO", "Submission", "Decision"]] = None
    winner_name: Optional[str] = None
    event_date: Optional[str] = None
    is_title_fight: Optional[bool] = None


class PendingFight(PendingFightCreate):
    id: str
    source: Literal["manual", "scraped"]
    status: Literal["pending", "approved", "rejected"] = "pending"
    submitted_at: str
    reviewed_at: Optional[str] = None


class RetrainRequest(BaseModel):
    pending_ids: list[str]


class ScrapeStatus(BaseModel):
    status: Literal["idle", "running", "done", "error"]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    result_summary: Optional[str] = None


class RetrainStatus(BaseModel):
    status: Literal["idle", "running", "done", "error"]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    result_summary: Optional[str] = None


class ModelVersion(BaseModel):
    version_id: str
    created_at: str
    is_current: bool
    metrics_summary: dict


class ModelVersionsResponse(BaseModel):
    versions: list[ModelVersion]
