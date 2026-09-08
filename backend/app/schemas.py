from typing import Optional

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
