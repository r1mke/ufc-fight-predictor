from fastapi import APIRouter, Depends

from app.config import MODEL_NAMES
from app.schemas import ModelMetrics, ModelsResponse, TargetMetrics
from app.services.prediction_service import PredictionService, get_prediction_service

router = APIRouter(prefix="/api/models", tags=["models"])


def _target_metrics(service: PredictionService, target: str) -> TargetMetrics:
    labels = service.metrics["targets"][target]["labels"]
    models = [ModelMetrics(**service.model_metrics(target, name)) for name in MODEL_NAMES]
    return TargetMetrics(
        labels=labels,
        models=models,
        anova_feature_importance=service.metrics["targets"][target]["anova_feature_importance"],
    )


@router.get("", response_model=ModelsResponse)
def list_models(service: PredictionService = Depends(get_prediction_service)):
    return ModelsResponse(
        winner=_target_metrics(service, "winner"),
        method=_target_metrics(service, "method"),
        correlation_pairs=service.metrics["correlation_pairs"],
    )
