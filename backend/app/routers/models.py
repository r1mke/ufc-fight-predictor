from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import DEFAULT_MODEL_VARIANT, MODEL_NAMES, MODEL_VARIANTS
from app.schemas import ModelMetrics, ModelsResponse, TargetMetrics
from app.services.prediction_service import PredictionService, get_prediction_service

router = APIRouter(prefix="/api/models", tags=["models"])


def _target_metrics(service: PredictionService, target: str, variant: str) -> TargetMetrics:
    metrics = service.get_metrics(variant)
    labels = metrics["targets"][target]["labels"]
    models = [ModelMetrics(**service.model_metrics(target, name, variant=variant)) for name in MODEL_NAMES]
    return TargetMetrics(
        labels=labels,
        models=models,
        anova_feature_importance=metrics["targets"][target]["anova_feature_importance"],
    )


@router.get("", response_model=ModelsResponse)
def list_models(
    variant: str = Query(DEFAULT_MODEL_VARIANT),
    service: PredictionService = Depends(get_prediction_service),
):
    if variant not in MODEL_VARIANTS:
        raise HTTPException(status_code=400, detail=f"variant must be one of {MODEL_VARIANTS}")
    try:
        metrics = service.get_metrics(variant)
        return ModelsResponse(
            winner=_target_metrics(service, "winner", variant),
            method=_target_metrics(service, "method", variant),
            correlation_pairs=metrics["correlation_pairs"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
