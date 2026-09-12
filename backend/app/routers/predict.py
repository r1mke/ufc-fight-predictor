from fastapi import APIRouter, Depends, HTTPException

from app.config import MODEL_NAMES, MODEL_VARIANTS
from app.schemas import MethodPrediction, PredictRequest, PredictResponse, WinnerPrediction
from app.services.fighter_service import FighterService, get_fighter_service
from app.services.prediction_service import PredictionService, get_prediction_service

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    fighter_service: FighterService = Depends(get_fighter_service),
):
    if request.model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"model_name must be one of {MODEL_NAMES}")
    if request.variant not in MODEL_VARIANTS:
        raise HTTPException(status_code=400, detail=f"variant must be one of {MODEL_VARIANTS}")
    if request.fighter1_id == request.fighter2_id:
        raise HTTPException(status_code=400, detail="fighter1_id and fighter2_id must differ")

    try:
        fighter1 = fighter_service.get_detail(request.fighter1_id)
        fighter2 = fighter_service.get_detail(request.fighter2_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {e}")

    try:
        result = prediction_service.predict(
            request.fighter1_id,
            request.fighter2_id,
            request.weight_class,
            request.model_name,
            request.is_title_fight,
            variant=request.variant,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {e}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    a_wins = result["a_win_probability"] >= 0.5
    winner = WinnerPrediction(
        fighter_id=fighter1.id if a_wins else fighter2.id,
        fighter_name=fighter1.name if a_wins else fighter2.name,
        probability=result["a_win_probability"] if a_wins else result["b_win_probability"],
    )

    return PredictResponse(
        winner=winner,
        method=MethodPrediction(**result["method_probabilities"]),
        model_name=request.model_name,
        variant=request.variant,
        winner_model_metrics=prediction_service.model_metrics("winner", request.model_name, variant=request.variant),
        method_model_metrics=prediction_service.model_metrics("method", request.model_name, variant=request.variant),
        top_features=prediction_service.top_features("method", variant=request.variant),
        fighter1=fighter1,
        fighter2=fighter2,
    )
