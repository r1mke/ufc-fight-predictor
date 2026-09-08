from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas import FighterDetail, FighterSummary
from app.services.fighter_service import FighterService, get_fighter_service

router = APIRouter(prefix="/api/fighters", tags=["fighters"])


@router.get("", response_model=list[FighterSummary])
def search_fighters(
    search: str = Query("", description="Case-insensitive substring match on fighter name"),
    limit: int = Query(20, ge=1, le=100),
    service: FighterService = Depends(get_fighter_service),
):
    return service.search(search, limit)


@router.get("/{fighter_id}", response_model=FighterDetail)
def get_fighter(fighter_id: str, service: FighterService = Depends(get_fighter_service)):
    try:
        return service.get_detail(fighter_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Fighter not found")
