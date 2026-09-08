from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException

from app.config import ADMIN_TOKEN
from app.schemas import (
    ModelVersionsResponse,
    PendingFight,
    PendingFightCreate,
    PendingFightUpdate,
    RetrainRequest,
    RetrainStatus,
    ScrapeStatus,
)
from app.services import model_version_service, retrain_service, scraper_service, submission_service

router = APIRouter(prefix="/api/admin", tags=["admin"])


def require_admin_token(admin_token: str = Header(..., alias="Admin-Token")):
    if not ADMIN_TOKEN or admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Admin-Token")


@router.post("/scrape", response_model=ScrapeStatus, dependencies=[Depends(require_admin_token)])
def trigger_scrape(background_tasks: BackgroundTasks):
    if scraper_service.is_running():
        raise HTTPException(status_code=409, detail="A scrape is already running")
    background_tasks.add_task(scraper_service.run_scrape)
    return ScrapeStatus(status="running")


@router.get("/scrape/status", response_model=ScrapeStatus, dependencies=[Depends(require_admin_token)])
def scrape_status():
    return scraper_service.get_status()


@router.get("/pending", response_model=list[PendingFight], dependencies=[Depends(require_admin_token)])
def list_pending():
    return submission_service.list_pending()


@router.post("/pending", response_model=PendingFight, dependencies=[Depends(require_admin_token)])
def create_pending(payload: PendingFightCreate):
    return submission_service.create_manual(payload)


@router.patch("/pending/{pending_id}", response_model=PendingFight, dependencies=[Depends(require_admin_token)])
def update_pending(pending_id: str, patch: PendingFightUpdate):
    try:
        return submission_service.update(pending_id, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pending fight not found")


@router.delete("/pending/{pending_id}", response_model=PendingFight, dependencies=[Depends(require_admin_token)])
def reject_pending(pending_id: str):
    try:
        return submission_service.reject(pending_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pending fight not found")


@router.post("/retrain", response_model=RetrainStatus, dependencies=[Depends(require_admin_token)])
def trigger_retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    if not request.pending_ids:
        raise HTTPException(status_code=400, detail="pending_ids must not be empty")
    if retrain_service.is_running():
        raise HTTPException(status_code=409, detail="A retrain is already running")
    background_tasks.add_task(retrain_service.run_retrain, request.pending_ids)
    return RetrainStatus(status="running")


@router.get("/retrain/status", response_model=RetrainStatus, dependencies=[Depends(require_admin_token)])
def retrain_status():
    return retrain_service.get_status()


@router.get("/model-versions", response_model=ModelVersionsResponse, dependencies=[Depends(require_admin_token)])
def list_model_versions():
    return ModelVersionsResponse(versions=model_version_service.list_versions())


@router.post("/model-versions/{version_id}/restore", dependencies=[Depends(require_admin_token)])
def restore_model_version(version_id: str):
    try:
        model_version_service.restore(version_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model version not found")
    return {"status": "restored", "version_id": version_id}
