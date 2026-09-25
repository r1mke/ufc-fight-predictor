"""Orchestrates a full retrain: archive the current models, approve the
selected pending fights into the raw dataset, rebuild the cleaned data and
training table (now including the newly-approved fights), retrain all models
for every feature-engineering variant (diff/concat/combined), and hot-reload
the running services - no server restart needed.

Only one retrain can run at a time (in-memory flag); good enough for a
single-process, single-admin deployment.
"""

import threading
import traceback
from datetime import datetime, timezone

from app.config import MODEL_NAMES, MODEL_VARIANTS, TARGETS
from app.schemas import RetrainStatus
from app.services import model_version_service, submission_service
from app.services.fighter_service import reset_instance as reset_fighter_service
from app.services.prediction_service import reset_instance as reset_prediction_service

_status = RetrainStatus(status="idle")
_lock = threading.Lock()


def get_status() -> RetrainStatus:
    return _status


def is_running() -> bool:
    return _status.status == "running"


def run_retrain(pending_ids: list[str]) -> None:
    """Runs synchronously - the caller (the admin router) is responsible for
    invoking this from a FastAPI BackgroundTask so the HTTP request returns
    immediately while this keeps working."""
    global _status

    with _lock:
        if _status.status == "running":
            raise RuntimeError("a retrain is already running")
        _status = RetrainStatus(status="running", started_at=datetime.now(timezone.utc).isoformat())

    try:
        approved_count = submission_service.approve_selected(pending_ids)
        model_version_service.archive_current()

        # Imported lazily so a fresh process always picks up the latest
        # user_submitted_*.csv files written just above.
        from app.ml import data_pipeline, features, train

        data_pipeline.run()
        for variant in MODEL_VARIANTS:
            features.build_and_save(variant=variant)
            train.run(variant=variant)

        reset_fighter_service()
        reset_prediction_service()

        model_count = len(MODEL_VARIANTS) * len(MODEL_NAMES) * len(TARGETS)
        _status = RetrainStatus(
            status="done",
            started_at=_status.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            result_summary=(
                f"Approved {approved_count} fight(s); retrained {model_count} models "
                f"({len(MODEL_NAMES)} model types x {len(TARGETS)} targets x {len(MODEL_VARIANTS)} variants)."
            ),
        )
    except Exception as exc:
        _status = RetrainStatus(
            status="error",
            started_at=_status.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error_message=f"{exc}\n{traceback.format_exc()}",
        )
