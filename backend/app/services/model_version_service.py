"""Archives and restores whole `models/` snapshots so a retrain that makes
things worse can be undone. Each archive is a plain copy of the models/
folder (small - a handful of .joblib files + metrics.json), named by the
timestamp it was taken."""

import json
import shutil
from datetime import datetime

from app.config import MODELS_ARCHIVE_DIR, MODELS_DIR, TARGETS

from app.services import fighter_service, prediction_service


def _summarize(metrics: dict) -> dict:
    summary = {}
    for target in TARGETS:
        models = metrics.get("targets", {}).get(target, {}).get("models", {})
        summary[target] = {
            name: {"accuracy": m.get("accuracy"), "log_loss": m.get("log_loss")}
            for name, m in models.items()
        }
    return summary


def list_versions() -> list[dict]:
    versions = []

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        versions.append({
            "version_id": "current",
            "created_at": datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(),
            "is_current": True,
            "metrics_summary": _summarize(metrics),
        })

    if MODELS_ARCHIVE_DIR.exists():
        for entry in sorted(MODELS_ARCHIVE_DIR.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            archived_metrics_path = entry / "metrics.json"
            if not archived_metrics_path.exists():
                continue
            metrics = json.loads(archived_metrics_path.read_text())
            versions.append({
                "version_id": entry.name,
                "created_at": entry.name,
                "is_current": False,
                "metrics_summary": _summarize(metrics),
            })

    return versions


def archive_current() -> str | None:
    """Copies the current models/ folder into models_archive/<timestamp>/.
    Returns the new version id, or None if there's nothing trained yet."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None

    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = MODELS_ARCHIVE_DIR / version_id
    dest.mkdir(parents=True, exist_ok=True)
    for item in MODELS_DIR.iterdir():
        if item.name == ".gitkeep" or not item.is_file():
            continue
        shutil.copy2(item, dest / item.name)
    return version_id


def restore(version_id: str) -> None:
    src = MODELS_ARCHIVE_DIR / version_id
    if not src.exists() or not src.is_dir():
        raise KeyError(version_id)

    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, MODELS_DIR / item.name)

    fighter_service.reset_instance()
    prediction_service.reset_instance()
