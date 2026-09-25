"""Archives and restores whole `models/` snapshots so a retrain that makes
things worse can be undone. Each archive is a plain recursive copy of the
models/ folder - the default (diff) variant's files at the top level plus
every experimental variant under models/variants/<name>/ - named by the
timestamp it was taken."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from app.config import DEFAULT_MODEL_VARIANT, MODEL_VARIANTS, MODELS_ARCHIVE_DIR, MODELS_DIR, TARGETS

from app.services import fighter_service, prediction_service


def _variant_dir(base_dir: Path, variant: str) -> Path:
    """Mirrors app.config.models_dir_for(), but rooted at an arbitrary base
    directory - MODELS_DIR for the current models, or an archived snapshot's
    own directory - instead of always MODELS_DIR."""
    if variant == DEFAULT_MODEL_VARIANT:
        return base_dir
    return base_dir / "variants" / variant


def _summarize_metrics(metrics: dict) -> dict:
    summary = {}
    for target in TARGETS:
        models = metrics.get("targets", {}).get(target, {}).get("models", {})
        summary[target] = {
            name: {"accuracy": m.get("accuracy"), "log_loss": m.get("log_loss")}
            for name, m in models.items()
        }
    return summary


def _summarize(base_dir: Path) -> dict:
    """Per-variant metrics summary for a models/ snapshot rooted at base_dir."""
    summary = {}
    for variant in MODEL_VARIANTS:
        metrics_path = _variant_dir(base_dir, variant) / "metrics.json"
        if not metrics_path.exists():
            continue
        summary[variant] = _summarize_metrics(json.loads(metrics_path.read_text()))
    return summary


def list_versions() -> list[dict]:
    versions = []

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        versions.append({
            "version_id": "current",
            "created_at": datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(),
            "is_current": True,
            "metrics_summary": _summarize(MODELS_DIR),
        })

    if MODELS_ARCHIVE_DIR.exists():
        for entry in sorted(MODELS_ARCHIVE_DIR.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            if not (entry / "metrics.json").exists():
                continue
            versions.append({
                "version_id": entry.name,
                "created_at": entry.name,
                "is_current": False,
                "metrics_summary": _summarize(entry),
            })

    return versions


def archive_current() -> str | None:
    """Recursively copies the current models/ folder (including variants/)
    into models_archive/<timestamp>/. Returns the new version id, or None if
    there's nothing trained yet."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None

    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = MODELS_ARCHIVE_DIR / version_id
    for item in MODELS_DIR.rglob("*"):
        if item.name == ".gitkeep" or not item.is_file():
            continue
        target = dest / item.relative_to(MODELS_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
    return version_id


def restore(version_id: str) -> None:
    src = MODELS_ARCHIVE_DIR / version_id
    if not src.exists() or not src.is_dir():
        raise KeyError(version_id)

    for item in src.rglob("*"):
        if item.is_file():
            target = MODELS_DIR / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

    fighter_service.reset_instance()
    prediction_service.reset_instance()
