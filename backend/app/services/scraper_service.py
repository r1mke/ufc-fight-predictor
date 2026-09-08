"""Orchestrates a scrape run: figures out the cutoff date and the set of
already-known fighters, drives app.ml.scraper, and stores results as pending
fights via submission_service."""

import threading
import traceback
from datetime import datetime, timezone

import pandas as pd

from app.config import FIGHTERS_CLEAN_PARQUET, FIGHTS_CLEAN_PARQUET
from app.ml.scraper import scrape_new_fights
from app.schemas import ScrapeStatus
from app.services import submission_service

_status = ScrapeStatus(status="idle")
_lock = threading.Lock()


def get_status() -> ScrapeStatus:
    return _status


def is_running() -> bool:
    return _status.status == "running"


async def run_scrape() -> None:
    global _status

    with _lock:
        if _status.status == "running":
            raise RuntimeError("a scrape is already running")
        _status = ScrapeStatus(status="running", started_at=datetime.now(timezone.utc).isoformat())

    try:
        fights = pd.read_parquet(FIGHTS_CLEAN_PARQUET)
        fighters = pd.read_parquet(FIGHTERS_CLEAN_PARQUET)
        latest_known_date = fights["event_date"].max()
        existing_fighter_urls = set(fighters["fighter_url"])

        results = await scrape_new_fights(latest_known_date, existing_fighter_urls)
        added = submission_service.add_scraped_batch(results)

        _status = ScrapeStatus(
            status="done",
            started_at=_status.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            result_summary=f"Found {len(results)} new fight(s), {added} added to the review queue.",
        )
    except Exception as exc:
        _status = ScrapeStatus(
            status="error",
            started_at=_status.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error_message=f"{exc}\n{traceback.format_exc()}",
        )
