"""Scrapes new UFC events/fights from ufcstats.com - the same site the
original dataset came from (hence Fighter_URL/Fight_URL columns).

ufcstats.com only serves plain HTTP and runs a JavaScript proof-of-work
anti-bot challenge in front of every page; a plain HTTP client (requests/
httpx) only ever sees the challenge page. A real (headless) browser executes
the challenge automatically, which is why this module drives Playwright
instead of a lightweight HTTP library - confirmed by hand during planning.

Matches fighters by Fighter_URL (not name) to sidestep the duplicate-name
problem described in the fighter dataset (a handful of names belong to two
different real fighters). If a fighter's URL isn't in our existing roster,
their fighter-details page is scraped too so they can be added as a debutant.
"""

import re
from datetime import datetime
from typing import Optional

import pandas as pd
from playwright.async_api import Page, async_playwright

from app.config import SCRAPE_EVENTS_URL
from app.ml.data_pipeline import (
    is_title_fight,
    normalize_method,
    normalize_weight_class,
    parse_height_to_inches,
    parse_reach_to_inches,
    parse_weight_lbs,
)

NAV_TIMEOUT_MS = 30000
NAV_DELAY_MS = 1500

TOTALS_TABLE_COLUMN = {
    "kd": 1,
    "sig_str": 2,
    "td": 5,
    "sub_att": 7,
    "ctrl": 9,
}


def fighter_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _parse_landed_of_attempted(text: str) -> tuple[Optional[float], Optional[float]]:
    m = re.match(r"(\d+)\s+of\s+(\d+)", text.strip())
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _parse_ctrl_seconds(text: str) -> Optional[float]:
    text = text.strip()
    m = re.match(r"(\d+):(\d+)", text)
    if not m:
        return None
    minutes, seconds = int(m.group(1)), int(m.group(2))
    return float(minutes * 60 + seconds)


def _parse_number(text: str) -> Optional[float]:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


async def _new_page(browser) -> Page:
    context = await browser.new_context(user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    ))
    return await context.new_page()


async def fetch_events_list(page: Page) -> list[dict]:
    """Returns [{url, name, date}] for every event listed, newest first."""
    await page.goto(SCRAPE_EVENTS_URL, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    await page.wait_for_timeout(NAV_DELAY_MS)
    return await page.eval_on_selector_all(
        "tr.b-statistics__table-row",
        """rows => rows.map(r => {
            const a = r.querySelector('a[href*="event-details"]');
            const dateEl = r.querySelector('.b-statistics__date');
            if (!a) return null;
            return { url: a.href, name: a.textContent.trim(), date: dateEl ? dateEl.textContent.trim() : null };
        }).filter(Boolean)""",
    )


async def fetch_event_fight_urls(page: Page, event_url: str) -> list[str]:
    await page.goto(event_url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    await page.wait_for_timeout(NAV_DELAY_MS)
    urls = await page.eval_on_selector_all(
        "tr.b-fight-details__table-row[data-link]",
        "rows => rows.map(r => r.getAttribute('data-link'))",
    )
    return [u for u in urls if u]


async def fetch_fighter_info(page: Page, fighter_url: str) -> dict:
    await page.goto(fighter_url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    await page.wait_for_timeout(NAV_DELAY_MS)
    raw = await page.evaluate("""
        () => {
            const name = document.querySelector('.b-content__title-highlight')?.textContent.trim();
            const items = Array.from(document.querySelectorAll('.b-list__box-list-item')).map(
                e => e.textContent.replace(/\\s+/g, ' ').trim()
            );
            return { name, items };
        }
    """)
    fields = {}
    for item in raw["items"]:
        if ":" not in item:
            continue
        label, _, value = item.partition(":")
        fields[label.strip().upper()] = value.strip()

    return {
        "name": raw["name"],
        "height_in": parse_height_to_inches(fields.get("HEIGHT", "")),
        "reach_in": parse_reach_to_inches(fields.get("REACH", "")),
        "weight_lbs": parse_weight_lbs(fields.get("WEIGHT", "")),
        "stance": fields.get("STANCE") or None,
        "dob": fields.get("DOB") or None,
    }


async def fetch_fight_details(page: Page, fight_url: str) -> Optional[dict]:
    """Returns a dict shaped like schemas.PendingFightCreate, or None if the
    fight has no clean winner/method (draw, no-contest, DQ, overturned...) -
    the same filter clean_fights() applies to the original dataset."""
    await page.goto(fight_url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    await page.wait_for_timeout(NAV_DELAY_MS)

    raw = await page.evaluate("""
        () => {
            const persons = Array.from(document.querySelectorAll('.b-fight-details__person')).map(p => ({
                name: p.querySelector('.b-fight-details__person-name')?.textContent.trim(),
                status: p.querySelector('.b-fight-details__person-status')?.textContent.trim(),
            }));
            const weightClass = document.querySelector('.b-fight-details__fight-head')?.textContent.replace(/\\s+/g, ' ').trim();
            const methodItem = document.querySelector('.b-fight-details__text-item_first');
            const methodText = methodItem ? methodItem.textContent.replace(/\\s+/g, ' ').trim() : '';

            const table = document.querySelectorAll('table')[0];
            const row = table ? table.querySelectorAll('tbody tr')[0] : null;
            const cells = row ? Array.from(row.querySelectorAll('td')).map(
                td => Array.from(td.querySelectorAll('p')).map(p => p.textContent.trim())
            ) : [];
            const fighterLinks = row
                ? Array.from(row.querySelectorAll('td')[0].querySelectorAll('a[href*="fighter-details"]')).map(a => a.href)
                : [];

            return { persons, weightClass, methodText, cells, fighterLinks };
        }
    """)

    if len(raw["persons"]) != 2 or len(raw["fighterLinks"]) != 2 or len(raw["cells"]) < 10:
        return None  # unexpected page shape - skip rather than guess

    method_raw = raw["methodText"].replace("Method:", "").strip()
    method = normalize_method(method_raw)
    if method is None:
        return None  # DQ / Overturned / Could Not Continue / Other - not usable

    statuses = [p["status"] for p in raw["persons"]]
    if statuses[0] == "W":
        winner_name = raw["persons"][0]["name"]
    elif statuses[1] == "W":
        winner_name = raw["persons"][1]["name"]
    else:
        return None  # draw / no contest

    cells = raw["cells"]
    f1_kd, f2_kd = (_parse_number(v) for v in cells[TOTALS_TABLE_COLUMN["kd"]])
    (f1_sig_landed, f1_sig_att), (f2_sig_landed, f2_sig_att) = (
        _parse_landed_of_attempted(cells[TOTALS_TABLE_COLUMN["sig_str"]][0]),
        _parse_landed_of_attempted(cells[TOTALS_TABLE_COLUMN["sig_str"]][1]),
    )
    (f1_td_landed, f1_td_att), (f2_td_landed, f2_td_att) = (
        _parse_landed_of_attempted(cells[TOTALS_TABLE_COLUMN["td"]][0]),
        _parse_landed_of_attempted(cells[TOTALS_TABLE_COLUMN["td"]][1]),
    )
    f1_sub_att, f2_sub_att = (_parse_number(v) for v in cells[TOTALS_TABLE_COLUMN["sub_att"]])
    f1_ctrl_sec, f2_ctrl_sec = (_parse_ctrl_seconds(v) for v in cells[TOTALS_TABLE_COLUMN["ctrl"]])

    return {
        "source_url": fight_url,
        "fighter1_name": raw["persons"][0]["name"],
        "fighter2_name": raw["persons"][1]["name"],
        "fighter1_url": raw["fighterLinks"][0],
        "fighter2_url": raw["fighterLinks"][1],
        "weight_class": normalize_weight_class(raw["weightClass"]),
        "is_title_fight": is_title_fight(raw["weightClass"]),
        "method": method,
        "winner_name": winner_name,
        "f1_kd": f1_kd, "f2_kd": f2_kd,
        "f1_sig_landed": f1_sig_landed, "f1_sig_att": f1_sig_att,
        "f2_sig_landed": f2_sig_landed, "f2_sig_att": f2_sig_att,
        "f1_td_landed": f1_td_landed, "f1_td_att": f1_td_att,
        "f2_td_landed": f2_td_landed, "f2_td_att": f2_td_att,
        "f1_sub_att": f1_sub_att, "f2_sub_att": f2_sub_att,
        "f1_ctrl_sec": f1_ctrl_sec, "f2_ctrl_sec": f2_ctrl_sec,
    }


async def scrape_new_fights(latest_known_date, existing_fighter_urls: set[str]) -> list[dict]:
    """Top-level entry point. `latest_known_date` is a pandas/py Timestamp -
    only events strictly after it are scraped. `existing_fighter_urls` is the
    set of Fighter_URL values already present in fighters_clean, used to
    detect and fetch info for debutants. Returns a list of dicts shaped like
    schemas.PendingFightCreate (as plain dicts - the caller validates them).
    """
    results = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await _new_page(browser)
            events = await fetch_events_list(page)

            new_events = []
            for event in events:
                event_date = _parse_event_date(event["date"])
                if event_date is not None and event_date > latest_known_date:
                    new_events.append({**event, "iso_date": event_date.strftime("%Y-%m-%d")})

            seen_fighter_urls = set(existing_fighter_urls)

            for event in new_events:
                try:
                    fight_urls = await fetch_event_fight_urls(page, event["url"])
                except Exception as exc:
                    print(f"[scraper] failed to load event {event['url']}: {exc}")
                    continue

                for fight_url in fight_urls:
                    try:
                        fight = await fetch_fight_details(page, fight_url)
                    except Exception as exc:
                        print(f"[scraper] failed to parse fight {fight_url}: {exc}")
                        continue
                    if fight is None:
                        continue

                    fight["event_date"] = event["iso_date"]

                    for side in ("1", "2"):
                        url = fight[f"fighter{side}_url"]
                        if url not in seen_fighter_urls:
                            try:
                                info = await fetch_fighter_info(page, url)
                                fight[f"new_fighter_{side}"] = info
                            except Exception as exc:
                                print(f"[scraper] failed to fetch fighter {url}: {exc}")
                            seen_fighter_urls.add(url)

                    fight["fighter1_id"] = fighter_id_from_url(fight.pop("fighter1_url"))
                    fight["fighter2_id"] = fighter_id_from_url(fight.pop("fighter2_url"))
                    results.append(fight)
        finally:
            await browser.close()

    return results


def _parse_event_date(text: Optional[str]):
    if not text:
        return None
    try:
        return pd.Timestamp(datetime.strptime(text.strip(), "%B %d, %Y"))
    except ValueError:
        return None
