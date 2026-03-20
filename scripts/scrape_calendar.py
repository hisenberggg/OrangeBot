"""
Scrape Syracuse University academic calendar pages into Data/calendar.json.

Fetches:
  - Academic Year Calendar (detailed per-semester dates for current + next year)
  - Five-Year Calendar (key dates from Fall 2025 through Summer 2030)
  - Main calendar page (degree award dates)

Run:  python -m scripts.scrape_calendar
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag

URLS = {
    "academic_year": "https://www.syracuse.edu/academics/calendars/academic-year/",
    "five_year": "https://www.syracuse.edu/academics/calendars/five-year/",
    "main": "https://www.syracuse.edu/academics/calendars/",
}

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "Data" / "calendar.json"


def _fetch(url: str) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_two_col_tables(html: str, source_url: str) -> list[dict[str, str]]:
    """Parse tables with two columns: event | date, grouped under h2 headings."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[dict[str, str]] = []
    current_section = ""

    content = soup.find("main") or soup
    for el in content.find_all(["h2", "h3", "h4", "table"]):
        if el.name in ("h2", "h3", "h4"):
            current_section = _clean(el.get_text())
            continue

        if not isinstance(el, Tag) or el.name != "table":
            continue

        rows = el.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                event = _clean(cells[0].get_text())
                date = _clean(cells[1].get_text())
                if event and date:
                    entries.append({
                        "semester": current_section,
                        "event": event,
                        "date": date,
                        "source_url": source_url,
                    })

    return entries


def _parse_multi_col_tables(html: str, source_url: str) -> list[dict[str, str]]:
    """Parse tables with year columns (five-year calendar format)."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[dict[str, str]] = []
    current_section = ""

    content = soup.find("main") or soup
    for el in content.find_all(["h2", "h3", "h4", "table"]):
        if el.name in ("h2", "h3", "h4"):
            current_section = _clean(el.get_text())
            continue

        if not isinstance(el, Tag) or el.name != "table":
            continue

        rows = el.find_all("tr")
        if not rows:
            continue

        header_cells = rows[0].find_all(["th", "td"])
        years = [_clean(c.get_text()) for c in header_cells]

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            event = _clean(cells[0].get_text())
            for i, cell in enumerate(cells[1:], 1):
                date = _clean(cell.get_text())
                year_label = years[i] if i < len(years) else ""
                if event and date:
                    semester = f"{current_section} {year_label}".strip() if year_label else current_section
                    entries.append({
                        "semester": semester,
                        "event": event,
                        "date": date,
                        "source_url": source_url,
                    })

    return entries


def _parse_degree_awards(html: str, source_url: str) -> list[dict[str, str]]:
    """Parse the degree award dates table from the main calendar page."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[dict[str, str]] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = rows[0].find_all(["th", "td"])
        years = [_clean(c.get_text()) for c in header_cells]

        if not any("20" in y for y in years):
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            for i, cell in enumerate(cells):
                date = _clean(cell.get_text())
                year_label = years[i] if i < len(years) else ""
                if date and year_label:
                    entries.append({
                        "semester": f"Degree Awards {year_label}",
                        "event": "Degree Award Date",
                        "date": date,
                        "source_url": source_url,
                    })

    return entries


def scrape() -> dict[str, Any]:
    print("Fetching academic year calendar...", flush=True)
    ay_html = _fetch(URLS["academic_year"])
    academic_year = _parse_two_col_tables(ay_html, URLS["academic_year"])
    print(f"  -> {len(academic_year)} entries", flush=True)

    print("Fetching five-year calendar...", flush=True)
    fy_html = _fetch(URLS["five_year"])
    five_year = _parse_multi_col_tables(fy_html, URLS["five_year"])
    print(f"  -> {len(five_year)} entries", flush=True)

    print("Fetching degree award dates...", flush=True)
    main_html = _fetch(URLS["main"])
    degree_awards = _parse_degree_awards(main_html, URLS["main"])
    print(f"  -> {len(degree_awards)} entries", flush=True)

    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "calendars": {
            "academic_year": academic_year,
            "five_year": five_year,
            "degree_awards": degree_awards,
        },
    }


def main() -> None:
    data = scrape()
    total = sum(len(v) for v in data["calendars"].values())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Wrote {total} entries to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
