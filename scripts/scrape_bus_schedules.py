"""
Scrape Syracuse University campus shuttle and Centro bus schedules into Data/bus_schedules.json.

Fetches the main shuttle page, extracts PDF links, downloads each PDF,
extracts schedule data via pdfplumber, and writes structured JSON with
per-trip prose (stop-time pairs) for reliable LLM parsing.

Run:  python -m scripts.scrape_bus_schedules
"""
from __future__ import annotations

import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import pdfplumber
from bs4 import BeautifulSoup

MAIN_URL = "https://parking.syr.edu/transportation/shuttle-information/campus-shuttle-schedules/"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "Data" / "bus_schedules.json"

BOILERPLATE_PATTERNS = [
    re.compile(r"(?:Accessibility|Contact Centro|Title VI|Fares & Passes).*?(?=\n[A-Z]{2,}|\Z)", re.S),
    re.compile(r"Centro's policy is to fully comply.*?(?=\n[A-Z]|\Z)", re.S),
    re.compile(r"Get Next Bus.*?(?=\n[A-Z]|\Z)", re.S),
    re.compile(r"Access Real-Time.*?(?=\n[A-Z]|\Z)", re.S),
    re.compile(r"Download the official GoCentroBus.*?(?=\n[A-Z]|\Z)", re.S),
    re.compile(r"Choose the pass.*?(?=\n[A-Z]|\Z)", re.S),
    re.compile(r"Email:.*?(?=\n[A-Z]|\Z)", re.S),
]

TIME_RE = re.compile(r"\d{1,2}:\d{2}(?:\s*[APap][Mm])?")

KNOWN_STOPS: dict[str, list[str]] = {
    "Euclid Loop": [
        "College Place", "Genesee Irving", "Genesee Crouse",
        "Genesee Westcott", "Westcott Euclid", "College Place",
    ],
    "Orange Loop": [
        "Veterans Affairs", "National Resource Center",
        "Harrison & University Ave", "Comstock & Adams St",
        "University Pl", "College Place", "Flint Hall",
        "Shaw Hall (Euclid Ave)", "Bathgate", "Forestry",
        "Bona Van", "BBB", "Campus West", "Henry St Lot", "Lawrinson",
    ],
    "Blue Loop": [
        "Barnes Center", "Sims College Drive", "Flint Hall",
        "Shaw Hall (Euclid)", "Life Sciences", "Comstock & Adams",
        "Waverly Ave", "Walnut Harrison Lot",
        "Harrison & University Ave", "Veterans National Resource Center",
        "Quad Lot", "BBB Bona Van", "Campus West", "Henry St Lot",
    ],
    "South Campus Loop": [],
    "South Campus  Loop": [],
}

EMPTY_SCHEDULE_ROUTES = {"Comstock / Colvin St. Loop", "Warehouse Loop"}


def _fetch_html(url: str) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _download_pdf(url: str) -> bytes | None:
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        print(f"  Warning: could not download {url}: {e}", file=sys.stderr, flush=True)
        return None


def _is_garbled_header(row: list[str | None]) -> bool:
    """Detect garbled headers from vertically-oriented PDF text."""
    non_empty = [(c or "").strip() for c in row]
    non_empty = [c for c in non_empty if c]
    if not non_empty:
        return True
    avg_len = sum(len(c) for c in non_empty) / len(non_empty)
    whitespace_ratio = sum(c.count(" ") + c.count("\n") for c in non_empty) / max(sum(len(c) for c in non_empty), 1)
    return avg_len < 3 or whitespace_ratio > 0.4


def _is_time_row(row: list[str | None]) -> bool:
    """Check if a row contains mostly time values."""
    non_empty = [(c or "").strip() for c in row]
    non_empty = [c for c in non_empty if c and c != "-"]
    if not non_empty:
        return False
    time_count = sum(1 for c in non_empty if TIME_RE.search(c))
    return time_count >= len(non_empty) * 0.5


def _clean_header(cell: str) -> str:
    """Clean up a header cell by collapsing internal newlines."""
    return re.sub(r"\s+", " ", cell).strip()


def _extract_structured(pdf_bytes: bytes, route_name: str) -> dict[str, Any]:
    """Extract structured trip data from a PDF.

    Returns dict with 'stops', 'trips', 'notes', 'raw_text'.
    """
    tables: list[list[list[str]]] = []
    raw_text_parts: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)
            page_text = page.extract_text()
            if page_text:
                raw_text_parts.append(page_text)

    raw_text = "\n".join(raw_text_parts)

    known = KNOWN_STOPS.get(route_name)
    if known is not None and len(known) > 0:
        return _build_trips_with_known_stops(tables, known, raw_text)

    if route_name.startswith("South Campus") and tables:
        return _build_trips_from_clean_tables(tables, raw_text)

    return _build_centro_result(raw_text)


def _build_trips_with_known_stops(
    tables: list[list[list[str]]], stops: list[str], raw_text: str
) -> dict[str, Any]:
    """Build per-trip prose using hardcoded stop names and extracted time rows."""
    trips: list[str] = []
    num_stops = len(stops)

    for table in tables:
        for row in table:
            if not _is_time_row(row):
                continue
            cells = [(c or "").strip() for c in row]
            times = cells[:num_stops]
            if len(times) < num_stops:
                times.extend([""] * (num_stops - len(times)))

            parts: list[str] = []
            for stop, time_val in zip(stops, times):
                if time_val and time_val != "-":
                    parts.append(f"{stop} {time_val}")
            if parts:
                trips.append(" -> ".join(parts))

    notes = _extract_notes(raw_text)
    unique_stops = list(dict.fromkeys(s for s in stops if s))
    return {"stops": unique_stops, "trips": trips, "notes": notes, "raw_text": ""}


def _build_trips_from_clean_tables(
    tables: list[list[list[str]]], raw_text: str
) -> dict[str, Any]:
    """Build per-trip prose from tables with clean (parseable) headers."""
    all_trips: list[str] = []
    all_stops: list[str] = []

    for table in tables:
        if len(table) < 2:
            continue

        header_row = table[0]
        if _is_garbled_header(header_row) or _is_time_row(header_row):
            continue

        stops = [_clean_header(c or "") for c in header_row]
        if not all_stops:
            all_stops = [s for s in stops if s]

        for row in table[1:]:
            if not _is_time_row(row):
                continue
            cells = [(c or "").strip() for c in row]
            parts: list[str] = []
            for stop, time_val in zip(stops, cells):
                if stop and time_val and time_val != "-":
                    parts.append(f"{stop} {time_val}")
            if parts:
                all_trips.append(" -> ".join(parts))

    notes = _extract_notes(raw_text)
    return {"stops": all_stops, "trips": all_trips, "notes": notes, "raw_text": ""}


def _build_centro_result(raw_text: str) -> dict[str, Any]:
    """For Centro PDFs, strip boilerplate and return cleaned text as raw_text."""
    cleaned = raw_text
    for pat in BOILERPLATE_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return {"stops": [], "trips": [], "notes": "", "raw_text": cleaned}


def _extract_notes(raw_text: str) -> str:
    """Pull out schedule notes like days of operation, SUID requirements."""
    notes_parts: list[str] = []
    for pattern in [
        r"(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)[\s\-–]+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)",
        r"SUID\s+Required[^\n]*",
        r"Revised\s+\d+/\d+/\d+",
    ]:
        match = re.search(pattern, raw_text, re.I)
        if match:
            notes_parts.append(match.group(0).strip())
    return "; ".join(notes_parts) if notes_parts else ""


def _parse_schedule_links(html: str) -> list[dict[str, str]]:
    """Extract route entries with name, category, PDF url, and optional map url."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[dict[str, str]] = []
    current_category = ""

    content = soup.find("div", class_="entry-content") or soup.find("main") or soup
    for el in content.find_all(["h3", "h2", "li", "ul", "p"]):
        if el.name in ("h2", "h3"):
            heading = el.get_text(strip=True)
            if heading:
                current_category = heading
            continue

        if el.name == "li":
            links = el.find_all("a", href=True)
            pdf_url = ""
            map_url = ""
            name = ""

            for a in links:
                href = a["href"]
                link_text = a.get_text(strip=True).lower()
                is_pdf = ".pdf" in href.lower()
                if is_pdf:
                    if "map" in link_text:
                        map_url = href
                    else:
                        pdf_url = href
                        raw_name = a.get_text(strip=True)
                        if raw_name:
                            name = raw_name

            if not name:
                name = el.get_text(strip=True).split("–")[0].split("—")[0].strip()
                name = re.sub(r"\s*Map\s*$", "", name).strip()

            if pdf_url:
                if not pdf_url.startswith("http"):
                    pdf_url = urljoin(MAIN_URL, pdf_url)
                if map_url and not map_url.startswith("http"):
                    map_url = urljoin(MAIN_URL, map_url)

                entries.append({
                    "name": name,
                    "category": current_category,
                    "source_url": pdf_url,
                    "map_url": map_url,
                })

    return entries


def scrape() -> dict[str, Any]:
    print("Fetching shuttle schedules page...", flush=True)
    html = _fetch_html(MAIN_URL)
    entries = _parse_schedule_links(html)
    print(f"Found {len(entries)} route PDFs.", flush=True)

    routes: list[dict[str, Any]] = []
    for entry in entries:
        name = entry["name"]
        url = entry["source_url"]

        if name in EMPTY_SCHEDULE_ROUTES:
            print(f"  Skipping {name} (known empty/image-based PDF).", flush=True)
            routes.append({
                "name": name,
                "category": entry["category"],
                "stops": [],
                "trips": [],
                "notes": "[Schedule not available digitally. Check printed schedules on campus.]",
                "raw_text": "",
                "source_url": url,
                "map_url": entry.get("map_url", ""),
            })
            continue

        print(f"  Downloading: {name} ({url})...", flush=True)
        pdf_bytes = _download_pdf(url)
        if pdf_bytes is None:
            print(f"  Skipped (download failed).", flush=True)
            continue

        result = _extract_structured(pdf_bytes, name)
        trip_count = len(result["trips"])
        raw_len = len(result["raw_text"])

        routes.append({
            "name": name,
            "category": entry["category"],
            "stops": result["stops"],
            "trips": result["trips"],
            "notes": result["notes"],
            "raw_text": result["raw_text"],
            "source_url": url,
            "map_url": entry.get("map_url", ""),
        })
        print(f"  OK ({trip_count} trips, {raw_len} chars raw)", flush=True)

    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source_page": MAIN_URL,
        "routes": routes,
    }


def main() -> None:
    data = scrape()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    total_trips = sum(len(r.get("trips", [])) for r in data["routes"])
    print(f"Done. Wrote {len(data['routes'])} routes ({total_trips} trips) to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
