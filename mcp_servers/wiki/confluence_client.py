"""
Confluence REST client: CQL search and fetch page by ID.
Uses anonymous access; base URL and limits from wiki config.
"""
import urllib.parse
from typing import Any

import httpx

from mcp_servers.wiki import config as wiki_config


def _base() -> str:
    return wiki_config.CONFLUENCE_BASE_URL.rstrip("/")


def list_pages(limit: int = 25, start: int = 0) -> dict[str, Any]:
    """
    List Confluence pages (type=page) with pagination. Use for indexing.
    Returns results and total size; use start+limit for next page.
    """
    url = f"{_base()}?type=page&limit={limit}&start={start}&expand=version,space,_links"
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for item in data.get("results", []):
        version = item.get("version") or {}
        space = item.get("space") or {}
        links = item.get("_links") or {}
        webui = links.get("webui") or ""
        if isinstance(webui, dict):
            webui = webui.get("href") or webui.get("url") or ""
        results.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "spaceKey": space.get("key"),
            "spaceName": space.get("name"),
            "updated": version.get("when"),
            "webuiUrl": webui,
        })
    return {
        "results": results,
        "size": data.get("size", len(results)),
        "next_start": start + len(results) if len(results) >= limit else None,
    }


def search_cql(
    query: str,
    limit: int = 5,
    cursor: str | None = None,
) -> dict[str, Any]:
    """
    Search Confluence via CQL. Returns results and optional nextCursorUrl.
    """
    cql = f'type = page AND siteSearch ~ "{query}"'
    encoded_cql = urllib.parse.quote(cql)
    if cursor:
        url = cursor
    else:
        url = f"{_base()}/search?cql={encoded_cql}&limit={limit}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for item in data.get("results", []):
        results.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "spaceKey": item.get("space", {}).get("key"),
            "webuiUrl": (item.get("_links") or {}).get("webui"),
            "selfUrl": (item.get("_links") or {}).get("self"),
            "tinyUrl": (item.get("_links") or {}).get("tinyui"),
        })
    next_links = (data.get("_links") or {})
    next_cursor = next_links.get("next")
    if isinstance(next_cursor, str):
        next_cursor_url = next_cursor
    else:
        next_cursor_url = (next_cursor or {}).get("href") if next_cursor else None
    return {
        "results": results,
        "nextCursorUrl": next_cursor_url,
    }


def fetch_page(page_id: str) -> dict[str, Any]:
    """
    Fetch a single page by ID with body.view, version, space, _links.
    """
    url = f"{_base()}/{page_id}?expand=body.view,version,space,_links"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    body_view = (data.get("body") or {}).get("view") or {}
    version = data.get("version") or {}
    space = data.get("space") or {}
    links = data.get("_links") or {}
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "bodyHtml": body_view.get("value"),
        "updated": version.get("when"),
        "spaceKey": space.get("key"),
        "spaceName": space.get("name"),
        "webuiUrl": links.get("webui"),
    }
