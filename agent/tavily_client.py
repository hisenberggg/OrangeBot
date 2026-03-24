"""
Tavily search + extract: up to N URLs from search, then batch extract for LLM context.
Blocking SDK calls run via asyncio.to_thread from async graph nodes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


def _format_extract_block(url: str, title: str, body: str) -> str:
    header = title or url
    return f"### {header}\nSource: {url}\n\n{body.strip()}\n"


def _parts_from_search_snippets(
    search: dict[str, Any],
    title_by_url: dict[str, str],
    max_chars: int,
) -> list[str]:
    """Use Tavily search result snippets when full page extract yields nothing."""
    parts: list[str] = []
    total = 0
    for r in search.get("results") or []:
        if not isinstance(r, dict):
            continue
        u = (r.get("url") or "").strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        if not snippet or not u:
            continue
        title = title_by_url.get(u) or (r.get("title") or "").strip() or u
        block = _format_extract_block(u, title, f"[Search snippet]\n{snippet}")
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(block[:remain] + "\n[truncated]\n")
            break
        parts.append(block)
        total += len(block)
    return parts


def fetch_web_context_sync(query: str) -> str:
    """
    Run Tavily search then extract on result URLs. Returns markdown-ish text or "".
    """
    key = getattr(settings, "tavily_api_key", "") or ""
    if not key.strip():
        return ""

    max_results = max(1, min(10, getattr(settings, "tavily_max_results", 5) or 5))
    max_chars = max(2000, getattr(settings, "tavily_context_max_chars", 12000) or 12000)

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not installed; skipping web context")
        return ""

    try:
        client = TavilyClient(api_key=key)
        logger.info("Tavily web search query (exact): %r", query)
        search: dict[str, Any] = client.search(
            query,
            max_results=max_results,
            search_depth="advanced",
            timeout=60.0,
        )
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return ""

    urls: list[str] = []
    title_by_url: dict[str, str] = {}
    for r in search.get("results") or []:
        if not isinstance(r, dict):
            continue
        u = (r.get("url") or "").strip()
        if not u or u in urls:
            continue
        urls.append(u)
        title_by_url[u] = (r.get("title") or "").strip()

    if not urls:
        return ""

    try:
        extracted: dict[str, Any] = client.extract(
            urls=urls,
            query=query,
            extract_depth="advanced",
            format="markdown",
            timeout=60.0,
        )
    except Exception as exc:
        logger.warning("Tavily extract failed: %s", exc)
        fb = _parts_from_search_snippets(search, title_by_url, max_chars)
        if fb:
            header = (
                "--- WEB CONTEXT (third-party sources; verify; not official SU policy) ---\n"
            )
            return header + "\n".join(fb)
        return ""

    parts: list[str] = []
    total = 0
    for item in extracted.get("results") or []:
        if not isinstance(item, dict):
            continue
        u = (item.get("url") or "").strip()
        raw = (
            item.get("raw_content")
            or item.get("content")
            or item.get("markdown")
            or ""
        )
        if isinstance(raw, dict):
            raw = str(raw)
        raw = str(raw).strip()
        if not raw:
            continue
        title = title_by_url.get(u, u)
        block = _format_extract_block(u, title, raw)
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                block = block[:remain] + "\n[truncated]\n"
                parts.append(block)
            break
        parts.append(block)
        total += len(block)

    if not parts:
        fb = _parts_from_search_snippets(search, title_by_url, max_chars)
        if fb:
            parts = fb
        else:
            return ""

    header = (
        "--- WEB CONTEXT (third-party sources; verify; not official SU policy) ---\n"
    )
    return header + "\n".join(parts)


async def fetch_web_context(query: str) -> str:
    """Async wrapper: runs sync Tavily client in a worker thread."""
    return await asyncio.to_thread(fetch_web_context_sync, query)
