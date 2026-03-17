"""
Wiki RAG indexer: list Confluence pages, fetch, chunk, embed, and store in Chroma.
Run periodically (cron/scheduled) or on-demand. Set OPENAI_API_KEY and VECTOR_STORE_PATH (optional).
Usage: python -m mcp_servers.wiki.indexer [--incremental]  (default: full reindex, clears collection first)
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from mcp_servers.wiki import config as wiki_config
from mcp_servers.wiki.chunker import chunk_page
from mcp_servers.wiki.confluence_client import fetch_page, list_pages
from mcp_servers.wiki.vector_store import add_chunks, clear_vector_store, get_vector_store

PAGE_BATCH = 25
CHUNK_BATCH = 200


def _normalize_webui(webui: Any, item: dict[str, Any]) -> str:
    if isinstance(webui, dict):
        webui = webui.get("href") or webui.get("url") or ""
    if not webui and item.get("webuiUrl"):
        w = item["webuiUrl"]
        if isinstance(w, dict):
            w = w.get("href") or w.get("url") or ""
        return str(w) if w else ""
    return str(webui) if webui else ""


def run_indexer(clear_first: bool = True, persist_directory: str | None = None) -> None:
    persist = persist_directory or wiki_config.VECTOR_STORE_PATH
    if not persist:
        print("VECTOR_STORE_PATH is not set; using in-memory Chroma (data will not persist).", file=sys.stderr)
    if clear_first:
        clear_vector_store(persist_directory=persist)
        print("Cleared existing vector collection.")
    print("Loading vector store...", flush=True)
    store = get_vector_store(persist_directory=persist)
    print(f"Vector store ready (persist_dir={persist}).", flush=True)
    start = 0
    total_chunks = 0
    total_pages = 0
    chunk_batch: list[dict[str, Any]] = []
    while True:
        print(f"Fetching page list from Confluence (start={start}, limit={PAGE_BATCH})...", flush=True)
        out = list_pages(limit=PAGE_BATCH, start=start)
        results = out.get("results") or []
        if not results:
            print("No more pages.", flush=True)
            break
        print(f"Got {len(results)} pages.", flush=True)
        for i, item in enumerate(results, 1):
            pid = item.get("id")
            if not pid:
                continue
            title = (item.get("title") or "")[:50]
            print(f"  [{total_pages + i}] Fetching page {pid}: {title!r}...", flush=True)
            try:
                page = fetch_page(str(pid))
            except Exception as e:
                print(f"  Skip page {pid}: {e}", file=sys.stderr, flush=True)
                continue
            total_pages += 1
            webui = page.get("webuiUrl") or ""
            webui = _normalize_webui(webui, item)
            chunks = chunk_page(
                page.get("bodyHtml") or "",
                page_id=str(pid),
                title=page.get("title") or item.get("title") or "",
                space_key=page.get("spaceKey") or item.get("spaceKey") or "",
                space_name=page.get("spaceName") or "",
                updated=page.get("updated") or "",
                url=webui,
            )
            chunk_batch.extend(chunks)
            print(f"    -> {len(chunks)} chunks (batch size now {len(chunk_batch)}).", flush=True)
            while len(chunk_batch) >= CHUNK_BATCH:
                print(f"  Adding {CHUNK_BATCH} chunks to vector store...", flush=True)
                add_chunks(store, chunk_batch[:CHUNK_BATCH], start_id=total_chunks)
                total_chunks += CHUNK_BATCH
                chunk_batch = chunk_batch[CHUNK_BATCH:]
                print(f"    Total chunks in index: {total_chunks}.", flush=True)
        next_start = out.get("next_start")
        if next_start is None or next_start <= start:
            print("Reached end of page list.", flush=True)
            break
        start = next_start
    if chunk_batch:
        print(f"Adding final batch of {len(chunk_batch)} chunks to vector store...", flush=True)
        add_chunks(store, chunk_batch, start_id=total_chunks)
        total_chunks += len(chunk_batch)
    print(f"Done. Indexed {total_pages} pages, {total_chunks} chunks.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Syracuse Answers wiki into vector store.")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Do not clear collection first (appends; for future incremental use).",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Override VECTOR_STORE_PATH for Chroma persistence.",
    )
    args = parser.parse_args()
    run_indexer(clear_first=not args.incremental, persist_directory=args.persist_dir)


if __name__ == "__main__":
    main()
