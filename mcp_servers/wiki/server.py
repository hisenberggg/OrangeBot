"""
Syracuse Answers wiki MCP server: CQL search, fetch page, retrieve (semantic from pre-index or CQL fallback).
Run: python -m mcp_servers.wiki.server
"""
import json
import os
import sys
from typing import Any

from fastmcp import FastMCP

from mcp_servers.wiki import config as wiki_config
from mcp_servers.wiki.confluence_client import fetch_page, search_cql
from mcp_servers.wiki.chunker import chunk_page
from mcp_servers.wiki.url_utils import normalize_confluence_webui_url

# chromadb / onnxruntime native code may write directly to fd 1 (C-level stdout)
# during import.  In an MCP stdio server fd 1 is the JSON-RPC channel, so any
# stray bytes corrupt the protocol and deadlock the client.
# Redirect fd 1 -> fd 2 (stderr) while loading the heavy vector-store module,
# then restore it before FastMCP starts listening.
_VS_AVAILABLE = False
get_vector_store = None
semantic_search = None

_orig_fd1 = os.dup(1)
os.dup2(2, 1)  # fd 1 now points to stderr
try:
    from mcp_servers.wiki.vector_store import (
        get_vector_store as _gvs,
        semantic_search as _ss,
    )
    get_vector_store = _gvs
    semantic_search = _ss
    _VS_AVAILABLE = True
except Exception:
    pass
finally:
    os.dup2(_orig_fd1, 1)  # restore real stdout
    os.close(_orig_fd1)

mcp = FastMCP(
    name="Syracuse Answers Wiki",
    instructions="Search and retrieve content from Syracuse University Answers (Confluence) wiki.",
)


@mcp.tool(
    description="Search and return evidence snippets for RAG: uses pre-indexed vector search when available (fast), otherwise falls back to CQL search + fetch + chunk. Returns structured snippets with source, section, and snippet text.",
)
def answers_retrieve(query: str, top_k: int = 5) -> str:
    """Retrieve evidence snippets from Answers wiki for a query. Best for procedures, how-to, policy questions."""
    top_k = top_k or wiki_config.DEFAULT_TOP_K
    if wiki_config.WIKI_USE_SEMANTIC_RETRIEVE and _VS_AVAILABLE:
        try:
            store = get_vector_store()
            evidence = semantic_search(store, query, top_k=top_k)
            if evidence:
                return json.dumps(evidence, indent=2)
        except Exception:
            pass
    search_out = search_cql(query=query, limit=top_k, cursor=None)
    results = search_out.get("results") or []
    evidence = _retrieve_via_cql(results)
    return json.dumps(evidence, indent=2)


def _retrieve_via_cql(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Original retrieve path: fetch each page and chunk."""
    evidence: list[dict[str, Any]] = []
    for item in results:
        pid = item.get("id")
        if not pid:
            continue
        try:
            page = fetch_page(str(pid))
        except Exception:
            continue
        webui = page.get("webuiUrl") or ""
        if not webui and item.get("webuiUrl"):
            webui = item["webuiUrl"]
        webui = normalize_confluence_webui_url(webui)
        chunks = chunk_page(
            page.get("bodyHtml") or "",
            page_id=str(pid),
            title=page.get("title") or item.get("title") or "",
            space_key=page.get("spaceKey") or item.get("spaceKey") or "",
            space_name=page.get("spaceName") or "",
            updated=page.get("updated") or "",
            url=webui,
        )
        evidence.extend(chunks)
    return evidence


if __name__ == "__main__":
    mcp.run()
