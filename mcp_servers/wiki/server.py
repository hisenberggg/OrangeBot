"""
Syracuse Answers wiki MCP server: CQL search, fetch page, retrieve (search + fetch + chunk).
Run: python -m mcp_servers.wiki.server
"""
import json
from typing import Any

from fastmcp import FastMCP

from mcp_servers.wiki import config as wiki_config
from mcp_servers.wiki.confluence_client import fetch_page, search_cql
from mcp_servers.wiki.chunker import chunk_page

mcp = FastMCP(
    name="Syracuse Answers Wiki",
    instructions="Search and retrieve content from Syracuse University Answers (Confluence) wiki.",
)


@mcp.tool(description="Search Syracuse Answers wiki by CQL. Returns page IDs, titles, space keys, and web UI links.")
def answers_search_cql(
    query: str,
    limit: int = 5,
    cursor: str | None = None,
) -> str:
    """Search the Answers wiki with CQL. Use cursor for pagination (from previous nextCursorUrl)."""
    limit = limit or wiki_config.DEFAULT_SEARCH_LIMIT
    out = search_cql(query=query, limit=limit, cursor=cursor)
    return json.dumps(out, indent=2)


@mcp.tool(description="Fetch full Confluence page by ID. Returns body HTML and metadata.")
def answers_fetch_page(page_id: str) -> str:
    """Fetch a single page by its Confluence page ID."""
    data = fetch_page(page_id)
    return json.dumps(data, indent=2)


@mcp.tool(
    description="Search and return evidence snippets for RAG: runs CQL search, fetches page bodies, chunks by headings, returns structured snippets with source, section, and snippet text.",
)
def answers_retrieve(query: str, top_k: int = 5) -> str:
    """Retrieve evidence snippets from Answers wiki for a query. Best for procedures, how-to, policy questions."""
    top_k = top_k or wiki_config.DEFAULT_TOP_K
    search_out = search_cql(query=query, limit=top_k, cursor=None)
    results = search_out.get("results") or []
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
        if isinstance(webui, dict):
            webui = webui.get("href") or webui.get("url") or str(webui)
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
    return json.dumps(evidence, indent=2)


if __name__ == "__main__":
    mcp.run()
