"""
Canonical public URLs for Syracuse Answers (Confluence) wiki pages.

Confluence REST returns _links.webui as a relative path (e.g. /spaces/KEY/pages/...).
Downstream we must always expose absolute URLs under:

  https://answers.atlassian.syr.edu/wiki/spaces/...

so the model does not invent roots like https://answers.syr.edu/spaces/...
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

CANONICAL_NETLOC = "answers.atlassian.syr.edu"
WRONG_NETLOCS = frozenset(
    {
        "answers.syr.edu",
        "www.answers.syr.edu",
    }
)


def _default_api_base() -> str:
    from mcp_servers.wiki import config as wiki_config

    return wiki_config.CONFLUENCE_BASE_URL


def canonical_wiki_public_base(api_base: str | None = None) -> str:
    """
    Derive the browser wiki base from the REST API base URL.

    e.g. https://answers.atlassian.syr.edu/wiki/rest/api/content
         -> https://answers.atlassian.syr.edu/wiki
    """
    base = (api_base or _default_api_base()).rstrip("/")
    suffix = "/rest/api/content"
    if base.endswith(suffix):
        return base[: -len(suffix)]
    if "/wiki/" in base:
        idx = base.find("/wiki/")
        return base[: idx + len("/wiki")].rstrip("/")
    return f"https://{CANONICAL_NETLOC}/wiki"


def normalize_confluence_webui_url(webui: Any, api_base: str | None = None) -> str:
    """
    Return an absolute canonical wiki URL, or empty string if input is unusable.

    Accepts str, dict with href/url, or other types (coerced to str).
    """
    if webui is None:
        return ""

    if isinstance(webui, dict):
        webui = webui.get("href") or webui.get("url") or ""

    s = str(webui).strip()
    if not s or s == "None":
        return ""

    wiki_base = canonical_wiki_public_base(api_base).rstrip("/")

    # Relative path from API
    if s.startswith("/"):
        if s.startswith("/wiki"):
            return f"https://{CANONICAL_NETLOC}{s}"
        return f"{wiki_base}{s}"

    if not s.lower().startswith(("http://", "https://")):
        return s

    parsed = urlparse(s)
    scheme = parsed.scheme or "https"
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    tail = ""
    if parsed.query:
        tail += "?" + parsed.query
    if parsed.fragment:
        tail += "#" + parsed.fragment

    if host in WRONG_NETLOCS:
        if path.startswith("/wiki"):
            return f"{scheme}://{CANONICAL_NETLOC}{path}{tail}"
        if path.startswith("/spaces"):
            return f"{scheme}://{CANONICAL_NETLOC}/wiki{path}{tail}"
        return f"{scheme}://{CANONICAL_NETLOC}/wiki{path if path.startswith('/') else '/' + path}{tail}"

    if CANONICAL_NETLOC in host or host.endswith(".atlassian.syr.edu"):
        if path.startswith("/spaces"):
            return f"{scheme}://{CANONICAL_NETLOC}/wiki{path}{tail}"
        return f"{scheme}://{CANONICAL_NETLOC}{path}{tail}"

    return s
