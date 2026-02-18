"""
HTML to text and heading-based chunking for Confluence body.view.value.
Outputs list of dicts: source, pageId, title, spaceKey, spaceName, updated, url, section, snippet.
"""
import re
from typing import Any

from bs4 import BeautifulSoup

from mcp_servers.wiki import config as wiki_config

CHUNK_MAX_WORDS = wiki_config.CHUNK_MAX_WORDS
CHUNK_MIN_WORDS = wiki_config.CHUNK_MIN_WORDS


def _text_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_page(
    html: str,
    *,
    page_id: str,
    title: str,
    space_key: str = "",
    space_name: str = "",
    updated: str = "",
    url: str = "",
) -> list[dict[str, Any]]:
    """
    Split HTML into sections by h2/h3/h4; each section becomes a chunk with optional
    sub-splitting by paragraphs if over CHUNK_MAX_WORDS.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    chunks: list[dict[str, Any]] = []
    current_heading = ""
    current_parts: list[str] = []

    def flush_section():
        nonlocal current_heading, current_parts
        if not current_parts:
            return
        text = " ".join(current_parts)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            current_parts.clear()
            return
        word_count = _word_count(text)
        if word_count <= CHUNK_MAX_WORDS:
            chunks.append({
                "source": "answers",
                "pageId": page_id,
                "title": title,
                "spaceKey": space_key,
                "spaceName": space_name,
                "updated": updated,
                "url": url,
                "section": current_heading or title,
                "snippet": text,
            })
        else:
            # Split by paragraphs
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            acc = []
            acc_words = 0
            for p in paras:
                w = _word_count(p)
                if acc_words + w > CHUNK_MAX_WORDS and acc:
                    snippet = " ".join(acc)
                    chunks.append({
                        "source": "answers",
                        "pageId": page_id,
                        "title": title,
                        "spaceKey": space_key,
                        "spaceName": space_name,
                        "updated": updated,
                        "url": url,
                        "section": current_heading or title,
                        "snippet": snippet,
                    })
                    acc = []
                    acc_words = 0
                acc.append(p)
                acc_words += w
            if acc:
                chunks.append({
                    "source": "answers",
                    "pageId": page_id,
                    "title": title,
                    "spaceKey": space_key,
                    "spaceName": space_name,
                    "updated": updated,
                    "url": url,
                    "section": current_heading or title,
                    "snippet": " ".join(acc),
                })
        current_parts.clear()

    for elem in soup.find_all(["h2", "h3", "h4", "p", "li", "td", "th"]):
        name = elem.name
        text = elem.get_text(separator=" ", strip=True)
        if not text:
            continue
        if name in ("h2", "h3", "h4"):
            flush_section()
            current_heading = text
            current_parts = [text]
        else:
            current_parts.append(text)
    flush_section()

    return chunks
