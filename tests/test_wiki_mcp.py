"""Tests for Wiki MCP (confluence client and chunker)."""
import os
# Confluence URL for tests (anonymous access)
os.environ.setdefault("CONFLUENCE_BASE_URL", "https://answers.atlassian.syr.edu/wiki/rest/api/content")


def test_chunker_output_schema():
    """Chunker returns list of dicts with source, pageId, title, section, snippet."""
    from mcp_servers.wiki.chunker import chunk_page
    html = "<h2>Section One</h2><p>Some text here.</p>"
    chunks = chunk_page(
        html,
        page_id="123",
        title="Test Page",
        space_key="TEST",
        space_name="Test Space",
        updated="2025-01-01T00:00:00Z",
        url="https://example.com/page",
    )
    assert isinstance(chunks, list)
    if chunks:
        c = chunks[0]
        assert c.get("source") == "answers"
        assert "pageId" in c and "title" in c and "section" in c and "snippet" in c


def test_config_reads_env():
    """Wiki config reads CONFLUENCE_BASE_URL from env or default."""
    from mcp_servers.wiki import config as wiki_config
    assert wiki_config.CONFLUENCE_BASE_URL
    assert wiki_config.DEFAULT_SEARCH_LIMIT >= 1
    assert wiki_config.DEFAULT_TOP_K >= 1
