"""
Wiki MCP config: reads from central config or env. No hardcoded values.
"""
import os

try:
    from config.settings import settings
    CONFLUENCE_BASE_URL = getattr(settings, "confluence_base_url", "") or os.environ.get(
        "CONFLUENCE_BASE_URL", "https://answers.atlassian.syr.edu/wiki/rest/api/content"
    )
    DEFAULT_SEARCH_LIMIT = getattr(settings, "default_search_limit", None) or int(
        os.environ.get("DEFAULT_SEARCH_LIMIT", "5")
    )
    DEFAULT_TOP_K = getattr(settings, "default_top_k", None) or int(
        os.environ.get("DEFAULT_TOP_K", "5")
    )
    CHUNK_MAX_WORDS = getattr(settings, "chunk_max_words", None) or int(
        os.environ.get("CHUNK_MAX_WORDS", "1200")
    )
    CHUNK_MIN_WORDS = getattr(settings, "chunk_min_words", None) or int(
        os.environ.get("CHUNK_MIN_WORDS", "500")
    )
except ImportError:
    CONFLUENCE_BASE_URL = os.environ.get(
        "CONFLUENCE_BASE_URL", "https://answers.atlassian.syr.edu/wiki/rest/api/content"
    )
    DEFAULT_SEARCH_LIMIT = int(os.environ.get("DEFAULT_SEARCH_LIMIT", "5"))
    DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))
    CHUNK_MAX_WORDS = int(os.environ.get("CHUNK_MAX_WORDS", "1200"))
    CHUNK_MIN_WORDS = int(os.environ.get("CHUNK_MIN_WORDS", "500"))
