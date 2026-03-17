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

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip()
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "").strip() or None
WIKI_USE_SEMANTIC_RETRIEVE = os.environ.get("WIKI_USE_SEMANTIC_RETRIEVE", "true").strip().lower() in ("1", "true", "yes")

try:
    from config.settings import settings
    if getattr(settings, "embedding_model", None):
        EMBEDDING_MODEL = settings.embedding_model
    if getattr(settings, "vector_store_path", None) is not None:
        VECTOR_STORE_PATH = settings.vector_store_path
    if getattr(settings, "wiki_use_semantic_retrieve", None) is not None:
        WIKI_USE_SEMANTIC_RETRIEVE = settings.wiki_use_semantic_retrieve
except ImportError:
    pass
