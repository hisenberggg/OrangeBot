"""
Central configuration: all API keys and globals loaded from environment.
No hardcoded secrets or static URLs in source.
"""
import os
from pathlib import Path

# Load .env from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass


def _str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


# LLM
OPENAI_API_KEY = _str("OPENAI_API_KEY")

# LangSmith (tracing; LangChain/LangGraph read these from env when set)
LANGCHAIN_TRACING_V2 = _bool("LANGCHAIN_TRACING_V2", False)
LANGCHAIN_API_KEY = _str("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = _str("LANGCHAIN_PROJECT", "university-chat")

# Confluence / Syracuse Answers wiki
CONFLUENCE_BASE_URL = _str(
    "CONFLUENCE_BASE_URL",
    "https://answers.atlassian.syr.edu/wiki/rest/api/content",
)

# Wiki MCP defaults
DEFAULT_SEARCH_LIMIT = _int("DEFAULT_SEARCH_LIMIT", 10)
DEFAULT_TOP_K = _int("DEFAULT_TOP_K", 10)
CHUNK_MAX_WORDS = _int("CHUNK_MAX_WORDS", 1200)
CHUNK_MIN_WORDS = _int("CHUNK_MIN_WORDS", 500)

# Wiki RAG: pre-indexed vector store
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_STORE_PATH = _str("VECTOR_STORE_PATH", "").strip() or None
WIKI_USE_SEMANTIC_RETRIEVE = _bool("WIKI_USE_SEMANTIC_RETRIEVE", True)

# MCP server (for LangGraph spawning wiki MCP via stdio)
WIKI_MCP_COMMAND = _str("WIKI_MCP_COMMAND", "python")
WIKI_MCP_ARGS = _str("WIKI_MCP_ARGS", "-m,mcp_servers.wiki.server")


class _Settings:
    """Single namespace for all config; import as 'from config import settings'."""

    openai_api_key = OPENAI_API_KEY
    langchain_tracing_v2 = LANGCHAIN_TRACING_V2
    langchain_api_key = LANGCHAIN_API_KEY
    langchain_project = LANGCHAIN_PROJECT
    confluence_base_url = CONFLUENCE_BASE_URL
    default_search_limit = DEFAULT_SEARCH_LIMIT
    default_top_k = DEFAULT_TOP_K
    chunk_max_words = CHUNK_MAX_WORDS
    chunk_min_words = CHUNK_MIN_WORDS
    wiki_mcp_command = WIKI_MCP_COMMAND
    wiki_mcp_args = tuple(WIKI_MCP_ARGS.split(",")) if WIKI_MCP_ARGS else ("-m", "mcp_servers.wiki.server")
    embedding_model = EMBEDDING_MODEL
    vector_store_path = VECTOR_STORE_PATH
    wiki_use_semantic_retrieve = WIKI_USE_SEMANTIC_RETRIEVE


settings = _Settings()
