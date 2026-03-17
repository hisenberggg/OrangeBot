"""
Vector store for wiki RAG: Chroma + OpenAI embeddings.
Used by the indexer to add chunk embeddings and by the MCP server for semantic retrieval.
"""
from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback if langchain-chroma not installed

from mcp_servers.wiki import config as wiki_config

COLLECTION_NAME = "syracuse_answers_chunks"
BATCH_SIZE = 100


def _text_to_embed(chunk: dict[str, Any]) -> str:
    """Build a single string for embedding from chunk fields."""
    title = chunk.get("title") or ""
    section = chunk.get("section") or ""
    snippet = chunk.get("snippet") or ""
    return f"{title}\n{section}\n{snippet}".strip()


def _chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, str | int]:
    """Chroma metadata: only primitives. Store all fields needed for response."""
    return {
        "source": str(chunk.get("source", "answers")),
        "pageId": str(chunk.get("pageId", "")),
        "title": str(chunk.get("title", "")),
        "spaceKey": str(chunk.get("spaceKey", "")),
        "spaceName": str(chunk.get("spaceName", "")),
        "updated": str(chunk.get("updated", "")),
        "url": str(chunk.get("url", "")),
        "section": str(chunk.get("section", "")),
        "snippet": str(chunk.get("snippet", "")),
    }


def _metadata_to_chunk(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct chunk dict from Chroma document metadata."""
    return {
        "source": metadata.get("source", "answers"),
        "pageId": metadata.get("pageId", ""),
        "title": metadata.get("title", ""),
        "spaceKey": metadata.get("spaceKey", ""),
        "spaceName": metadata.get("spaceName", ""),
        "updated": metadata.get("updated", ""),
        "url": metadata.get("url", ""),
        "section": metadata.get("section", ""),
        "snippet": metadata.get("snippet", ""),
    }


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings from config."""
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        from config.settings import settings
        api_key = api_key or getattr(settings, "openai_api_key", None)
    except ImportError:
        pass
    return OpenAIEmbeddings(
        model=wiki_config.EMBEDDING_MODEL,
        api_key=api_key or None,
    )


def get_vector_store(persist_directory: str | None = None) -> Chroma:
    """Get or create Chroma vector store. Uses VECTOR_STORE_PATH from config if not provided."""
    persist = persist_directory or wiki_config.VECTOR_STORE_PATH
    embedding = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist if persist else None,
    )


def clear_vector_store(persist_directory: str | None = None) -> None:
    """Delete the Chroma collection so the next index run starts fresh."""
    persist = persist_directory or wiki_config.VECTOR_STORE_PATH
    if not persist:
        return
    import chromadb
    client = chromadb.PersistentClient(path=persist)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def add_chunks(store: Chroma, chunks: list[dict[str, Any]], start_id: int = 0) -> None:
    """Add chunk dicts to the vector store. Embeds in batches. Use start_id so IDs are unique across multiple calls (append not replace)."""
    if not chunks:
        return
    texts = [_text_to_embed(c) for c in chunks]
    metadatas = [_chunk_to_metadata(c) for c in chunks]
    ids = [f"chunk_{start_id + i}" for i in range(len(chunks))]
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metadatas = metadatas[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        store.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
    if hasattr(store, "_persist") and store._persist_directory:
        store.persist()


def semantic_search(store: Chroma, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Run similarity search and return chunk dicts in the same shape as answers_retrieve."""
    docs = store.similarity_search(query, k=top_k)
    return [_metadata_to_chunk(doc.metadata) for doc in docs]
