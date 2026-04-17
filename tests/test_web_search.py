"""Tests for web route and Tavily client (no live API)."""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-ci")


def test_web_node_no_messages():
    from agent.planner import _web_node

    out = asyncio.run(_web_node({}))
    assert "No question" in (out.get("final_response") or "")


def test_web_node_with_excerpts_mocked(monkeypatch):
    from agent.planner import _web_node
    from langchain_core.messages import HumanMessage

    async def fake_fetch(q: str) -> str:
        return "### Page\nSource: https://example.com\n\nHello world."

    monkeypatch.setattr("agent.tavily_client.fetch_web_context", fake_fetch)

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="Answer grounded in excerpts.")
    )
    monkeypatch.setattr("agent.planner.ChatOpenAI", lambda **kwargs: fake_llm)

    out = asyncio.run(
        _web_node({"messages": [HumanMessage(content="What is example?")]})
    )
    assert out.get("final_response") == "Answer grounded in excerpts."
    fake_llm.ainvoke.assert_called_once()


def test_fetch_web_context_sync_empty_without_key(monkeypatch):
    from config import settings
    from agent.tavily_client import fetch_web_context_sync

    monkeypatch.setattr(settings, "tavily_api_key", "")
    assert fetch_web_context_sync("any query") == ""


def test_planner_graph_includes_web_node():
    from agent import create_planner_graph

    graph = create_planner_graph()
    assert graph is not None
    if hasattr(graph, "get_graph"):
        g = graph.get_graph()
        assert "web" in g.nodes
        assert "web_enrich" not in g.nodes


def test_fetch_web_context_uses_enhanced_query_for_vague_input(monkeypatch):
    from config import settings
    from agent import tavily_client

    monkeypatch.setattr(settings, "tavily_api_key", "fake-tavily-key")
    monkeypatch.setattr(settings, "openai_api_key", "fake-openai-key")

    class _FakeStructuredLLM:
        def invoke(self, _messages):
            return tavily_client.QueryEnhancement(
                is_vague=True,
                enhanced_query="B1 hold Syracuse University MySlice",
            )

    class _FakeBaseLLM:
        def with_structured_output(self, _schema):
            return _FakeStructuredLLM()

    class _FakeTavily:
        last_search_query = None
        last_extract_query = None

        def __init__(self, api_key):
            assert api_key == "fake-tavily-key"

        def search(self, query, **_kwargs):
            _FakeTavily.last_search_query = query
            return {
                "results": [
                    {"url": "https://example.com/a", "title": "A", "content": "snippet"}
                ]
            }

        def extract(self, urls, query, **_kwargs):
            _FakeTavily.last_extract_query = query
            return {
                "results": [
                    {"url": urls[0], "raw_content": "content"}
                ]
            }

    monkeypatch.setattr(tavily_client, "ChatOpenAI", lambda **_kwargs: _FakeBaseLLM())
    monkeypatch.setattr("tavily.TavilyClient", _FakeTavily)

    out = tavily_client.fetch_web_context_sync("what is b1 hold")
    assert out
    assert _FakeTavily.last_search_query == "B1 hold Syracuse University MySlice"
    assert _FakeTavily.last_extract_query == "B1 hold Syracuse University MySlice"


def test_fetch_web_context_keeps_specific_query(monkeypatch):
    from config import settings
    from agent import tavily_client

    original_query = "Syracuse University B1 hold meaning in MySlice"
    monkeypatch.setattr(settings, "tavily_api_key", "fake-tavily-key")
    monkeypatch.setattr(settings, "openai_api_key", "fake-openai-key")

    class _FakeStructuredLLM:
        def invoke(self, _messages):
            return tavily_client.QueryEnhancement(
                is_vague=False,
                enhanced_query=original_query,
            )

    class _FakeBaseLLM:
        def with_structured_output(self, _schema):
            return _FakeStructuredLLM()

    class _FakeTavily:
        last_search_query = None
        last_extract_query = None

        def __init__(self, api_key):
            assert api_key == "fake-tavily-key"

        def search(self, query, **_kwargs):
            _FakeTavily.last_search_query = query
            return {
                "results": [
                    {"url": "https://example.com/b", "title": "B", "content": "snippet"}
                ]
            }

        def extract(self, urls, query, **_kwargs):
            _FakeTavily.last_extract_query = query
            return {
                "results": [
                    {"url": urls[0], "raw_content": "content"}
                ]
            }

    monkeypatch.setattr(tavily_client, "ChatOpenAI", lambda **_kwargs: _FakeBaseLLM())
    monkeypatch.setattr("tavily.TavilyClient", _FakeTavily)

    out = tavily_client.fetch_web_context_sync(original_query)
    assert out
    assert _FakeTavily.last_search_query == original_query
    assert _FakeTavily.last_extract_query == original_query


def test_fetch_web_context_enhancement_failure_falls_back(monkeypatch):
    from config import settings
    from agent import tavily_client

    original_query = "what is b1 hold"
    monkeypatch.setattr(settings, "tavily_api_key", "fake-tavily-key")
    monkeypatch.setattr(settings, "openai_api_key", "fake-openai-key")

    class _FakeStructuredLLM:
        def invoke(self, _messages):
            raise RuntimeError("llm unavailable")

    class _FakeBaseLLM:
        def with_structured_output(self, _schema):
            return _FakeStructuredLLM()

    class _FakeTavily:
        last_search_query = None
        last_extract_query = None

        def __init__(self, api_key):
            assert api_key == "fake-tavily-key"

        def search(self, query, **_kwargs):
            _FakeTavily.last_search_query = query
            return {
                "results": [
                    {"url": "https://example.com/c", "title": "C", "content": "snippet"}
                ]
            }

        def extract(self, urls, query, **_kwargs):
            _FakeTavily.last_extract_query = query
            return {
                "results": [
                    {"url": urls[0], "raw_content": "content"}
                ]
            }

    monkeypatch.setattr(tavily_client, "ChatOpenAI", lambda **_kwargs: _FakeBaseLLM())
    monkeypatch.setattr("tavily.TavilyClient", _FakeTavily)

    out = tavily_client.fetch_web_context_sync(original_query)
    assert out
    assert _FakeTavily.last_search_query == original_query
    assert _FakeTavily.last_extract_query == original_query
