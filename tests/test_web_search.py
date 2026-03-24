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
