"""Smoke: checkpointed thread accumulates AIMessage between turns (mocked LLM)."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# Planner imports need OPENAI_API_KEY in CI
pytest.importorskip("langgraph", reason="langgraph required")


class _FakeStructured:
    def invoke(self, _msgs):
        from agent.planner import RouteOutput

        return RouteOutput(route="general", rationale="test")


class _FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.ainvoke_calls: list = []

    def with_structured_output(self, _schema):
        return _FakeStructured()

    async def ainvoke(self, msgs):
        self.ainvoke_calls.append(list(msgs))
        class _R:
            content = "Assistant reply"

        return _R()


@pytest.fixture
def fake_llm():
    return _FakeChatOpenAI()


def test_two_turns_checkpoint_contains_ai_messages(fake_llm):
    async def _run():
        with patch("agent.planner.ChatOpenAI", return_value=fake_llm):
            from agent.planner import create_planner_graph

            graph = create_planner_graph(wiki_graph=None)
            config = {"configurable": {"thread_id": "t-multiturn-1"}}

            await graph.ainvoke({"messages": [HumanMessage("first")]}, config)
            await graph.ainvoke({"messages": [HumanMessage("second")]}, config)

            snap = graph.get_state(config)
            msgs = snap.values.get("messages") or []
            ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
            human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]

            assert len(human_msgs) == 2
            assert len(ai_msgs) == 2
            assert all("Assistant reply" in str(m.content) for m in ai_msgs)

            # Second general call should include prior user + assistant + new user
            assert len(fake_llm.ainvoke_calls) >= 2
            second_batch = fake_llm.ainvoke_calls[1]
            assert len(second_batch) >= 4  # system + H + AI + H

    asyncio.run(_run())
