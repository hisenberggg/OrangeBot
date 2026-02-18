"""Tests for the Planner agent (routing only)."""
import os
import pytest
from unittest.mock import patch

# Ensure config is loadable when running tests from project root
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-ci")


def test_planner_graph_builds_without_wiki():
    """Planner compiles without passing a wiki graph (placeholder used)."""
    from agent import create_planner_graph
    graph = create_planner_graph()
    assert graph is not None


def test_planner_state_shape():
    """AgentState has expected keys."""
    from agent.state import AgentState, Route
    state: AgentState = {"messages": [], "route": "general"}
    assert state.get("route") in ("wiki", "calendar", "general") or state.get("route") is None
