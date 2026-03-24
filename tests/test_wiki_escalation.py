"""Tests for wiki -> web escalation routing (no live MCP)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-ci")


def test_after_wiki_edges_escalates_when_flag_set():
    from agent.planner import _after_wiki_edges

    assert _after_wiki_edges({"wiki_escalate_to_web": True}) == "web"
    assert _after_wiki_edges({"wiki_escalate_to_web": False}) == "done"
    assert _after_wiki_edges({}) == "done"


def test_planner_graph_wiki_branches_to_web_and_end():
    from agent import create_planner_graph

    g = create_planner_graph().get_graph()
    wiki_out = [e for e in g.edges if e.source == "wiki"]
    targets = {e.target for e in wiki_out}
    assert "web" in targets
    assert "__end__" in targets
