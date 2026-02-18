"""
Planner agent: classifies the user question and routes to wiki / calendar / general.
No MCP tools; single LLM call with structured output.
"""
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import settings
from agent.state import AgentState, Route

ROUTING_PROMPT = """You are a router for a university help assistant. Classify the user's question into exactly one route.

Routes:
- wiki: Procedures, how-to, policy, or meaning (e.g. "How do I drop a course?", "What happens after I drop?", "What is add/drop?")
- calendar: Exact dates or academic calendar (e.g. "When is add/drop deadline for spring 2026?", "What is the last day to add a class?")
- general: Unclear, out of scope, or greeting (e.g. "Hi", "What can you do?"). Use for polite fallback.

Respond with only the route: wiki, calendar, or general."""


class RouteOutput(BaseModel):
    """Structured output for the planner router."""

    route: Literal["wiki", "calendar", "general"] = Field(
        description="One of: wiki, calendar, general"
    )
    rationale: str = Field(default="", description="Brief reason for this route.")


def _route_node(state: AgentState) -> AgentState:
    """Single node: call LLM to get route, then return updated state."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(RouteOutput)
    messages = state.get("messages") or []
    if not messages:
        return {**state, "route": "general"}
    user_content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    response = structured_llm.invoke(
        [
            SystemMessage(content=ROUTING_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    route: Route = response.route or "general"
    return {**state, "route": route}


def _general_node(state: AgentState) -> AgentState:
    """Fallback when route is general."""
    return {
        **state,
        "final_response": "I can help with Syracuse University topics: academic calendar dates and Answers wiki (procedures, how-to, policies). What would you like to know?",
    }


def _calendar_placeholder_node(state: AgentState) -> AgentState:
    """Placeholder for future Calendar agent."""
    return {
        **state,
        "final_response": "Academic calendar dates are not yet available in this assistant. Please check the university academic calendar directly. I can help with procedures and policies from the Answers wiki.",
    }


def _wiki_placeholder_node(state: AgentState) -> AgentState:
    """Placeholder when wiki graph is not provided."""
    return {
        **state,
        "final_response": "[Wiki agent not wired. Start with create_planner_graph(wiki_graph=...) to enable.]",
    }


def _route_edges(state: AgentState) -> Literal["wiki", "calendar", "general"]:
    """Conditional edge: next node from state['route']."""
    return state.get("route") or "general"


def create_planner_graph(wiki_graph=None):
    """
    Build the Planner graph: route node -> conditional -> wiki | calendar | general -> END.
    Pass wiki_graph (compiled Wiki agent from create_wiki_graph) to wire the Wiki agent.
    """
    from agent.wiki_agent import wiki_node
    graph = StateGraph(AgentState)

    def _make_async_wiki_node(wg):
        async def _async_wiki(state):
            return await wiki_node(state, wg)
        return _async_wiki

    graph.add_node("route", _route_node)
    graph.add_node(
        "wiki",
        _make_async_wiki_node(wiki_graph) if wiki_graph else _wiki_placeholder_node,
    )
    graph.add_node("calendar", _calendar_placeholder_node)
    graph.add_node("general", _general_node)

    graph.add_conditional_edges("route", _route_edges, ["wiki", "calendar", "general"])
    graph.add_edge("wiki", END)
    graph.add_edge("calendar", END)
    graph.add_edge("general", END)

    # Entry: we need an entry point. Plan says "user question" -> Planner, so START -> route
    graph.set_entry_point("route")

    return graph.compile()
