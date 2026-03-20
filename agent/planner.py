"""
Planner agent: classifies the user question and routes to wiki / calendar / general.
No MCP tools; single LLM call with structured output.
"""
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import settings
from agent.state import AgentState, Route

_CALENDAR_JSON_PATH = Path(__file__).resolve().parent.parent / "Data" / "calendar.json"
_calendar_cache: str | None = None


def _load_calendar_context() -> str:
    """Load calendar JSON and format as a text block for the LLM. Cached after first read."""
    global _calendar_cache
    if _calendar_cache is not None:
        return _calendar_cache

    if not _CALENDAR_JSON_PATH.exists():
        _calendar_cache = "[Calendar data not available. Run: python -m scripts.scrape_calendar]"
        return _calendar_cache

    data = json.loads(_CALENDAR_JSON_PATH.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cal_type, entries in data.get("calendars", {}).items():
        for entry in entries:
            lines.append(f"[{entry.get('semester', '')}] {entry.get('event', '')}: {entry.get('date', '')}")

    _calendar_cache = "\n".join(lines)
    return _calendar_cache

ROUTING_PROMPT = """You are a router for a university help assistant. Classify the user's question into exactly one route.

Routes:
- wiki: Procedures, how-to, policy, or meaning (e.g. "How do I drop a course?", "What happens after I drop?", "What is add/drop?")
- calendar: Exact dates, deadlines, or academic calendar questions (e.g. "When is add/drop deadline for spring 2026?", "What is the last day to add a class?", "When is spring break?")
- general: Greetings, general knowledge, off-topic, or anything not covered by wiki/calendar routes (e.g. "Hi", "What can you do?", "Tell me about Syracuse University").

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
    return {**state, "route": route, "route_rationale": response.rationale or ""}


GENERAL_SYSTEM_PROMPT = """You are a helpful Syracuse University assistant. Answer the user's question to the best of your ability. You specialize in Syracuse University topics and have access to an Answers wiki for procedures and policies. If the question is a greeting, respond warmly and mention what you can help with. If the question is completely off-topic, politely redirect.

When relevant, include links to well-known Syracuse University resources you are confident about, such as:
- Syracuse University homepage: https://www.syracuse.edu
- MySlice student portal: https://myslice.syr.edu
- SU Answers wiki: https://answers.atlassian.syr.edu/wiki
- Financial Aid: https://www.syracuse.edu/admissions-aid/financial-aid/
- Registrar: https://www.syracuse.edu/academics/registrar/
- Campus directory: https://www.syracuse.edu/directory/
Only include links you are confident are correct. Do not guess or fabricate URLs."""


async def _general_node(state: AgentState) -> AgentState:
    """General-purpose node: calls the LLM to answer greetings, general knowledge, and off-topic questions."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0.3,
    )
    messages = state.get("messages") or []
    user_content = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    response = await llm.ainvoke([
        SystemMessage(content=GENERAL_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    return {**state, "final_response": response.content}


CALENDAR_SYSTEM_PROMPT = """You are a Syracuse University academic calendar assistant. Use the calendar data provided below to answer the user's question about dates and deadlines. Always cite the specific date. If the data doesn't contain the answer, say so and suggest checking https://www.syracuse.edu/academics/calendars/."""


async def _calendar_node(state: AgentState) -> AgentState:
    """Calendar node: loads full calendar data as LLM context and answers date questions."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0,
    )
    messages = state.get("messages") or []
    user_content = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    calendar_context = _load_calendar_context()
    response = await llm.ainvoke([
        SystemMessage(content=f"{CALENDAR_SYSTEM_PROMPT}\n\n--- CALENDAR DATA ---\n{calendar_context}"),
        HumanMessage(content=user_content),
    ])
    return {**state, "final_response": response.content}


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
    graph.add_node("calendar", _calendar_node)
    graph.add_node("general", _general_node)

    graph.add_conditional_edges("route", _route_edges, ["wiki", "calendar", "general"])
    graph.add_edge("wiki", END)
    graph.add_edge("calendar", END)
    graph.add_edge("general", END)

    # Entry: we need an entry point. Plan says "user question" -> Planner, so START -> route
    graph.set_entry_point("route")

    return graph.compile()
