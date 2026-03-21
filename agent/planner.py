"""
Planner agent: classifies the user question and routes to wiki / calendar / general.
No MCP tools; single LLM call with structured output.
"""
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import settings
from agent.conversation import build_system_plus_history
from agent.state import AgentState, Route

_CALENDAR_JSON_PATH = Path(__file__).resolve().parent.parent / "Data" / "calendar.json"
_calendar_cache: str | None = None

_BUS_JSON_PATH = Path(__file__).resolve().parent.parent / "Data" / "bus_schedules.json"
_bus_cache: str | None = None


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
- transit: Bus schedules, shuttle times, trolley routes, Centro routes (e.g. "When does the South Campus shuttle run?", "What time is the next Blue Loop?", "Which bus goes to Destiny USA?")
- general: Greetings, general knowledge, off-topic, or anything not covered by the above routes (e.g. "Hi", "What can you do?", "Tell me about Syracuse University").

Respond with only the route: wiki, calendar, transit, or general."""


class RouteOutput(BaseModel):
    """Structured output for the planner router."""

    route: Literal["wiki", "calendar", "general", "transit"] = Field(
        description="One of: wiki, calendar, general, transit"
    )
    rationale: str = Field(default="", description="Brief reason for this route.")


def _route_node(state: AgentState) -> AgentState:
    """Classify the latest user message into a route (no full history — lowest latency)."""
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
    msg_list = build_system_plus_history(GENERAL_SYSTEM_PROMPT, messages)
    response = await llm.ainvoke(msg_list)
    content = response.content if hasattr(response, "content") else str(response)
    return {
        **state,
        "final_response": content,
        "messages": [AIMessage(content=content)],
    }


CALENDAR_SYSTEM_PROMPT = """You are a Syracuse University academic calendar assistant. Use the calendar data provided below to answer the user's question about dates and deadlines. Always cite the specific date. If the data doesn't contain the answer, say so and suggest checking https://www.syracuse.edu/academics/calendars/."""


async def _calendar_node(state: AgentState) -> AgentState:
    """Calendar node: loads full calendar data as LLM context and answers date questions."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0,
    )
    messages = state.get("messages") or []
    calendar_context = _load_calendar_context()
    system_full = f"{CALENDAR_SYSTEM_PROMPT}\n\n--- CALENDAR DATA ---\n{calendar_context}"
    msg_list = build_system_plus_history(system_full, messages)
    response = await llm.ainvoke(msg_list)
    content = response.content if hasattr(response, "content") else str(response)
    return {
        **state,
        "final_response": content,
        "messages": [AIMessage(content=content)],
    }


def _load_bus_context() -> str:
    """Load bus schedule JSON and format as a text block for the LLM. Cached after first read."""
    global _bus_cache
    if _bus_cache is not None:
        return _bus_cache

    if not _BUS_JSON_PATH.exists():
        _bus_cache = "[Bus schedule data not available. Run: python -m scripts.scrape_bus_schedules]"
        return _bus_cache

    data = json.loads(_BUS_JSON_PATH.read_text(encoding="utf-8"))
    lines: list[str] = []
    for route in data.get("routes", []):
        name = route.get("name", "Unknown")
        category = route.get("category", "").replace("Link", "")
        lines.append(f"=== {name} ({category}) ===")
        stops = route.get("stops", [])
        if stops:
            lines.append(f"Stops: {' -> '.join(stops)}")
        notes = route.get("notes", "")
        if notes:
            lines.append(f"Notes: {notes}")
        trips = route.get("trips", [])
        if trips:
            for trip in trips:
                lines.append(trip)
        raw_text = route.get("raw_text", "")
        if raw_text and not trips:
            lines.append(raw_text)
        if route.get("source_url"):
            lines.append(f"Source: {route['source_url']}")
        if route.get("map_url"):
            lines.append(f"Map: {route['map_url']}")
        lines.append("")

    _bus_cache = "\n".join(lines)
    return _bus_cache


TRANSIT_SYSTEM_PROMPT = """You are a Syracuse University transit assistant. Use the bus and shuttle schedule data provided below to answer the user's question about routes, stops, and times.

The current date and time is provided below. Use it to determine which buses are upcoming or have already passed.

Each trip is listed as: StopName TIME -> StopName TIME -> ... showing the exact time the bus reaches each stop in sequence. Use these stop-time pairs directly when answering — do NOT guess or interpolate times.

Always cite the specific route name, stop name, and times. If the data doesn't contain the answer, say so and suggest checking https://parking.syr.edu/transportation/shuttle-information/campus-shuttle-schedules/ for the latest schedules.

When mentioning a route, include the source PDF link if available."""


async def _transit_node(state: AgentState) -> AgentState:
    """Transit node: loads full bus schedule data as LLM context and answers route/time questions."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0,
    )
    messages = state.get("messages") or []
    from datetime import datetime as _dt
    time_str = _dt.now().strftime("%A, %B %d, %Y at %I:%M %p")
    bus_context = _load_bus_context()
    system_full = (
        f"{TRANSIT_SYSTEM_PROMPT}\n\nCurrent time: {time_str}\n\n"
        f"--- BUS SCHEDULE DATA ---\n{bus_context}"
    )
    msg_list = build_system_plus_history(system_full, messages)
    response = await llm.ainvoke(msg_list)
    content = response.content if hasattr(response, "content") else str(response)
    return {
        **state,
        "final_response": content,
        "messages": [AIMessage(content=content)],
    }


def _wiki_placeholder_node(state: AgentState) -> AgentState:
    """Placeholder when wiki graph is not provided."""
    text = (
        "[Wiki agent not wired. Start with create_planner_graph(wiki_graph=...) to enable.]"
    )
    return {
        **state,
        "final_response": text,
        "messages": [AIMessage(content=text)],
    }


def _route_edges(state: AgentState) -> Literal["wiki", "calendar", "general", "transit"]:
    """Conditional edge: next node from state['route']."""
    return state.get("route") or "general"


def create_planner_graph(wiki_graph=None, checkpointer=None):
    """
    Build the Planner graph: route node -> conditional -> wiki | calendar | general -> END.
    Pass wiki_graph (compiled Wiki agent from create_wiki_graph) to wire the Wiki agent.
    Pass checkpointer for thread persistence (SqliteSaver / AsyncSqliteSaver / MemorySaver).
    If checkpointer is None, uses in-memory MemorySaver (tests / local fallback).
    """
    from langgraph.checkpoint.memory import MemorySaver
    from agent.wiki_agent import wiki_node

    if checkpointer is None:
        checkpointer = MemorySaver()

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
    graph.add_node("transit", _transit_node)

    graph.add_conditional_edges("route", _route_edges, ["wiki", "calendar", "general", "transit"])
    graph.add_edge("wiki", END)
    graph.add_edge("calendar", END)
    graph.add_edge("general", END)
    graph.add_edge("transit", END)

    # Entry: we need an entry point. Plan says "user question" -> Planner, so START -> route
    graph.set_entry_point("route")

    return graph.compile(checkpointer=checkpointer)
