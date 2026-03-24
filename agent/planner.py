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
- wiki: Syracuse University procedures, how-to, policy, or meaning in Answers wiki scope (e.g. "How do I drop a course?", "What is add/drop?")
- calendar: Exact dates, deadlines, or academic calendar questions answerable from the registrar calendar (e.g. "When is spring break?", "Last day to add a class?")
- transit: Bus schedules, shuttle times, trolley routes, Centro routes on campus (e.g. "South Campus shuttle", "Blue Loop")
- web: Questions that need current or broad web information — news, comparisons, general knowledge, facts not covered by wiki/calendar/transit, "latest" topics, or anything where search results would clearly improve the answer. Prefer web over general whenever the user wants substantive information from the wider web.
- general: ONLY basic interactions — greetings, thanks, "what can you do?", minimal chitchat, or tiny meta questions that need no tools and no web search. Do NOT use general for factual or open-ended knowledge questions; use web instead.

Pick exactly one route. Output structured fields: route, rationale."""


class RouteOutput(BaseModel):
    """Structured output for the planner router."""

    route: Literal["wiki", "calendar", "general", "transit", "web"] = Field(
        description="One of: wiki, calendar, general, transit, web"
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
        return {**state, "route": "general", "route_rationale": ""}
    user_content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    response = structured_llm.invoke(
        [
            SystemMessage(content=ROUTING_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    route: Route = response.route or "general"
    return {
        **state,
        "route": route,
        "route_rationale": response.rationale or "",
    }


GENERAL_SYSTEM_PROMPT = """You are a brief, friendly Syracuse University chat assistant for simple turns only (greetings, thanks, what you can help with).

Keep answers short. Mention you can help with: Answers wiki procedures, academic calendar dates, campus transit schedules, and web search for current or general topics.

If the user asks for substantive facts or research, say they can ask a specific question and the app will route it to the right tool — do not pretend to browse the web yourself.

Well-known links you may cite when relevant:
- https://www.syracuse.edu
- https://myslice.syr.edu
- https://answers.atlassian.syr.edu/wiki"""


WEB_SYSTEM_PROMPT_BASE = """You are a helpful assistant for a Syracuse University user. You answer using the WEB EXCERPTS below (from a live search and page extract). Ground your answer in those excerpts when possible; mention source titles and URLs when helpful.

When the excerpts do not help: if they are missing or empty, or nothing in them substantively answers the user's question, say clearly that no relevant information was found in the retrieved web results. Give a short, cautious answer without inventing facts; suggest refining the question or checking an official or primary source when appropriate.

Third-party web content may be wrong or outdated — encourage the user to verify critical facts.

--- WEB EXCERPTS ---
"""


async def _web_node(state: AgentState) -> AgentState:
    """Run Tavily search+extract, then answer with the LLM using retrieved text."""
    from agent.tavily_client import fetch_web_context

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0.25,
    )
    messages = state.get("messages") or []
    if not messages:
        text = "No question was provided."
        return {
            **state,
            "final_response": text,
            "messages": [AIMessage(content=text)],
        }
    last = messages[-1]
    user_content = (last.content if hasattr(last, "content") else str(last)).strip()
    if not user_content:
        text = "No question was provided."
        return {
            **state,
            "final_response": text,
            "messages": [AIMessage(content=text)],
        }

    excerpts = await fetch_web_context(user_content)
    if not excerpts.strip():
        excerpts = "[No web excerpts returned. TAVILY_API_KEY may be unset, or search/extract returned no content.]"

    system_full = f"{WEB_SYSTEM_PROMPT_BASE}{excerpts}"
    msg_list = build_system_plus_history(system_full, messages)
    response = await llm.ainvoke(msg_list)
    content = response.content if hasattr(response, "content") else str(response)
    return {
        **state,
        "final_response": content,
        "messages": [AIMessage(content=content)],
    }


async def _general_node(state: AgentState) -> AgentState:
    """Minimal general node: greetings and short meta only."""
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
        "wiki_escalate_to_web": False,
        "messages": [AIMessage(content=text)],
    }


def _route_edges(state: AgentState) -> Literal["wiki", "calendar", "general", "transit", "web"]:
    """Conditional edge: next node from state['route']."""
    return state.get("route") or "general"


def _after_wiki_edges(state: AgentState) -> Literal["web", "done"]:
    """After wiki node: escalate to web search or finish."""
    if state.get("wiki_escalate_to_web"):
        return "web"
    return "done"


def create_planner_graph(wiki_graph=None, checkpointer=None):
    """
    Build the Planner graph: route node -> conditional -> wiki | calendar | transit | general | web -> END.
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
    graph.add_node("web", _web_node)

    graph.add_conditional_edges(
        "route", _route_edges, ["wiki", "calendar", "general", "transit", "web"]
    )
    graph.add_conditional_edges(
        "wiki",
        _after_wiki_edges,
        {"web": "web", "done": END},
    )
    graph.add_edge("calendar", END)
    graph.add_edge("general", END)
    graph.add_edge("transit", END)
    graph.add_edge("web", END)

    # Entry: we need an entry point. Plan says "user question" -> Planner, so START -> route
    graph.set_entry_point("route")

    return graph.compile(checkpointer=checkpointer)
