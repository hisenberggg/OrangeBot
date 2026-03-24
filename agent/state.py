"""Shared state typings for Planner and downstream agents."""
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

Route = Literal["wiki", "calendar", "general", "transit", "web"]


class AgentState(TypedDict, total=False):
    """State passed through the Planner and into specialized agents."""

    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
    route_rationale: str
    final_response: str
    wiki_hops: int
    wiki_eval_reasoning: str
    wiki_escalate_to_web: bool
