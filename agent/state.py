"""Shared state typings for Planner and downstream agents."""
from typing import Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage

Route = Literal["wiki", "calendar", "general", "transit"]


class AgentState(TypedDict, total=False):
    """State passed through the Planner and into specialized agents."""

    messages: list[BaseMessage]
    route: Route
    route_rationale: str
    final_response: str
