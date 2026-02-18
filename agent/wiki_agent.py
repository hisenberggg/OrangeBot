"""
Wiki agent: LangGraph sub-graph that uses wiki MCP tools via langchain-mcp-adapters,
runs a tool-calling loop (ReAct), and returns an answer with citations.
"""
import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings

WIKI_SYSTEM_PROMPT = """You are a helpful assistant for Syracuse University. Use the Answers wiki tools to find up-to-date information about procedures, policies, and how-to topics. When you use answers_retrieve or answers_search_cql, cite the sources (page title and section) in your answer. If you cannot find relevant information, say so clearly."""


async def _get_wiki_tools():
    """Load wiki MCP tools via stdio transport."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    import sys
    # Run from project root so mcp_servers.wiki.server is importable
    cmd = getattr(settings, "wiki_mcp_command", "python") or "python"
    args = getattr(settings, "wiki_mcp_args", ("-m", "mcp_servers.wiki.server")) or ("-m", "mcp_servers.wiki.server")
    if isinstance(args, str):
        args = [a.strip() for a in args.split(",")]
    client = MultiServerMCPClient({
        "wiki": {
            "transport": "stdio",
            "command": cmd,
            "args": list(args),
        }
    })
    tools = await client.get_tools()
    return tools


def create_wiki_graph_sync():
    """Synchronous factory: run async create_wiki_graph in event loop."""
    return asyncio.run(create_wiki_graph())


async def create_wiki_graph():
    """
    Build the Wiki agent graph: ReAct agent with wiki MCP tools.
    Returns a compiled graph that accepts state with 'messages' and returns state with 'messages'.
    """
    tools = await _get_wiki_tools()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0,
    )
    # create_react_agent uses 'prompt' (str → SystemMessage) and returns a CompiledStateGraph (already compiled)
    app = create_react_agent(llm, tools, prompt=WIKI_SYSTEM_PROMPT)
    return app


async def wiki_node(state: dict[str, Any], wiki_compiled_graph) -> dict[str, Any]:
    """
    Planner node: invoke the Wiki subgraph (async so MCP tools run with ainvoke) and set final_response.
    """
    messages = state.get("messages") or []
    if not messages:
        return {**state, "final_response": "No question was provided."}
    result = await wiki_compiled_graph.ainvoke({"messages": messages})
    out_messages = result.get("messages") or []
    final_response = ""
    for m in reversed(out_messages):
        if isinstance(m, AIMessage):
            final_response = m.content if hasattr(m, "content") else str(m)
            break
    if not final_response:
        final_response = "I couldn't generate an answer from the wiki. Please try rephrasing."
    return {**state, "final_response": final_response}
