"""
Wiki agent: LangGraph sub-graph that uses wiki MCP tools via langchain-mcp-adapters,
runs a tool-calling loop (ReAct), evaluates response quality, and retries with
reformulated queries if the answer is insufficient (up to MAX_WIKI_HOPS attempts).
"""
import asyncio
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config import settings
from agent.conversation import wiki_question_with_context

MAX_WIKI_HOPS = 3

WIKI_SYSTEM_PROMPT = """You are a helpful assistant for Syracuse University. Use the Answers wiki tool to find up-to-date information about procedures, policies, and how-to topics. Call answers_retrieve to obtain evidence snippets from the pre-indexed Answers wiki (page title, section, and snippet text), and cite those sources in your answer. If you cannot find relevant information, say so clearly.

When citing sources, copy the exact `url` field from each evidence item verbatim in markdown links. Do not change the hostname, path, or rewrite links to a different root domain."""

EVAL_PROMPT = """You are an evaluation judge for a Syracuse Answers wiki assistant. Compare the original user question to the wiki-based response.

Set is_adequate to true ONLY if the response clearly and directly answers what the user asked, with specific on-topic information. When the answer claims procedures or policy, it should cite wiki evidence (URLs/snippets) from retrieval.

Set is_adequate to false if ANY of these hold:
- The response is vague, hedging, or only partially addresses the question
- It says it could not find information, or gives generic advice without concrete wiki-backed details
- It is tangential, off-topic, or likely misunderstands the question
- It would leave a reasonable user still needing to search elsewhere

Be strict: if you are unsure, choose inadequate. When attempt {attempt} equals the last try ({max_attempts} of {max_attempts}), inadequate answers may trigger a web search fallback—do not pass borderline answers to avoid that.

If inadequate and this is not the final attempt, suggest a rephrased search query (different keywords, synonyms, broader or narrower scope than previous tries).

Previous queries tried: {previous_queries}
Attempt: {attempt} of {max_attempts}"""


class WikiEvaluation(BaseModel):
    """Structured output from the wiki response evaluator."""

    is_adequate: bool = Field(
        description=(
            "True only if the response clearly and fully answers the user's question "
            "with specific, on-topic content (and wiki citations when claiming policy/procedure)."
        )
    )
    reasoning: str = Field(description="Brief explanation of the evaluation")
    suggested_query: str = Field(
        default="", description="Rephrased search query if inadequate"
    )
    strategy: Literal["rephrase", "broaden", "more_results"] = Field(
        default="rephrase",
        description="Retry strategy: rephrase (different keywords), broaden (wider scope), more_results (more docs)",
    )


def _extract_ai_response(result: dict[str, Any]) -> str:
    """Extract the final AI message content from ReAct agent output."""
    out_messages = result.get("messages") or []
    for m in reversed(out_messages):
        if isinstance(m, AIMessage):
            return m.content if hasattr(m, "content") else str(m)
    return ""


async def _get_wiki_tools():
    """Load wiki MCP tools via stdio transport."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    cmd = getattr(settings, "wiki_mcp_command", "python") or "python"
    args = getattr(settings, "wiki_mcp_args", ("-m", "mcp_servers.wiki.server")) or (
        "-m",
        "mcp_servers.wiki.server",
    )
    if isinstance(args, str):
        args = [a.strip() for a in args.split(",")]
    client = MultiServerMCPClient(
        {
            "wiki": {
                "transport": "stdio",
                "command": cmd,
                "args": list(args),
            }
        }
    )
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
    app = create_react_agent(llm, tools, prompt=WIKI_SYSTEM_PROMPT)
    return app


def _build_hop_message(
    original_question: str,
    query: str,
    attempt: int,
    strategy: str,
) -> list[HumanMessage]:
    """Build the user message for a retrieval hop."""
    if attempt == 1:
        return [HumanMessage(content=original_question)]

    prefix = {
        "broaden": "Search broadly and consider related topics. ",
        "more_results": "Retrieve more documents and try multiple searches. ",
    }.get(strategy, "")

    return [
        HumanMessage(
            content=(
                f"{prefix}Search for: {query}\n\n"
                f"(Original question: {original_question})"
            )
        )
    ]


async def wiki_node(state: dict[str, Any], wiki_compiled_graph) -> dict[str, Any]:
    """
    Planner node: invoke the Wiki subgraph with evaluate-and-retry loop.
    Evaluates response quality after each ReAct pass and retries with
    reformulated queries if inadequate (up to MAX_WIKI_HOPS attempts).
    """
    messages = state.get("messages") or []
    if not messages:
        err = "No question was provided."
        return {
            **state,
            "final_response": err,
            "wiki_hops": 0,
            "wiki_escalate_to_web": False,
            "messages": [AIMessage(content=err)],
        }

    original_question = wiki_question_with_context(messages)

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key or None,
        temperature=0,
    ).with_structured_output(WikiEvaluation)

    current_query = original_question
    strategy = "rephrase"
    queries_tried: list[str] = []
    best_response = ""
    eval_reasoning = ""
    wiki_escalate_to_web = False

    for attempt in range(1, MAX_WIKI_HOPS + 1):
        queries_tried.append(current_query)

        hop_messages = _build_hop_message(
            original_question, current_query, attempt, strategy
        )
        result = await wiki_compiled_graph.ainvoke({"messages": hop_messages})
        response = _extract_ai_response(result)

        if not response:
            response = "I couldn't generate an answer from the wiki."

        best_response = response

        try:
            evaluation = await evaluator_llm.ainvoke(
                [
                    SystemMessage(
                        content=EVAL_PROMPT.format(
                            previous_queries=", ".join(
                                f'"{q}"' for q in queries_tried
                            ),
                            attempt=attempt,
                            max_attempts=MAX_WIKI_HOPS,
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Original question: {original_question}\n\n"
                            f"Wiki response:\n{response}"
                        )
                    ),
                ],
                config={"tags": ["wiki_evaluator"]},
            )
        except Exception:
            eval_reasoning = "Evaluation failed."
            if attempt >= MAX_WIKI_HOPS:
                wiki_escalate_to_web = True
                break
            current_query = f"{original_question} (retry after evaluation error)"
            strategy = "rephrase"
            continue

        eval_reasoning = evaluation.reasoning

        if evaluation.is_adequate:
            wiki_escalate_to_web = False
            break

        if attempt >= MAX_WIKI_HOPS:
            wiki_escalate_to_web = True
            break

        if evaluation.suggested_query:
            current_query = evaluation.suggested_query
        else:
            current_query = f"{original_question} (alternative search)"
        strategy = evaluation.strategy

    if wiki_escalate_to_web:
        return {
            **state,
            "wiki_escalate_to_web": True,
            "wiki_hops": len(queries_tried),
            "wiki_eval_reasoning": eval_reasoning,
            "messages": [],
        }

    if not best_response:
        best_response = (
            "I couldn't generate an answer from the wiki. Please try rephrasing."
        )

    return {
        **state,
        "wiki_escalate_to_web": False,
        "final_response": best_response,
        "wiki_hops": len(queries_tried),
        "wiki_eval_reasoning": eval_reasoning,
        "messages": [AIMessage(content=best_response)],
    }
