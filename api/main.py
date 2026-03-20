"""
Chat API: FastAPI endpoint that runs the Planner graph (with Wiki agent wired)
and returns the final response.
"""
# Load .env before any LangChain/LangSmith imports so LANGCHAIN_TRACING_V2 and
# LANGCHAIN_API_KEY are in os.environ (langsmith caches env at first read).
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import json
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent import create_planner_graph, create_wiki_graph

# Build graph lazily on first request (wiki MCP + planner with wiki wired)
_planner_graph = None


async def _get_planner_graph():
    """Build planner + wiki graph once; use async so we never call asyncio.run() from a running loop."""
    global _planner_graph
    if _planner_graph is None:
        wiki_graph = await create_wiki_graph()
        _planner_graph = create_planner_graph(wiki_graph=wiki_graph)
    return _planner_graph


app = FastAPI(title="University Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or question")
    include_route: bool = Field(default=True, description="Include chosen route in response")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Final assistant response")
    route: str | None = Field(default=None, description="Planner route: wiki, calendar, or general")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Run the Planner (with Wiki agent when route is wiki) and return the final response."""
    graph = await _get_planner_graph()
    state: Any = await graph.ainvoke({
        "messages": [HumanMessage(content=req.message)],
    })
    final = state.get("final_response") or "I couldn't process that. Please try again."
    route = state.get("route") if req.include_route else None
    return ChatResponse(response=final, route=route)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_graph(message: str) -> AsyncGenerator[str, None]:
    """Run the planner graph and yield SSE events for thinking steps + response tokens."""
    graph = await _get_planner_graph()
    past_routing = False
    accumulated = ""
    route_value: str | None = None

    yield _sse("thinking", {"type": "status", "message": "Deciding route..."})

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_end" and name == "route":
                output = event.get("data", {}).get("output", {})
                route_value = output.get("route", "general")
                rationale = output.get("route_rationale", "")
                past_routing = True
                yield _sse("thinking", {
                    "type": "route",
                    "route": route_value,
                    "rationale": rationale,
                })
                label = {
                    "wiki": "Searching Answers wiki...",
                    "calendar": "Checking academic calendar...",
                    "general": "Generating response...",
                    "transit": "Checking bus schedules...",
                }.get(route_value, "Processing...")
                yield _sse("thinking", {"type": "status", "message": label})

            elif kind == "on_tool_start" and past_routing:
                tool_input = event.get("data", {}).get("input", {})
                query = tool_input.get("query", str(tool_input)) if isinstance(tool_input, dict) else str(tool_input)
                yield _sse("thinking", {
                    "type": "tool_call",
                    "tool": name,
                    "input": query[:200],
                })

            elif kind == "on_chat_model_stream" and past_routing:
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    accumulated += chunk.content
                    yield _sse("delta", {"content": chunk.content})

            elif kind == "on_chain_end" and name in ("wiki", "calendar", "general", "transit"):
                output = event.get("data", {}).get("output", {})
                final = output.get("final_response") or accumulated
                yield _sse("done", {"response": final, "route": route_value})

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint: emits thinking steps + response tokens."""
    return StreamingResponse(
        _stream_graph(req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
