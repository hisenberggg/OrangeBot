"""
Chat API: FastAPI endpoint that runs the Planner graph (with Wiki agent wired)
and returns the final response. Uses JWT auth, per-thread checkpoints, and async transcript persistence.
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
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent import create_planner_graph, create_wiki_graph
from api import chat_store
from api.app_data import ensure_data_dir, get_data_dir
from api.auth_routes import router as auth_router
from api.chat_routes import router as chat_router
from api.deps import get_current_user_id

_planner_graph = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer, _planner_graph
    ensure_data_dir()
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = str((get_data_dir() / "checkpoints.db").resolve())
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        _checkpointer = saver
        _planner_graph = None
        yield
        _planner_graph = None
        _checkpointer = None


app = FastAPI(title="University Chat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)


async def _get_planner_graph():
    """Build planner + wiki graph once; compile with AsyncSqlite checkpointer from lifespan."""
    global _planner_graph
    if _planner_graph is None:
        from langgraph.checkpoint.memory import MemorySaver

        cp = _checkpointer if _checkpointer is not None else MemorySaver()
        wiki_graph = await create_wiki_graph()
        _planner_graph = create_planner_graph(wiki_graph=wiki_graph, checkpointer=cp)
    return _planner_graph


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or question")
    include_route: bool = Field(default=True, description="Include chosen route in response")
    thread_id: str = Field(..., description="Chat thread id (from POST /chats)")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Final assistant response")
    route: str | None = Field(default=None, description="Planner route: wiki, calendar, or general")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_graph(
    message: str,
    thread_id: str,
    user_id: str,
    background_tasks: BackgroundTasks,
    include_route: bool,
) -> AsyncGenerator[str, None]:
    """Run the planner graph and yield SSE events; schedule transcript persist after success."""
    if not chat_store.assert_thread_owned(user_id, thread_id):
        yield _sse("error", {"message": "Thread not found"})
        return

    graph = await _get_planner_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    past_routing = False
    accumulated = ""
    route_value: str | None = None
    wiki_tool_calls = 0
    has_streamed_response = False
    final_response: str | None = None
    stream_error: str | None = None

    yield _sse("thinking", {"type": "status", "message": "Deciding route..."})

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config,
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            tags = event.get("tags") or []

            if kind == "on_chain_end" and name == "route":
                output = event.get("data", {}).get("output", {})
                route_value = output.get("route", "general")
                rationale = output.get("route_rationale", "")
                past_routing = True
                yield _sse(
                    "thinking",
                    {
                        "type": "route",
                        "route": route_value,
                        "rationale": rationale,
                    },
                )
                label = {
                    "wiki": "Searching Answers wiki...",
                    "calendar": "Checking academic calendar...",
                    "general": "Generating response...",
                    "transit": "Checking bus schedules...",
                }.get(route_value, "Processing...")
                yield _sse("thinking", {"type": "status", "message": label})

            elif kind == "on_tool_start" and past_routing:
                tool_input = event.get("data", {}).get("input", {})
                query = (
                    tool_input.get("query", str(tool_input))
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )

                if route_value == "wiki":
                    wiki_tool_calls += 1
                    if wiki_tool_calls > 1 and has_streamed_response:
                        accumulated = ""
                        has_streamed_response = False
                        yield _sse(
                            "thinking",
                            {
                                "type": "status",
                                "message": f"Evaluating response... retrying with different search (attempt {wiki_tool_calls})...",
                            },
                        )

                yield _sse(
                    "thinking",
                    {
                        "type": "tool_call",
                        "tool": name,
                        "input": query[:200],
                    },
                )

            elif kind == "on_chat_model_stream" and past_routing:
                if "wiki_evaluator" in tags:
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    accumulated += chunk.content
                    has_streamed_response = True
                    yield _sse("delta", {"content": chunk.content})

            elif kind == "on_chain_end" and name in ("wiki", "calendar", "general", "transit"):
                output = event.get("data", {}).get("output", {})
                final = output.get("final_response") or accumulated
                final_response = final
                yield _sse(
                    "done",
                    {
                        "response": final,
                        "route": route_value if include_route else None,
                    },
                )

    except Exception as exc:
        stream_error = str(exc)
        yield _sse("error", {"message": stream_error})
    finally:
        if stream_error is None and final_response:
            background_tasks.add_task(
                chat_store.append_turn,
                user_id,
                thread_id,
                message,
                final_response,
                route_value if include_route else None,
            )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> ChatResponse:
    """Run the Planner (with Wiki agent when route is wiki) and return the final response."""
    if not chat_store.assert_thread_owned(user_id, req.thread_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")

    graph = await _get_planner_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": req.thread_id}}
    state: Any = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config,
    )
    final = state.get("final_response") or "I couldn't process that. Please try again."
    route = state.get("route") if req.include_route else None

    background_tasks.add_task(
        chat_store.append_turn,
        user_id,
        req.thread_id,
        req.message,
        final,
        route,
    )

    return ChatResponse(response=final, route=route)


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """SSE streaming endpoint: emits thinking steps + response tokens."""
    return StreamingResponse(
        _stream_graph(
            req.message,
            req.thread_id,
            user_id,
            background_tasks,
            req.include_route,
        ),
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
