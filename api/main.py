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

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
