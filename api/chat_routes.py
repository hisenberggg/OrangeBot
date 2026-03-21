"""Chat thread CRUD (JWT protected)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api import chat_store
from api.deps import get_current_user_id

router = APIRouter(prefix="/chats", tags=["chats"])


class ThreadCreateBody(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ThreadOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: str
    route: str | None = None


@router.get("", response_model=list[ThreadOut])
def list_chats(user_id: str = Depends(get_current_user_id)) -> list[ThreadOut]:
    threads = chat_store.list_threads(user_id)
    return [
        ThreadOut(
            id=t["id"],
            title=t.get("title") or "New chat",
            created_at=t.get("created_at") or "",
            updated_at=t.get("updated_at") or "",
        )
        for t in threads
    ]


@router.post("", response_model=ThreadOut)
def create_chat(
    body: ThreadCreateBody,
    user_id: str = Depends(get_current_user_id),
) -> ThreadOut:
    title = (body.title or "New chat").strip() or "New chat"
    thread_id = chat_store.create_thread(user_id, title=title)
    threads = chat_store.list_threads(user_id)
    t = next((x for x in threads if x["id"] == thread_id), None)
    if not t:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Thread creation failed")
    return ThreadOut(
        id=t["id"],
        title=t.get("title") or "New chat",
        created_at=t.get("created_at") or "",
        updated_at=t.get("updated_at") or "",
    )


@router.get("/{thread_id}/messages", response_model=list[MessageOut])
def get_thread_messages(
    thread_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[MessageOut]:
    if not chat_store.assert_thread_owned(user_id, thread_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")
    raw = chat_store.get_messages(user_id, thread_id)
    return [
        MessageOut(
            role=m["role"],
            content=m.get("content") or "",
            timestamp=m.get("timestamp") or "",
            route=m.get("route"),
        )
        for m in raw
    ]
