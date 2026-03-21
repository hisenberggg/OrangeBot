"""Trim and format checkpointed chat messages for multi-turn LLM calls."""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Cap Human+AI messages passed to leaf LLMs (each turn = up to 2 messages).
MAX_CHAT_MESSAGES_FOR_LEAF = 20
# Slightly smaller prefix for wiki context string (prior turns only).
MAX_PRIOR_MESSAGES_FOR_WIKI = 16


def message_text(m: BaseMessage) -> str:
    """String content from a message (handles str or multimodal list)."""
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return " ".join(parts).strip() or str(c)
    return str(c)


def filter_chat_messages(messages: Sequence[BaseMessage] | None) -> list[HumanMessage | AIMessage]:
    """Keep only human/assistant turns (no tool/system in checkpoint)."""
    out: list[HumanMessage | AIMessage] = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            out.append(m)
        elif isinstance(m, AIMessage):
            out.append(m)
    return out


def trim_chat_messages(
    messages: Sequence[BaseMessage] | None,
    *,
    max_messages: int = MAX_CHAT_MESSAGES_FOR_LEAF,
) -> list[HumanMessage | AIMessage]:
    """Return the last `max_messages` human/ai messages (in order)."""
    chat = filter_chat_messages(messages)
    if len(chat) <= max_messages:
        return chat
    return chat[-max_messages:]


def build_system_plus_history(
    system_text: str,
    state_messages: Sequence[BaseMessage] | None,
    *,
    max_messages: int = MAX_CHAT_MESSAGES_FOR_LEAF,
) -> list[BaseMessage]:
    """System prompt + trimmed conversation for chat models."""
    hist = trim_chat_messages(state_messages, max_messages=max_messages)
    return [SystemMessage(content=system_text), *hist]


def wiki_question_with_context(state_messages: Sequence[BaseMessage] | None) -> str:
    """
    Build wiki retrieval/eval question: prior turns as text + current user line.
    ReAct subgraph still gets a single HumanMessage per hop; this string carries memory.
    """
    chat = filter_chat_messages(state_messages)
    if not chat:
        return ""
    last = chat[-1]
    if not isinstance(last, HumanMessage):
        return message_text(last)
    last_user = message_text(last)
    prior = chat[:-1]
    if not prior:
        return last_user
    prior_trim = trim_chat_messages(prior, max_messages=MAX_PRIOR_MESSAGES_FOR_WIKI)
    parts = [
        "Use the prior conversation below to interpret follow-up questions. "
        "The final line is the current question to answer with wiki evidence.",
        "",
    ]
    for m in prior_trim:
        label = "User" if isinstance(m, HumanMessage) else "Assistant"
        parts.append(f"{label}: {message_text(m)}")
    parts.extend(["", f"Current question: {last_user}"])
    return "\n".join(parts)
