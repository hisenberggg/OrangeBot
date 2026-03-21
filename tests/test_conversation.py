"""Unit tests for multi-turn message trimming/formatting."""
from langchain_core.messages import AIMessage, HumanMessage

from agent.conversation import (
    build_system_plus_history,
    filter_chat_messages,
    message_text,
    trim_chat_messages,
    wiki_question_with_context,
)


def test_filter_chat_messages_drops_non_chat():
    msgs = [
        HumanMessage("hi"),
        AIMessage("hello"),
    ]
    assert len(filter_chat_messages(msgs)) == 2


def test_trim_chat_messages_max():
    msgs = [HumanMessage(str(i)) for i in range(25)]
    trimmed = trim_chat_messages(msgs, max_messages=20)
    assert len(trimmed) == 20
    assert message_text(trimmed[0]) == "5"
    assert message_text(trimmed[-1]) == "24"


def test_build_system_plus_history():
    msgs = [HumanMessage("a"), AIMessage("b"), HumanMessage("c")]
    out = build_system_plus_history("SYS", msgs)
    assert out[0].content == "SYS"
    assert len(out) == 4


def test_wiki_question_with_context_single_user():
    msgs = [HumanMessage("only me")]
    assert wiki_question_with_context(msgs) == "only me"


def test_wiki_question_with_context_prior_and_current():
    msgs = [
        HumanMessage("What is add drop?"),
        AIMessage("Add/drop is the period when you can change classes."),
        HumanMessage("When is it for spring?"),
    ]
    q = wiki_question_with_context(msgs)
    assert "Current question: When is it for spring?" in q
    assert "Assistant:" in q
    assert "User: What is add drop?" in q


def test_message_text_multimodal_list_dict():
    m = HumanMessage(content=[{"type": "text", "text": "hello"}])
    assert message_text(m) == "hello"
