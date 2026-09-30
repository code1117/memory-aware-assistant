"""Check temporary memory behavior without credentials or external services."""

import pytest

from app.chat import handle_chat
from app.llm import LLMError
from app.memory import ConversationMemory
from app.models import ChatRequest


def test_history_is_isolated_by_user_and_session():
    """Neither another session nor another user can retrieve the exchange."""
    memory = ConversationMemory()
    memory.add_exchange("alice", "one", "Hello", "Hi")
    assert memory.get_history("alice", "one") == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert memory.get_history("alice", "two") == []
    assert memory.get_history("bob", "one") == []


def test_history_keeps_only_recent_complete_turns():
    """Dropping old history preserves user/assistant pairs in order."""
    memory = ConversationMemory(max_turns=2)
    for number in range(3):
        memory.add_exchange("alice", "one", f"Q{number}", f"A{number}")
    assert [item["content"] for item in memory.get_history("alice", "one")] == [
        "Q1", "A1", "Q2", "A2"
    ]


def test_reading_history_does_not_expose_mutable_storage():
    """A caller editing its snapshot cannot rewrite a remembered message."""
    memory = ConversationMemory()
    memory.add_exchange("alice", "one", "Original", "Answer")
    history = memory.get_history("alice", "one")
    history[0]["content"] = "Changed"
    assert memory.get_history("alice", "one")[0]["content"] == "Original"


def test_least_recently_used_session_is_evicted():
    """Session capacity is bounded, while a recently read session is retained."""
    memory = ConversationMemory(max_sessions=2)
    memory.add_exchange("alice", "one", "Q1", "A1")
    memory.add_exchange("alice", "two", "Q2", "A2")
    memory.get_history("alice", "one")
    memory.add_exchange("alice", "three", "Q3", "A3")
    assert memory.get_history("alice", "two") == []
    assert memory.get_history("alice", "one")
    assert memory.get_history("alice", "three")


def test_chat_passes_history_and_reports_context(monkeypatch):
    """The second answer receives the first exchange without duplicating input."""
    memory = ConversationMemory()
    calls = []

    def fake_reply(message, history):
        """Capture model inputs and return a deterministic answer without an API call."""
        calls.append((message, history))
        return "Practice generators."

    monkeypatch.setattr("app.chat.conversation_memory", memory)
    monkeypatch.setattr("app.chat.generate_reply", fake_reply)
    first = handle_chat(ChatRequest(user_id="alice", session_id="one", message="Topic?"))
    second = handle_chat(ChatRequest(user_id="alice", session_id="one", message="Explain it."))
    assert first.context_used == []
    assert second.context_used == ["recent_conversation"]
    assert calls == [
        ("Topic?", []),
        ("Explain it.", [
            {"role": "user", "content": "Topic?"},
            {"role": "assistant", "content": "Practice generators."},
        ]),
    ]
    assert len(memory.get_history("alice", "one")) == 4


def test_failed_generation_does_not_change_history(monkeypatch):
    """A provider failure leaves the previously successful history intact."""
    memory = ConversationMemory()
    memory.add_exchange("alice", "one", "Hello", "Hi")
    before = memory.get_history("alice", "one")

    def failing_reply(message, history):
        """Simulate an unavailable AI service without making a network request."""
        raise LLMError("Unavailable")

    monkeypatch.setattr("app.chat.conversation_memory", memory)
    monkeypatch.setattr("app.chat.generate_reply", failing_reply)
    with pytest.raises(LLMError):
        handle_chat(ChatRequest(user_id="alice", session_id="one", message="New question"))
    assert memory.get_history("alice", "one") == before
