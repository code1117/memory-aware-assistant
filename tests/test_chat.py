"""Exercise the complete HTTP workflow with fake providers: no keys or Docker needed."""

import pytest
from fastapi.testclient import TestClient

from app.brain import BrainError
from app.config import ConfigurationError
from app.llm import LLMError, LLMTimeoutError
from app.main import app, get_brain
from app.memory import ConversationMemory
from app.models import MemoryExtraction, MemoryFact, MemoryProposal, ProfileUpdate, UserProfile


class FakeBrain:
    """Emulate persistent storage across requests while keeping tests deterministic."""

    def __init__(self):
        """Start with empty data and controllable read/write failures."""
        self.profiles = {}
        self.memories = {}
        self.read_failure = False
        self.write_failure = False

    def ensure_schema(self):
        """Simulate database availability without performing schema operations."""
        if self.read_failure:
            raise BrainError("Database unavailable")

    def get_profile(self, user_id):
        """Read only the selected user's profile."""
        return self.profiles.get(user_id)

    def get_memories(self, user_id, topics=None, limit=20):
        """Apply the same user scope, topic filter, and result limit as the store."""
        return [fact for (owner, _), fact in self.memories.items()
                if owner == user_id and (topics is None or fact.topic in topics)][:limit]

    def save_updates(self, user_id, profile, memories):
        """Patch profile fields and overwrite existing facts by their stable keys."""
        if self.write_failure:
            raise BrainError("Write failed")
        existing = self.profiles.get(user_id, UserProfile()).model_dump(mode="json")
        existing.update(profile.model_dump(mode="json", exclude_unset=True))
        self.profiles[user_id] = UserProfile.model_validate(existing)
        for fact in memories:
            self.memories[(user_id, fact.key)] = fact


@pytest.fixture
def workflow(monkeypatch):
    """Replace both external services and reset RAM history for every scenario."""
    brain = FakeBrain()
    memory = ConversationMemory()
    state = {"extraction": MemoryExtraction(), "calls": [], "answer_error": None, "extraction_error": None}

    def fake_reply(message, history, profile, memories):
        """Capture exactly which context the response-generation call receives."""
        if state["answer_error"]:
            raise state["answer_error"]
        state["calls"].append({"message": message, "history": history, "profile": profile, "memories": memories})
        return "Answer based on the supplied context."

    def fake_extract(message, history, memories):
        """Supply a controlled extraction result independently of generation."""
        if state["extraction_error"]:
            raise state["extraction_error"]
        return state["extraction"]

    monkeypatch.setattr("app.chat.conversation_memory", memory)
    monkeypatch.setattr("app.chat.generate_reply", fake_reply)
    monkeypatch.setattr("app.chat.extract_updates", fake_extract)
    app.dependency_overrides[get_brain] = lambda: brain
    # No lifespan is needed: the dependency override supplies the entire store.
    client = TestClient(app)
    try:
        yield client, brain, state, memory
    finally:
        client.close()
        app.dependency_overrides.clear()


def post(client, message, session="one", user="alice", **extra):
    """Send a chat request with reusable test identifiers."""
    return client.post("/chat", json={"user_id": user, "session_id": session, "message": message, **extra})


def goal(year=2027):
    """Provide one durable career goal for retrieval and correction scenarios."""
    return MemoryFact(key="career_change", kind="goal", topic="career",
                      content=f"Plans to change jobs in {year}.", target_year=year)


def test_new_user_missing_profile_and_empty_memory(workflow):
    """A new user can chat without invented profile data or any stored memories."""
    client, brain, state, _ = workflow
    response = post(client, "Hello")
    assert response.status_code == 200
    assert response.json()["context_used"] == []
    assert response.json()["memory_status"] == "unchanged"
    assert state["calls"][0]["profile"] == {}
    assert brain.memories == {}


def test_creates_memory_and_reads_it_in_a_new_session(workflow):
    """A later session receives a saved goal even with no recent conversation."""
    client, brain, state, _ = workflow
    state["extraction"] = MemoryExtraction(memories=[MemoryProposal(fact=goal(), evidence="change jobs in 2027")])
    first = post(client, "I plan to change jobs in 2027.")
    assert first.json()["memory_status"] == "saved"
    state["extraction"] = MemoryExtraction()
    second = post(client, "What are my career plans?", session="two")
    assert second.json()["context_used"] == ["career_goal"]
    assert state["calls"][-1]["history"] == []
    assert state["calls"][-1]["memories"] == [goal()]


def test_follow_up_uses_history_and_preserves_topic(workflow):
    """An elliptical follow-up inherits the last relevant user topic."""
    client, brain, state, _ = workflow
    brain.memories[("alice", "career_change")] = goal()
    post(client, "What should I focus on for my career?")
    response = post(client, "Why do you say that?")
    assert "recent_conversation" in response.json()["context_used"]
    assert "career_goal" in response.json()["context_used"]
    assert len(state["calls"][-1]["history"]) == 2


def test_irrelevant_memories_and_other_users_are_excluded(workflow):
    """A career question must not include hobby facts or another user's goals."""
    client, brain, state, _ = workflow
    brain.memories[("bob", "career_change")] = goal()
    brain.memories[("alice", "cricket")] = MemoryFact(
        key="cricket", kind="interest", topic="hobbies", content="Enjoys cricket."
    )
    response = post(client, "What should I focus on for my career?")
    assert response.json()["context_used"] == []
    assert state["calls"][-1]["memories"] == []


def test_correction_updates_the_existing_goal(workflow):
    """A correction is stored under the existing key and retrieved in a later session."""
    client, brain, state, _ = workflow
    brain.memories[("alice", "career_change")] = goal()
    post(client, "What are my career plans?")
    state["extraction"] = MemoryExtraction(memories=[MemoryProposal(fact=goal(2028), evidence="2028")])
    assert post(client, "Actually, I mean 2028.").json()["memory_status"] == "saved"
    assert len(brain.memories) == 1
    state["extraction"] = MemoryExtraction()
    post(client, "What are my career plans?", session="new")
    assert state["calls"][-1]["memories"] == [goal(2028)]


def test_supplied_profile_is_used_and_saved(workflow):
    """Current profile input is usable immediately, but unrelated birth details stay out."""
    client, brain, state, _ = workflow
    response = post(client, "Help me with my career.", profile={"name": "Rahul", "birth_place": "Delhi"})
    assert response.json()["memory_status"] == "saved"
    assert state["calls"][-1]["profile"] == {"name": "Rahul"}
    assert brain.profiles["alice"].birth_place == "Delhi"


def test_profile_from_message_is_saved(workflow):
    """Validated profile extraction can store a name supplied in natural language."""
    client, brain, state, _ = workflow
    state["extraction"] = MemoryExtraction(profile_updates=[
        ProfileUpdate(field="name", value="Rahul", evidence="My name is Rahul")
    ])
    assert post(client, "My name is Rahul.").json()["memory_status"] == "saved"
    assert brain.profiles["alice"].name == "Rahul"


def test_database_failure_returns_answer_with_warning(workflow):
    """A graph outage permits basic chat without claiming persistent memory worked."""
    client, brain, _, _ = workflow
    brain.read_failure = True
    response = post(client, "Hello")
    assert response.status_code == 200
    assert response.json()["memory_status"] == "unavailable"
    assert response.json()["warnings"]


@pytest.mark.parametrize("failure", ["extraction", "write"])
def test_post_answer_failures_preserve_reply_but_report_unsaved_memory(workflow, failure):
    """The caller keeps a usable answer and receives an explicit failed-save status."""
    client, brain, state, memory = workflow
    state["extraction"] = MemoryExtraction(memories=[MemoryProposal(fact=goal(), evidence="2027")])
    if failure == "extraction":
        state["extraction_error"] = LLMError("Extraction failed")
    else:
        brain.write_failure = True
    response = post(client, "I plan to change jobs in 2027.")
    assert response.status_code == 200
    assert response.json()["memory_status"] == "failed"
    assert response.json()["warnings"]
    assert brain.memories == {}
    assert len(memory.get_history("alice", "one")) == 2


@pytest.mark.parametrize("error,status", [(LLMError("Failed"), 502), (LLMTimeoutError("Timed out"), 504), (ConfigurationError("Missing key"), 503)])
def test_generation_failure_returns_http_error_without_saving(workflow, error, status):
    """Failed answer generation must not mutate either kind of memory."""
    client, brain, state, memory = workflow
    state["answer_error"] = error
    response = post(client, "I plan to change jobs in 2027.", profile={"name": "Rahul"})
    assert response.status_code == status
    assert brain.profiles == {}
    assert brain.memories == {}
    assert memory.get_history("alice", "one") == []


@pytest.mark.parametrize("message", ["   ", 123, "a" * 4001])
def test_invalid_input_is_rejected_before_generation(workflow, message):
    """Blank, incorrectly typed, and oversized messages never reach the model."""
    client, _, state, _ = workflow
    assert post(client, message).status_code == 422
    assert state["calls"] == []
