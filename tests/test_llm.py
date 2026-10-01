"""Verify provider payloads and extraction safeguards with a fake SDK client."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm import LLMError, extract_updates, generate_reply, validate_extraction
from app.models import MemoryExtraction, MemoryFact, MemoryProposal, ProfileUpdate


@pytest.fixture
def sdk(monkeypatch):
    """Replace the SDK and credentials; no network or local .env is accessed."""
    state = {"payload": None, "result": SimpleNamespace(status="completed", output_text="Hello", output_parsed=MemoryExtraction())}

    class FakeClient:
        """Provide the SDK context-manager and response methods used by our adapter."""

        def __init__(self, **kwargs):
            """Expose fake responses without opening any network resources."""
            self.responses = self

        def __enter__(self):
            """Return the fake client when entering its context."""
            return self

        def __exit__(self, *args):
            """Allow errors to propagate naturally out of the fake context."""
            return False

        def create(self, **kwargs):
            """Capture the payload and return the configured fake response."""
            state["payload"] = kwargs
            return state["result"]

        def parse(self, **kwargs):
            """Capture structured-output requests using the same fake response."""
            return self.create(**kwargs)

    monkeypatch.setattr("app.llm.OpenAI", FakeClient)
    monkeypatch.setattr("app.llm.get_settings", lambda: Settings(openai_api_key="fake-test-key"))
    return state


def test_generation_sends_context_then_history_then_current_message(sdk):
    """The actual provider payload preserves roles, ordering, and one current message."""
    history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    generate_reply("My question", history=history, profile={"name": "Rahul"})
    messages = sdk["payload"]["input"]
    assert "Rahul" in messages[0]["content"]
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "My question"}
    assert sdk["payload"]["store"] is False


@pytest.mark.parametrize("status,text", [("incomplete", "Partial"), ("completed", "   ")])
def test_empty_or_incomplete_generation_is_not_success(sdk, status, text):
    """An HTTP success from the provider alone does not guarantee a usable answer."""
    sdk["result"] = SimpleNamespace(status=status, output_text=text)
    with pytest.raises(LLMError):
        generate_reply("Hello")


def test_extraction_requires_evidence_from_current_user_message():
    """A quote from a previous message or assistant text cannot justify a new write."""
    extraction = MemoryExtraction(memories=[MemoryProposal(
        fact=MemoryFact(key="career_change", kind="goal", topic="career", content="Change jobs."),
        evidence="I will change jobs",
    )])
    with pytest.raises(LLMError):
        validate_extraction("What should I do?", extraction)
    validate_extraction("I will change jobs next year.", extraction)


def test_invalid_extracted_profile_date_is_rejected():
    """Invalid profile values are blocked even when the evidence quote is present."""
    extraction = MemoryExtraction(profile_updates=[ProfileUpdate(
        field="date_of_birth", value="not-a-date", evidence="my birthday"
    )])
    with pytest.raises(LLMError):
        validate_extraction("It is my birthday", extraction)


def test_structured_extraction_uses_schema_and_rejects_missing_result(sdk):
    """Refusals or missing parsed output cannot silently become successful updates."""
    assert extract_updates("Hello", [], []) == MemoryExtraction()
    assert sdk["payload"]["text_format"] is MemoryExtraction
    sdk["result"] = SimpleNamespace(status="completed", output_parsed=None)
    with pytest.raises(LLMError):
        extract_updates("Hello", [], [])
