"""Isolate provider-specific API calls behind a simple text-generation function."""

from openai import APIError, APITimeoutError, OpenAI
from openai.types.responses import ResponseInputParam

from app.config import get_settings
from app.memory import ConversationMessage


SYSTEM_INSTRUCTIONS = (
    "You are a friendly assistant for personalized astrology-themed conversations. "
    "Give concise, helpful answers based only on information supplied to you. "
    "Use the supplied recent conversation to understand follow-up questions. "
    "Do not invent user details or claim knowledge of conversations not supplied. "
    "Ask a brief clarifying question when essential information is missing. "
    "Astrology calculations are not connected: do not claim to have calculated "
    "a birth chart or present predictions as certain."
)


class LLMError(Exception):
    """Represent a provider failure without exposing its raw response or secrets."""


class LLMTimeoutError(LLMError):
    """Distinguish a timed-out provider request from other generation failures."""


def generate_reply(
    message: str, history: list[ConversationMessage] | None = None
) -> str:
    """Generate an answer from recent messages followed by the current user input."""
    settings = get_settings()
    # Preserve roles and chronological order; the current message is added once.
    messages: ResponseInputParam = [
        {"role": item["role"], "content": item["content"]}
        for item in (history or [])
    ]
    messages.append({"role": "user", "content": message})
    try:
        # A context manager closes network resources even if the request fails.
        # Disable automatic retries to keep this first synchronous flow simple.
        with OpenAI(
            api_key=settings.openai_api_key,
            timeout=30.0,
            max_retries=0,
        ) as client:
            result = client.responses.create(
                model=settings.openai_model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=messages,
                max_output_tokens=600,
                store=False,
            )
    except APITimeoutError as exc:
        raise LLMTimeoutError("The AI service timed out. Please try again.") from exc
    except APIError as exc:
        raise LLMError(
            "The AI request failed. Check your API credentials, model access, "
            "and account limits, or try again later."
        ) from exc

    # Never report empty or unfinished model output as a successful answer.
    reply = result.output_text.strip()
    if result.status != "completed" or not reply:
        raise LLMError("The AI service did not return a complete answer. Please try again.")
    return reply
