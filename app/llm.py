"""Isolate provider-specific API calls behind a simple text-generation function."""

from openai import APIError, APITimeoutError, OpenAI

from app.config import get_settings


SYSTEM_INSTRUCTIONS = (
    "You are a friendly assistant for personalized astrology-themed conversations. "
    "Give concise, helpful answers based only on information supplied to you. "
    "Do not invent user details or claim to remember earlier conversations. "
    "Ask a brief clarifying question when essential information is missing. "
    "Astrology calculations are not connected: do not claim to have calculated "
    "a birth chart or present predictions as certain."
)


class LLMError(Exception):
    """Represent a provider failure without exposing its raw response or secrets."""


class LLMTimeoutError(LLMError):
    """Distinguish a timed-out provider request from other generation failures."""

    
def generate_reply(message: str) -> str:
    """Generate one answer; keep SDK details and errors out of the chat workflow."""
    settings = get_settings()
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
                input=message,
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
