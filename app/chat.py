"""Coordinate the chat workflow independently of HTTP and provider SDK details."""

from app.llm import generate_reply
from app.models import ChatRequest, ChatResponse


def handle_chat(request: ChatRequest) -> ChatResponse:
    """Generate an answer and attach identifiers for the caller's conversation."""
    answer = generate_reply(request.message)
    # No stored context is used yet; ChatResponse supplies context_used=[].
    return ChatResponse(
        response=answer,
        user_id=request.user_id,
        session_id=request.session_id,
    )
