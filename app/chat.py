"""Coordinate the chat workflow independently of HTTP and provider SDK details."""

from app.llm import generate_reply
from app.memory import conversation_memory
from app.models import ChatRequest, ChatResponse


def handle_chat(request: ChatRequest) -> ChatResponse:
    """Use recent conversation to answer, then remember the successful exchange."""
    history = conversation_memory.get_history(request.user_id, request.session_id)
    answer = generate_reply(request.message, history=history)
    # If generation raises an error, execution never reaches this write: failed
    # requests do not leave an unanswered user message in conversation history.
    conversation_memory.add_exchange(
        request.user_id, request.session_id, request.message, answer
    )
    return ChatResponse(
        response=answer,
        user_id=request.user_id,
        session_id=request.session_id,
        context_used=["recent_conversation"] if history else [],
    )
