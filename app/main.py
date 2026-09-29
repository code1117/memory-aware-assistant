"""Create the FastAPI application and expose its HTTP endpoints."""

from fastapi import FastAPI

from app.models import ChatRequest, ChatResponse


# Uvicorn imports this application to serve incoming HTTP requests.
app = FastAPI(title="Memory-Aware Assistant")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm the API is responding; this does not check external services."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Accept a validated message and return a placeholder until AI is connected."""
    # FastAPI validates the request before calling this function. The response
    # echoes the identifiers so the caller can match it to their conversation.
    return ChatResponse(
        response="Placeholder response: your message was received. AI and memory are not connected yet.",
        user_id=request.user_id,
        session_id=request.session_id,
    )
