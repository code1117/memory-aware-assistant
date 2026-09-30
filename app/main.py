"""Create the FastAPI application and expose its HTTP endpoints."""

from fastapi import FastAPI, HTTPException

from app.chat import handle_chat
from app.config import ConfigurationError
from app.llm import LLMError, LLMTimeoutError
from app.models import ChatRequest, ChatResponse


# Uvicorn imports this application to serve incoming HTTP requests.
app = FastAPI(title="Memory-Aware Assistant")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm the API is responding; this does not check external services."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a validated message and translate expected failures into HTTP errors."""
    # Keep this endpoint synchronous: FastAPI runs it in a worker thread so the
    # blocking SDK call does not block the server's asynchronous event loop.
    try:
        return handle_chat(request)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
