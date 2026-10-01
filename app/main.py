"""Create the FastAPI application and expose its HTTP endpoints."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.brain import BrainError, BrainStore, create_driver
from app.chat import handle_chat
from app.config import ConfigurationError, get_neo4j_settings
from app.llm import LLMError, LLMTimeoutError
from app.models import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own one reusable database driver and release it when the app stops."""
    driver = None
    app.state.brain = None
    try:
        settings = get_neo4j_settings()
        driver = create_driver(settings)
        app.state.brain = BrainStore(driver, settings.database)
    except (ConfigurationError, BrainError):
        # Missing DB configuration should not take down /health or basic chat.
        # After fixing missing configuration, restart the app to load it.
        pass
    try:
        yield
    finally:
        if driver is not None:
            driver.close()


def get_brain(request: Request) -> BrainStore | None:
    """Supply the shared store; tests can replace this dependency with a fake."""
    return getattr(request.app.state, "brain", None)


# Uvicorn imports this application to serve incoming HTTP requests.
app = FastAPI(title="Memory-Aware Assistant", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm the API is responding; this does not check external services."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, brain: BrainStore | None = Depends(get_brain)) -> ChatResponse:
    """Answer a validated message and translate expected failures into HTTP errors."""
    # Keep this endpoint synchronous: FastAPI runs it in a worker thread so the
    # blocking SDK call does not block the server's asynchronous event loop.
    try:
        return handle_chat(request, brain=brain)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
