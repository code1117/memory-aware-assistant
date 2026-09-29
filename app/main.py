"""Create the FastAPI application and expose its HTTP endpoints."""

from fastapi import FastAPI


# Uvicorn imports this application to serve incoming HTTP requests.
app = FastAPI(title="Memory-Aware Assistant")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm the API is responding; this does not check external services."""
    return {"status": "ok"}
