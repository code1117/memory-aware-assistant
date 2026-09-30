"""Define and validate the JSON data accepted and returned by the chat API."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Identify the user and conversation, and require a nonblank message."""

    # Trim surrounding whitespace before checking each field's minimum length.
    # Strict validation rejects other JSON types instead of converting them.
    model_config = ConfigDict(str_strip_whitespace=True, strict=True)

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Return the answer, conversation identifiers, and context categories used."""

    response: str
    user_id: str
    session_id: str
    # Each response gets its own list; it stays empty if no prior context is used.
    context_used: list[str] = Field(default_factory=list)
