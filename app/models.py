"""Define and validate the JSON data accepted and returned by the chat API."""

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UserProfile(BaseModel):
    """Store optional profile details; unknown values remain unknown."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    date_of_birth: date | None = None
    time_of_birth: time | None = None
    birth_place: str | None = Field(default=None, min_length=1, max_length=200)
    preferred_language: str | None = Field(default=None, min_length=1, max_length=80)
    zodiac_sign: str | None = Field(default=None, min_length=1, max_length=80)


class MemoryFact(BaseModel):
    """Represent one useful fact; its stable key identifies later corrections."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # Reuse a key such as career_change when correcting the same fact. A
    # separate goal needs a different key, even if both concern career.
    key: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    kind: Literal["goal", "preference", "interest", "important_memory", "life_area", "astrology"]
    topic: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    content: str = Field(min_length=1, max_length=2000)
    target_year: int | None = Field(default=None, ge=1900, le=2200)
    timeframe: str | None = Field(default=None, min_length=1, max_length=100)


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
