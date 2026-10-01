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

    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    profile: UserProfile | None = None


class ChatResponse(BaseModel):
    """Return the answer, conversation identifiers, and context categories used."""

    response: str
    user_id: str
    session_id: str
    # Each response gets its own list; it stays empty if no prior context is used.
    context_used: list[str] = Field(default_factory=list)
    memory_status: Literal["saved", "unchanged", "unavailable", "failed"] = "unchanged"
    warnings: list[str] = Field(default_factory=list)


class ProfileUpdate(BaseModel):
    """One explicitly stated profile change, supported by a quote from the message."""

    model_config = ConfigDict(extra="forbid")
    field: Literal["name", "date_of_birth", "time_of_birth", "birth_place", "preferred_language", "zodiac_sign"]
    value: str | None
    evidence: str = Field(min_length=1, max_length=500)


class MemoryProposal(BaseModel):
    """A proposed durable fact and the exact user text supporting it."""

    model_config = ConfigDict(extra="forbid")
    fact: MemoryFact
    evidence: str = Field(min_length=1, max_length=500)


class MemoryExtraction(BaseModel):
    """Bound the structured output used to update persistent user knowledge."""

    model_config = ConfigDict(extra="forbid")
    profile_updates: list[ProfileUpdate] = Field(default_factory=list, max_length=6)
    memories: list[MemoryProposal] = Field(default_factory=list, max_length=5)
