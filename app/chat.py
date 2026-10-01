"""Coordinate the chat workflow independently of HTTP and provider SDK details."""

from pydantic import ValidationError

from app.brain import BrainError, BrainStore
from app.config import ConfigurationError
from app.context import merge_profile, select_profile, select_topics
from app.llm import LLMError, extract_updates, generate_reply
from app.memory import conversation_memory
from app.models import ChatRequest, ChatResponse, MemoryExtraction, UserProfile


def handle_chat(request: ChatRequest, brain: BrainStore | None = None) -> ChatResponse:
    """Retrieve relevant knowledge, answer, then persist validated user facts."""
    history = conversation_memory.get_history(request.user_id, request.session_id)
    topics = select_topics(request.message, history)
    warnings = []
    stored_profile = None
    memories = []
    database_ready = False
    if brain is not None:
        try:
            brain.ensure_schema()
            stored_profile = brain.get_profile(request.user_id)
            # Filtering occurs inside Neo4j; never load the entire user graph.
            memories = brain.get_memories(request.user_id, topics=topics, limit=5) if topics != [] else []
            database_ready = True
        except (BrainError, ValidationError):
            stored_profile = None
            memories = []
    if not database_ready:
        warnings.append("Persistent memory is unavailable; this turn cannot read or save long-term information.")

    profile = select_profile(merge_profile(stored_profile, request.profile), request.message, topics)
    context_used = (["recent_conversation"] if history else [])
    if profile:
        context_used.append("user_profile")
    context_used.extend(dict.fromkeys(f"{fact.topic}_{fact.kind}" for fact in memories))
    # Generation errors propagate to HTTP handling. Nothing is saved on failure.
    answer = generate_reply(request.message, history=history, profile=profile, memories=memories)

    memory_status = "unavailable"
    if database_ready:
        extraction = MemoryExtraction()
        extraction_failed = False
        try:
            extraction = extract_updates(request.message, history, memories)
        except (LLMError, ConfigurationError, ValidationError):
            extraction_failed = True
            warnings.append("The answer succeeded, but facts from this message could not be extracted for long-term memory.")

        changes = {change.field: change.value for change in extraction.profile_updates}
        # Explicit structured profile input wins over extracted profile guesses.
        if request.profile is not None:
            changes.update(request.profile.model_dump(mode="json", exclude_unset=True))
        facts = [proposal.fact for proposal in extraction.memories]
        memory_status = "failed" if extraction_failed else "unchanged"
        if changes or facts:
            try:
                brain.save_updates(request.user_id, UserProfile.model_validate(changes), facts)
                memory_status = "saved"
            except (BrainError, ValidationError):
                memory_status = "failed"
                warnings.append("The answer succeeded, but the long-term memory update was not saved.")

    # Preserve a successful conversation even if long-term extraction/storage
    # fails. Writes finish before returning HTTP so the next request sees them.
    conversation_memory.add_exchange(
        request.user_id, request.session_id, request.message, answer
    )
    return ChatResponse(
        response=answer,
        user_id=request.user_id,
        session_id=request.session_id,
        context_used=context_used,
        memory_status=memory_status,
        warnings=warnings,
    )
