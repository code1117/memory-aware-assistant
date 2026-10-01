"""Opt-in integration tests against local Neo4j; no OpenAI requests are made.

Run with RUN_NEO4J_TESTS=1 uv run pytest tests/test_brain.py -q.
Each test uses unique users and removes only its own generated data afterward.
"""

import os
from uuid import uuid4

import pytest

from app.brain import BrainStore, create_driver
from app.config import get_neo4j_settings
from app.models import MemoryFact, UserProfile


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1",
    reason="Set RUN_NEO4J_TESTS=1 to run tests against local Neo4j.",
)


@pytest.fixture
def brain():
    """Provide a real store and two isolated users, then clean up those users only."""
    settings = get_neo4j_settings()
    user_ids = [f"test-brain-{uuid4().hex}" for _ in range(2)]
    with create_driver(settings) as driver:
        store = BrainStore(driver, settings.database)
        store.ensure_schema()
        try:
            yield store, user_ids, settings
        finally:
            with driver.session(database=settings.database) as session:
                session.run(
                    "MATCH (m:Memory) WHERE m.user_id IN $ids DETACH DELETE m",
                    ids=user_ids,
                ).consume()
                session.run(
                    "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
                    ids=user_ids,
                ).consume()


def career_goal(year=2027):
    """Build the same keyed fact with a variable year for correction tests."""
    return MemoryFact(
        key="career_change", kind="goal", topic="career",
        content=f"Plans to change jobs in {year}.", target_year=year,
    )


def test_unknown_user_has_no_profile_or_memories(brain):
    """An unseen user returns empty results instead of inventing information."""
    store, users, _ = brain
    assert store.get_profile(users[0]) is None
    assert store.get_memories(users[0]) == []


def test_profile_round_trip_and_partial_correction(brain):
    """All profile fields survive storage; changing one preserves the others."""
    store, users, _ = brain
    profile = UserProfile(
        name="Rahul", date_of_birth="1995-08-15", time_of_birth="08:30:00",
        birth_place="Delhi", preferred_language="English", zodiac_sign="Leo",
    )
    store.save_profile(users[0], profile)
    assert store.get_profile(users[0]) == profile
    store.save_profile(users[0], UserProfile(birth_place="Mumbai"))
    updated = store.get_profile(users[0])
    assert updated.birth_place == "Mumbai"
    assert updated.name == "Rahul"
    assert updated.date_of_birth == profile.date_of_birth
    store.save_profile(users[0], UserProfile(time_of_birth=None))
    assert store.get_profile(users[0]).time_of_birth is None


def test_duplicate_and_corrected_fact_remains_one_memory(brain):
    """Repeated saves and an explicit correction reuse the same memory node."""
    store, users, _ = brain
    store.save_memory(users[0], career_goal())
    store.save_memory(users[0], career_goal())
    store.save_memory(users[0], career_goal(2028))
    assert store.get_memories(users[0]) == [career_goal(2028)]
    # Check actual node count too, so duplicate storage cannot hide behind LIMIT.
    with create_driver(brain[2]) as driver:
        with driver.session(database=brain[2].database) as session:
            record = session.run(
                "MATCH (m:Memory {user_id: $id}) RETURN count(m) AS total", id=users[0]
            ).single()
            assert record["total"] == 1


def test_users_can_have_the_same_fact_key_without_sharing_values(brain):
    """A correction for one user cannot change another user's career goal."""
    store, users, _ = brain
    store.save_profile(users[0], UserProfile(name="Rahul"))
    store.save_profile(users[1], UserProfile(name="Priya"))
    store.save_memory(users[0], career_goal(2027))
    store.save_memory(users[1], career_goal(2029))
    store.save_memory(users[0], career_goal(2028))
    assert store.get_memories(users[0]) == [career_goal(2028)]
    assert store.get_memories(users[1]) == [career_goal(2029)]
    assert store.get_profile(users[1]).name == "Priya"


def test_topic_filter_and_retrieval_limit(brain):
    """Career retrieval excludes unrelated facts, and results obey the limit."""
    store, users, _ = brain
    store.save_memory(users[0], career_goal())
    store.save_memory(users[0], MemoryFact(
        key="likes_cricket", kind="interest", topic="hobbies", content="Enjoys cricket."
    ))
    assert store.get_memories(users[0], topics=["career"]) == [career_goal()]
    assert store.get_memories(users[0], topics=["family"]) == []
    assert store.get_memories(users[0], topics=[]) == []
    assert len(store.get_memories(users[0], limit=1)) == 1


def test_data_survives_closing_and_reopening_the_connection(brain):
    """A new driver reads committed data even after the writing driver closes."""
    _, users, settings = brain
    with create_driver(settings) as writer:
        store = BrainStore(writer, settings.database)
        store.save_profile(users[0], UserProfile(name="Rahul"))
        store.save_memory(users[0], career_goal())
    # This second driver has no access to the first driver's Python objects.
    with create_driver(settings) as reader:
        store = BrainStore(reader, settings.database)
        assert store.get_profile(users[0]).name == "Rahul"
        assert store.get_memories(users[0]) == [career_goal()]


def test_combined_updates_and_profile_only_updates(brain):
    """The chat write operation supports a profile alone, then profile plus facts."""
    store, users, _ = brain
    store.save_updates(users[0], UserProfile(name="Rahul"), [])
    assert store.get_profile(users[0]).name == "Rahul"
    store.save_updates(users[0], UserProfile(birth_place="Delhi"), [career_goal()])
    assert store.get_profile(users[0]).name == "Rahul"
    assert store.get_profile(users[0]).birth_place == "Delhi"
    assert store.get_memories(users[0]) == [career_goal()]


def test_combined_update_is_rolled_back_if_query_fails(brain, monkeypatch):
    """An error later in the write query must not leave a partially patched profile."""
    from app.brain import BrainError

    store, users, _ = brain
    store.save_profile(users[0], UserProfile(name="Before"))
    fact = career_goal()
    # Simulate a bad serialized value after model validation. Neo4j cannot store
    # a nested map as a node property, so the single transaction must roll back.
    monkeypatch.setattr(MemoryFact, "model_dump", lambda self: {
        "key": "career_change", "content": {"invalid": "nested property"}
    })
    with pytest.raises(BrainError):
        store.save_updates(users[0], UserProfile(name="After"), [fact])
    assert store.get_profile(users[0]).name == "Before"
    assert store.get_memories(users[0]) == []
