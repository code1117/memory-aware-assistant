"""Check rule-based relevance and selective profile use without external services."""

import pytest

from app.context import merge_profile, select_profile, select_topics
from app.models import UserProfile


@pytest.mark.parametrize("message,expected", [
    ("What are my career plans?", ["career"]),
    ("Tell me about cricket", ["hobbies"]),
    ("What do you remember about me?", None),
    ("What is 2 + 2?", []),
    ("How is the weather?", []),
])
def test_explicit_topic_selection(message, expected):
    """Clear subjects route to their topic; unrelated questions get no memories."""
    assert select_topics(message, []) == expected


def test_follow_up_inherits_topic_but_explicit_new_subject_wins():
    """Only a reference-style follow-up should inherit the previous career topic."""
    history = [{"role": "user", "content": "I want to change jobs."}]
    assert select_topics("Actually, I mean 2028.", history) == ["career"]
    assert select_topics("Tell me about cricket", history) == ["hobbies"]
    assert select_topics("What is 2 + 2?", history) == []


def test_profile_selection_does_not_send_birth_data_to_unrelated_questions():
    """General personalization needs fewer profile fields than astrology questions."""
    profile = UserProfile(name="Rahul", birth_place="Delhi", zodiac_sign="Leo")
    assert select_profile(profile, "Career advice", ["career"]) == {"name": "Rahul"}
    assert select_profile(profile, "What is my name?", []) == {"name": "Rahul"}
    assert select_profile(profile, "What is my zodiac?", ["astrology"])["zodiac_sign"] == "Leo"


def test_profile_patch_preserves_unspecified_fields_and_supports_clearing():
    """An explicit null clears a field while an omitted field stays unchanged."""
    stored = UserProfile(name="Rahul", birth_place="Delhi")
    updated = merge_profile(stored, UserProfile(birth_place=None))
    assert updated.name == "Rahul"
    assert updated.birth_place is None
