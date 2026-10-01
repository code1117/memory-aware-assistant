"""Choose bounded, relevant context with transparent rules rather than a framework."""

import re

from app.memory import ConversationMessage
from app.models import UserProfile


# Extraction uses these same topic names so saved facts can be retrieved by rules.
TOPIC_WORDS = {
    "career": {"career", "job", "jobs", "work", "interview", "promotion", "business", "entrepreneurship"},
    "relationships": {"relationship", "relationships", "partner", "dating", "marriage", "spouse"},
    "family": {"family", "mother", "father", "parents", "child", "children", "sister", "brother"},
    "health": {"health", "fitness", "exercise", "sleep", "illness"},
    "finance": {"finance", "finances", "money", "savings", "salary", "budget", "debt"},
    "education": {"education", "study", "studying", "college", "university", "exam", "course"},
    "hobbies": {"hobby", "hobbies", "cricket", "sport", "sports", "music", "pet", "pets", "mouse"},
    "astrology": {"astrology", "zodiac", "horoscope", "birth", "chart", "sun", "sign"},
    "preferences": {"preference", "preferences", "prefer", "language", "food", "vegetarian"},
}
FOLLOW_UP = re.compile(r"\b(why|that|it|those|third|second|first|actually|instead|correction|meant|mean)\b")


def select_topics(message: str, history: list[ConversationMessage]) -> list[str] | None:
    """Use explicit topics first; inherit recent user topics only for follow-ups.

    None requests a bounded general memory overview; [] means no relevant topic.
    Keyword matching is deliberately simple and primarily supports English.
    """
    words = set(re.findall(r"[a-z]+", message.lower()))
    topics = [topic for topic, keywords in TOPIC_WORDS.items() if words & keywords]
    if topics:
        return topics
    if re.search(r"\b(remember|know)\b.*\b(about me|my goals|my plans)\b", message.lower()):
        return None
    if FOLLOW_UP.search(message.lower()):
        for item in reversed(history):
            if item["role"] == "user":
                previous = set(re.findall(r"[a-z]+", item["content"].lower()))
                topics = [topic for topic, keywords in TOPIC_WORDS.items() if previous & keywords]
                if topics:
                    return topics
    return []


def merge_profile(stored: UserProfile | None, supplied: UserProfile | None) -> UserProfile:
    """Overlay current explicit profile fields, preserving unspecified stored values."""
    values = stored.model_dump(mode="json") if stored else {}
    if supplied is not None:
        values.update(supplied.model_dump(mode="json", exclude_unset=True))
    return UserProfile.model_validate(values)


def select_profile(profile: UserProfile, message: str, topics: list[str] | None) -> dict:
    """Include name/language, plus birth details only for astrology or profile questions."""
    fields = {"name", "preferred_language"}
    text = message.lower()
    if topics is None or "astrology" in (topics or []) or re.search(r"\bprofile\b", text):
        fields = set(UserProfile.model_fields)
    else:
        if re.search(r"\b(birthday|born)\b", text):
            fields.update({"date_of_birth", "birth_place", "time_of_birth"})
        if "birthplace" in text:
            fields.add("birth_place")
    return profile.model_dump(mode="json", include=fields, exclude_none=True)
