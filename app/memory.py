"""Keep bounded, temporary conversation history in this application process.

History resets on restart or development reload and is not shared by multiple
server workers. The local demo expects sequential requests within each session.
"""

from collections import OrderedDict, deque
from threading import Lock
from typing import Literal, TypedDict


class ConversationMessage(TypedDict):
    """Represent one user or assistant message without provider-specific types."""

    role: Literal["user", "assistant"]
    content: str


class ConversationMemory:
    """Retain recent exchanges, evicting the least recently used session if full."""

    def __init__(self, max_turns: int = 5, max_sessions: int = 100) -> None:
        """Set limits; each turn contains one user message and one assistant reply."""
        if max_turns < 1 or max_sessions < 1:
            raise ValueError("Memory limits must be positive.")
        self._max_messages = max_turns * 2
        self._max_sessions = max_sessions
        self._histories: OrderedDict[
            tuple[str, str], deque[ConversationMessage]
        ] = OrderedDict()
        # Protect reads and writes from FastAPI's worker threads. This does not
        # serialize entire LLM calls: same-session requests should be sequential.
        self._lock = Lock()

    def get_history(self, user_id: str, session_id: str) -> list[ConversationMessage]:
        """Return a copy of this user's session history, or an empty list."""
        key = (user_id, session_id)
        with self._lock:
            history = self._histories.get(key)
            if history is None:
                return []
            self._histories.move_to_end(key)
            # Copies prevent callers from accidentally modifying stored history.
            return [message.copy() for message in history]

    def add_exchange(
        self, user_id: str, session_id: str, message: str, answer: str
    ) -> None:
        """Save a successful exchange, discarding the oldest complete turns if full."""
        key = (user_id, session_id)
        with self._lock:
            if key not in self._histories:
                if len(self._histories) >= self._max_sessions:
                    self._histories.popitem(last=False)
                self._histories[key] = deque(maxlen=self._max_messages)
            self._histories[key].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ])
            self._histories.move_to_end(key)


# Reuse one store across requests; creating it inside the endpoint would lose
# the history after every request.
conversation_memory = ConversationMemory()
