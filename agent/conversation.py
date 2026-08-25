"""
agent/conversation.py

Bounded per-session conversation state.
Sessions are stored in-process and never shared across session IDs.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

MAX_TURNS = 10   # keep only the last N turns in context


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    order_id: str | None = None   # order ID mentioned in this turn, if any


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[Turn] = field(default_factory=list)
    last_order_id: str | None = None    # most recent successfully looked-up order ID
    last_order_result: dict | None = None  # sanitized lookup_order result dict for last_order_id

    def add_turn(self, role: str, content: str, order_id: str | None = None):
        self.turns.append(Turn(role=role, content=content, order_id=order_id))
        if order_id:
            self.last_order_id = order_id
        # Trim to bounded window
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]

    def history_for_llm(self) -> list[dict]:
        """Return turns formatted for the Gemini multi-turn messages list."""
        return [
            {"role": t.role, "parts": [{"text": t.content}]}
            for t in self.turns
        ]


# ---------------------------------------------------------------------------
# Simple in-process session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def get_or_create_session(session_id: str | None = None) -> Session:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    s = Session(id=session_id or str(uuid.uuid4()))
    _sessions[s.id] = s
    return s


def clear_session(session_id: str):
    _sessions.pop(session_id, None)
