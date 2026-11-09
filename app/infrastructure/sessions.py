from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from app.domain.models import Message


class MemorySessionRepository:
    def __init__(self) -> None:
        self._messages: dict[str, list[Message]] = defaultdict(list)

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        message = Message(
            id=str(uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages[session_id].append(message)
        return message

    async def list_messages(self, session_id: str, limit: int = 20) -> list[Message]:
        return self._messages.get(session_id, [])[-limit:]

    async def count_sessions(self) -> int:
        return len(self._messages)
