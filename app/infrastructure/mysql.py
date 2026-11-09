from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.domain.models import Message


class MySQLSessionRepository:
    """SQLAlchemy adapter. Blocking driver calls are isolated from the FastAPI event loop."""

    def __init__(self, dsn: str) -> None:
        try:
            from sqlalchemy import JSON, DateTime, String, Text, create_engine, func, select
            from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
        except ImportError as exc:
            raise RuntimeError("Install FiberMind with the 'infra' extra to use MySQL") from exc

        class Base(DeclarativeBase):
            pass

        class MessageRow(Base):
            __tablename__ = "messages"

            id: Mapped[str] = mapped_column(String(36), primary_key=True)
            session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
            role: Mapped[str] = mapped_column(String(16), nullable=False)
            content: Mapped[str] = mapped_column(Text, nullable=False)
            created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
            extra: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)

        self.select = select
        self.func = func
        self.MessageRow = MessageRow
        self.engine = create_engine(dsn, pool_pre_ping=True, pool_recycle=1800)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

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

        def write() -> None:
            with self.session_factory.begin() as db:
                db.add(
                    self.MessageRow(
                        id=message.id,
                        session_id=message.session_id,
                        role=message.role,
                        content=message.content,
                        created_at=message.created_at,
                        extra=message.metadata,
                    )
                )

        await asyncio.to_thread(write)
        return message

    async def list_messages(self, session_id: str, limit: int = 20) -> list[Message]:
        def read() -> list[Message]:
            statement = (
                self.select(self.MessageRow)
                .where(self.MessageRow.session_id == session_id)
                .order_by(self.MessageRow.created_at.desc())
                .limit(limit)
            )
            with self.session_factory() as db:
                rows = list(db.scalars(statement))
            return [
                Message(
                    id=row.id,
                    session_id=row.session_id,
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                    metadata=row.extra,
                )
                for row in reversed(rows)
            ]

        return await asyncio.to_thread(read)

    async def count_sessions(self) -> int:
        def count() -> int:
            statement = self.select(self.func.count(self.func.distinct(self.MessageRow.session_id)))
            with self.session_factory() as db:
                return int(db.scalar(statement) or 0)

        return await asyncio.to_thread(count)
