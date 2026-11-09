from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Intent(str, Enum):
    KNOWLEDGE_QA = "knowledge_qa"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    OPERATION_GUIDE = "operation_guide"
    ALARM_EXPLANATION = "alarm_explanation"
    HISTORY_QUERY = "history_query"
    SMALL_TALK = "small_talk"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class Route(str, Enum):
    CACHE = "cache"
    STANDARD = "standard_knowledge"
    RAG = "rag"
    DIRECT = "direct"
    CLARIFY = "clarify"


@dataclass(slots=True)
class IntentResult:
    intent: Intent
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    classifier: str = "rule"
    signals: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    device_type: str
    fault_type: str
    alarm_codes: list[str]
    source: str
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KnowledgeChunk:
    id: str
    parent_id: str
    title: str
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievalResult:
    chunk: KnowledgeChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(slots=True)
class Message:
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceStep:
    name: str
    status: str
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)
