from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None
    filters: dict[str, str] | None = None
    debug: bool = True


class IntentView(BaseModel):
    name: str
    confidence: float
    classifier: str
    entities: dict[str, Any]


class CitationView(BaseModel):
    index: int
    document_id: str
    title: str
    source: str
    excerpt: str
    dense_score: float
    sparse_score: float
    fusion_score: float
    rerank_score: float


class TraceView(BaseModel):
    name: str
    status: str
    latency_ms: float
    detail: dict[str, Any]


class DiagnosisView(BaseModel):
    possible_causes: list[str] = []
    next_steps: list[str] = []
    confidence: float
    grounded: bool


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    route: str
    route_reason: str
    intent: IntentView
    diagnosis: DiagnosisView
    citations: list[CitationView]
    trace: list[TraceView]
    total_latency_ms: float
    cached: bool = False


class IntentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=30)
    filters: dict[str, str] | None = None


class MessageView(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    mode: str
    services: dict[str, str]
