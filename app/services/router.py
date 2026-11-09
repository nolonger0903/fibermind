from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import Intent, IntentResult, Route


@dataclass(slots=True)
class RouteDecision:
    route: Route
    reason: str


class IntentRouter:
    """Layered routing. Thresholds are injected from Settings (configurable, no magic numbers)."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.55,
        standard_min_confidence: float = 0.75,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.standard_min_confidence = standard_min_confidence

    def decide(
        self,
        query: str,
        intent: IntentResult,
        cache_hit: bool,
        standard_hit: bool,
        has_history: bool = False,
    ) -> RouteDecision:
        if cache_hit:
            return RouteDecision(Route.CACHE, "normalized query cache hit")
        if intent.intent == Intent.SMALL_TALK:
            return RouteDecision(Route.DIRECT, "non-domain conversational intent")
        if intent.intent == Intent.UNSUPPORTED:
            return RouteDecision(Route.DIRECT, "request is outside the optical-network domain")
        if intent.intent == Intent.HISTORY_QUERY:
            return RouteDecision(Route.DIRECT, "session history intent")
        if intent.intent == Intent.AMBIGUOUS:
            # 多轮追问中的模糊指代交给 Query Rewrite 用会话历史消解；
            # 首轮就信息不足时才 clarify 追问。
            if has_history:
                return RouteDecision(
                    Route.RAG,
                    "ambiguous follow-up resolved with multi-turn context via rewrite",
                )
            return RouteDecision(
                Route.CLARIFY,
                "intent is ambiguous (multiple competing signals or insufficient evidence)",
            )
        if intent.confidence < self.confidence_threshold:
            if has_history:
                return RouteDecision(
                    Route.RAG,
                    f"low confidence {intent.confidence:.2f} but multi-turn context available",
                )
            return RouteDecision(
                Route.CLARIFY,
                f"intent confidence {intent.confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}",
            )
        if standard_hit and intent.confidence >= self.standard_min_confidence:
            return RouteDecision(Route.STANDARD, "high-confidence deterministic answer")
        if len(query.strip()) < 4 or (
            intent.intent == Intent.FAULT_DIAGNOSIS
            and not intent.entities
            and len(query.strip()) < 10
        ):
            return RouteDecision(Route.CLARIFY, "fault description lacks actionable evidence")
        return RouteDecision(Route.RAG, "complex or long-tail domain query")
