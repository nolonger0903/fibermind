from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from app.domain.models import Intent, Route, TraceStep
from app.prompts.clarify import build_clarify_reply
from app.services.generation import GeneratedAnswer
from app.utils.text import normalize_text

EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class DiagnosisPipeline:
    def __init__(
        self,
        *,
        intent_classifier: Any,
        router: Any,
        cache: Any,
        standard_answers: Any,
        query_optimizer: Any,
        retriever: Any,
        reranker: Any,
        generator: Any,
        sessions: Any,
        retrieval_top_k: int,
        rerank_top_n: int,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.router = router
        self.cache = cache
        self.standard_answers = standard_answers
        self.query_optimizer = query_optimizer
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.sessions = sessions
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n
        self.request_count = 0
        self.route_counts: dict[str, int] = {}

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        filters: dict[str, str] | None = None,
        emit: EventCallback | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = str(uuid4())
        session_id = session_id or str(uuid4())
        trace: list[TraceStep] = []
        self.request_count += 1

        async def send(event: str, data: dict[str, Any]) -> None:
            if emit:
                await emit(event, {"request_id": request_id, **data})

        async def stream_fallback(text: str) -> None:
            if not emit:
                return
            for start in range(0, len(text), 18):
                await send("answer_delta", {"text": text[start : start + 18]})

        await send("request_started", {"session_id": session_id})
        await self.sessions.append(session_id, "user", query)

        step_started = time.perf_counter()
        intent = self.intent_classifier.predict(query)
        intent_detail: dict[str, Any] = {
            "intent": intent.intent.value,
            "confidence": round(intent.confidence, 3),
            "entities": intent.entities,
            "classifier": intent.classifier,
        }
        # BERT → rule 的降级事件必须透传到 trace，不允许静默 fallback。
        if hasattr(self.intent_classifier, "pending_fallback_events"):
            fallback_events = self.intent_classifier.pending_fallback_events()
            if fallback_events:
                intent_detail["fallback"] = fallback_events
        self._trace(trace, "intent_classification", step_started, intent_detail)
        await send("intent_completed", trace[-1].detail)

        cache_key = self._cache_key(query)
        step_started = time.perf_counter()
        cached_payload = await self.cache.get(cache_key)
        standard = self.standard_answers.match(query, intent.intent.value)
        # 路由前先取会话历史：多轮追问中的模糊指代优先交给 rewrite 消解。
        history_messages = await self.sessions.list_messages(session_id, limit=8)
        history = [
            {"role": item.role, "content": item.content}
            for item in history_messages[:-1]
            if item.role in {"user", "assistant"}
        ]
        decision = self.router.decide(
            query,
            intent,
            cached_payload is not None,
            standard is not None,
            has_history=bool(history),
        )
        self.route_counts[decision.route.value] = self.route_counts.get(decision.route.value, 0) + 1
        self._trace(
            trace,
            "route_decision",
            step_started,
            {
                "route": decision.route.value,
                "reason": decision.reason,
                "cache_hit": cached_payload is not None,
                "has_history": bool(history),
            },
        )
        await send("route_completed", trace[-1].detail)

        if decision.route == Route.CACHE:
            response = self._cached_response(
                cached_payload, request_id, session_id, intent, decision, trace, started
            )
            await self.sessions.append(
                session_id, "assistant", response["answer"], {"route": "cache"}
            )
            await stream_fallback(response["answer"])
            await send(
                "response_completed",
                {"cached": True, "total_latency_ms": response["total_latency_ms"]},
            )
            return response

        if decision.route == Route.STANDARD:
            generated = GeneratedAnswer(
                answer=str(standard["answer"]), confidence=0.98, grounded=True
            )
            response = self._build_response(
                request_id, session_id, intent, decision, generated, [], trace, started
            )
            await self.cache.set(cache_key, self._cache_payload(response))
            await self.sessions.append(
                session_id, "assistant", response["answer"], {"route": "standard"}
            )
            await stream_fallback(response["answer"])
            await send(
                "response_completed",
                {"cached": False, "total_latency_ms": response["total_latency_ms"]},
            )
            return response

        if decision.route == Route.DIRECT:
            answer = await self._direct_answer(intent.intent, session_id)
            generated = GeneratedAnswer(answer=answer, confidence=0.99, grounded=True)
            response = self._build_response(
                request_id, session_id, intent, decision, generated, [], trace, started
            )
            await self.sessions.append(session_id, "assistant", answer, {"route": "direct"})
            await stream_fallback(answer)
            await send(
                "response_completed",
                {"cached": False, "total_latency_ms": response["total_latency_ms"]},
            )
            return response

        if decision.route == Route.CLARIFY:
            # Clarify 策略：按缺失槽位生成针对性追问，而不是固定话术。
            known_slots = set(intent.entities)
            missing_slots = [
                slot
                for slot in ("device_type", "alarm_codes", "power_values", "symptoms")
                if slot not in known_slots
            ]
            generated = GeneratedAnswer(
                answer=build_clarify_reply(missing_slots),
                confidence=0.3,
                grounded=True,
            )
            response = self._build_response(
                request_id, session_id, intent, decision, generated, [], trace, started
            )
            await self.sessions.append(
                session_id, "assistant", generated.answer, {"route": "clarify"}
            )
            await stream_fallback(generated.answer)
            await send(
                "response_completed",
                {"cached": False, "total_latency_ms": response["total_latency_ms"]},
            )
            return response

        step_started = time.perf_counter()
        # Query 优化（Rewrite / Sub-Query / HyDE）：优先 LLM，失败自动降级规则版并标注。
        optimized = await self.query_optimizer.optimize(query, intent, history)
        retrieval_query = optimized.rewritten
        if optimized.hyde:
            retrieval_query = f"{retrieval_query} {optimized.hyde}"
        self._trace(
            trace,
            "query_optimization",
            step_started,
            {
                "original": optimized.original,
                "rewritten": optimized.rewritten,
                "sub_queries": optimized.sub_queries,
                "hyde": optimized.hyde,
                "strategies": optimized.strategies,
                "backend": optimized.backend,
                "fallback_reason": optimized.fallback_reason,
                "context_expanded": optimized.context_expanded,
            },
        )
        await send("query_optimized", trace[-1].detail)

        step_started = time.perf_counter()
        search_queries = list(dict.fromkeys([retrieval_query, *optimized.sub_queries]))
        search_results = [
            await self.retriever.search(search_query, self.retrieval_top_k, filters)
            for search_query in search_queries
        ]
        candidates = self._merge_candidates(search_results)[: self.retrieval_top_k]
        self._trace(
            trace,
            "hybrid_retrieval",
            step_started,
            {
                "candidates": len(candidates),
                "query_count": len(search_results),
                "top": [
                    {
                        "title": item.chunk.title,
                        "dense": round(item.dense_score, 3),
                        "sparse": round(item.sparse_score, 3),
                        "fusion": round(item.fusion_score, 3),
                    }
                    for item in candidates[:5]
                ],
            },
        )
        await send("retrieval_completed", trace[-1].detail)

        step_started = time.perf_counter()
        ranked_chunks = await self.reranker.rerank(query, candidates, len(candidates))
        reranked = self._select_parent_diverse(ranked_chunks, self.rerank_top_n)
        self._trace(
            trace,
            "rerank",
            step_started,
            {
                "selected": len(reranked),
                "top": [
                    {"title": item.chunk.title, "score": round(item.rerank_score, 3)}
                    for item in reranked
                ],
            },
        )
        await send("rerank_completed", trace[-1].detail)

        step_started = time.perf_counter()

        async def emit_token(token: str) -> None:
            await send("answer_delta", {"text": token})

        generated = await self.generator.generate(
            query,
            intent.intent,
            reranked,
            history,
            on_token=emit_token if emit else None,
        )
        if emit and not getattr(self.generator, "native_streaming", False):
            await stream_fallback(generated.answer)
        self._trace(
            trace,
            "grounded_generation",
            step_started,
            {"grounded": generated.grounded, "confidence": round(generated.confidence, 3)},
        )
        await send("generation_completed", trace[-1].detail)

        response = self._build_response(
            request_id, session_id, intent, decision, generated, reranked, trace, started
        )
        if intent.intent in {Intent.KNOWLEDGE_QA, Intent.ALARM_EXPLANATION} and generated.grounded:
            await self.cache.set(cache_key, self._cache_payload(response))
        await self.sessions.append(
            session_id,
            "assistant",
            generated.answer,
            {"route": "rag", "request_id": request_id},
        )
        await send(
            "response_completed",
            {"cached": False, "total_latency_ms": response["total_latency_ms"]},
        )
        return response

    async def _direct_answer(self, intent: Intent, session_id: str) -> str:
        if intent == Intent.HISTORY_QUERY:
            messages = await self.sessions.list_messages(session_id, limit=8)
            user_messages = [item.content for item in messages if item.role == "user"][:-1]
            if not user_messages:
                return "当前会话还没有更早的问题记录。"
            return "本次会话中你之前问过：" + "；".join(user_messages[-3:])
        if intent == Intent.UNSUPPORTED:
            return "这个请求不属于光网络运维范围。我可以协助处理 ONU、OLT、光功率、告警和链路故障。"
        return "你好，我可以协助分析光功率、LOS/LOF、ONU 掉线、误码和光链路类故障。"

    @staticmethod
    def _merge_candidates(result_groups: list[list[Any]]) -> list[Any]:
        merged: dict[str, Any] = {}
        for results in result_groups:
            for result in results:
                existing = merged.get(result.chunk.id)
                if existing is None:
                    merged[result.chunk.id] = result
                    continue
                existing.dense_score = max(existing.dense_score, result.dense_score)
                existing.sparse_score = max(existing.sparse_score, result.sparse_score)
                existing.fusion_score = max(existing.fusion_score, result.fusion_score)
        return sorted(merged.values(), key=lambda item: item.fusion_score, reverse=True)

    @staticmethod
    def _select_parent_diverse(candidates: list[Any], top_n: int) -> list[Any]:
        """Keep the strongest child per parent so duplicate chunks cannot crowd out evidence."""
        selected = []
        seen_parents: set[str] = set()
        for candidate in candidates:
            if candidate.chunk.parent_id in seen_parents:
                continue
            seen_parents.add(candidate.chunk.parent_id)
            selected.append(candidate)
            if len(selected) >= top_n:
                break
        return selected

    @staticmethod
    def _trace(trace: list[TraceStep], name: str, started: float, detail: dict[str, Any]) -> None:
        trace.append(
            TraceStep(
                name=name,
                status="completed",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=detail,
            )
        )

    @staticmethod
    def _cache_key(query: str) -> str:
        digest = hashlib.sha256(normalize_text(query).encode("utf-8")).hexdigest()[:24]
        return f"fibermind:answer:v1:{digest}"

    @staticmethod
    def _cache_payload(response: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": response["answer"],
            "diagnosis": response["diagnosis"],
            "citations": response["citations"],
            "source_route": response["route"],
        }

    def _cached_response(
        self,
        payload: dict[str, Any],
        request_id: str,
        session_id: str,
        intent: Any,
        decision: Any,
        trace: list[TraceStep],
        started: float,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "answer": payload["answer"],
            "route": decision.route.value,
            "route_reason": (
                f"{decision.reason}; original route={payload.get('source_route', 'unknown')}"
            ),
            "intent": self._intent_view(intent),
            "diagnosis": payload["diagnosis"],
            "citations": payload.get("citations", []),
            "trace": [asdict(item) for item in trace],
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "cached": True,
        }

    def _build_response(
        self,
        request_id: str,
        session_id: str,
        intent: Any,
        decision: Any,
        generated: GeneratedAnswer,
        contexts: list[Any],
        trace: list[TraceStep],
        started: float,
    ) -> dict[str, Any]:
        citations = []
        seen: set[str] = set()
        for result in contexts:
            if result.chunk.parent_id in seen:
                continue
            seen.add(result.chunk.parent_id)
            citations.append(
                {
                    "index": len(citations) + 1,
                    "document_id": result.chunk.parent_id,
                    "title": result.chunk.title,
                    "source": result.chunk.metadata["source"],
                    "excerpt": result.chunk.content[:180],
                    "dense_score": round(result.dense_score, 4),
                    "sparse_score": round(result.sparse_score, 4),
                    "fusion_score": round(result.fusion_score, 4),
                    "rerank_score": round(result.rerank_score, 4),
                }
            )
        return {
            "request_id": request_id,
            "session_id": session_id,
            "answer": generated.answer,
            "route": decision.route.value,
            "route_reason": decision.reason,
            "intent": self._intent_view(intent),
            "diagnosis": {
                "possible_causes": generated.possible_causes,
                "next_steps": generated.next_steps,
                "confidence": round(generated.confidence, 3),
                "grounded": generated.grounded,
            },
            "citations": citations,
            "trace": [asdict(item) for item in trace],
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "cached": False,
        }

    @staticmethod
    def _intent_view(intent: Any) -> dict[str, Any]:
        return {
            "name": intent.intent.value,
            "confidence": round(intent.confidence, 3),
            "classifier": intent.classifier,
            "entities": intent.entities,
        }
