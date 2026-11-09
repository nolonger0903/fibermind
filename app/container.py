from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.infrastructure.cache import MemoryTTLCache, RedisCache
from app.infrastructure.knowledge import JsonKnowledgeRepository, StandardAnswerRepository
from app.infrastructure.sessions import MemorySessionRepository
from app.services.chunking import ParentChildChunker
from app.services.generation import GroundedTemplateGenerator, OpenAICompatibleGenerator
from app.services.intent import (
    BertIntentClassifier,
    ResilientIntentClassifier,
    RuleIntentClassifier,
)
from app.services.pipeline import DiagnosisPipeline
from app.services.query_optimizer import LLMQueryOptimizer, RuleQueryOptimizer
from app.services.retrieval import InMemoryHybridRetriever, LocalReranker
from app.services.router import IntentRouter

logger = logging.getLogger("fibermind.container")


@dataclass(slots=True)
class Container:
    settings: Settings
    knowledge: JsonKnowledgeRepository
    standard_answers: StandardAnswerRepository
    intent_classifier: Any
    cache: Any
    sessions: Any
    retriever: Any
    reranker: Any
    generator: Any
    query_optimizer: Any
    pipeline: DiagnosisPipeline
    chunks_count: int


def _build_intent_classifier(settings: Settings) -> Any:
    if settings.intent_backend != "bert":
        return RuleIntentClassifier()
    primary = None
    try:
        primary = BertIntentClassifier(settings.intent_model_path)
    except Exception as exc:  # noqa: BLE001 - 加载失败必须降级而不是让进程起不来
        logger.warning(
            "BERT intent model failed to load (%s: %s); falling back to rule classifier",
            type(exc).__name__,
            exc,
        )
    return ResilientIntentClassifier(
        primary,
        fallback_on_error=settings.intent_fallback_on_error,
    )


def _build_query_optimizer(settings: Settings) -> Any:
    llm_ready = settings.llm_backend in {"openai", "deepseek"} and settings.rewrite_use_llm
    if llm_ready:
        return LLMQueryOptimizer(
            base_url=settings.resolved_llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.resolved_llm_model,
            provider=settings.llm_backend,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return RuleQueryOptimizer()


def build_container(settings: Settings) -> Container:
    knowledge = JsonKnowledgeRepository(settings.knowledge_path)
    standard_answers = StandardAnswerRepository(settings.standard_answers_path)
    documents = knowledge.list_documents()
    chunker = ParentChildChunker()
    chunks = [chunk for document in documents for chunk in chunker.split(document)]

    intent_classifier = _build_intent_classifier(settings)
    cache = (
        RedisCache(settings.redis_url, settings.cache_ttl_seconds)
        if settings.cache_backend == "redis"
        else MemoryTTLCache(settings.cache_ttl_seconds)
    )
    if settings.session_backend == "mysql":
        from app.infrastructure.mysql import MySQLSessionRepository

        sessions = MySQLSessionRepository(settings.mysql_dsn)
    else:
        sessions = MemorySessionRepository()

    if settings.retrieval_backend == "milvus":
        from app.infrastructure.milvus import MilvusHybridRetriever

        retriever = MilvusHybridRetriever(settings, chunks)
    elif settings.retrieval_backend == "bge_api":
        from app.infrastructure.bge_api import BGEAPIHybridRetriever

        retriever = BGEAPIHybridRetriever(
            chunks,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    else:
        retriever = InMemoryHybridRetriever(
            chunks, settings.dense_weight, settings.sparse_weight
        )

    if settings.rerank_backend == "bge":
        from app.infrastructure.bge import BGEReranker

        reranker = BGEReranker(settings.rerank_model)
    elif settings.rerank_backend == "bge_api":
        from app.infrastructure.bge_api import BGEAPIReranker

        reranker = BGEAPIReranker(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    else:
        reranker = LocalReranker()

    generator = (
        OpenAICompatibleGenerator(
            base_url=settings.resolved_llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.resolved_llm_model,
            provider=settings.llm_backend,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        if settings.llm_backend in {"openai", "deepseek"}
        else GroundedTemplateGenerator()
    )
    query_optimizer = _build_query_optimizer(settings)
    pipeline = DiagnosisPipeline(
        intent_classifier=intent_classifier,
        router=IntentRouter(
            confidence_threshold=settings.intent_confidence_threshold,
            standard_min_confidence=settings.router_standard_min_confidence,
        ),
        cache=cache,
        standard_answers=standard_answers,
        query_optimizer=query_optimizer,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        sessions=sessions,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_n=settings.rerank_top_n,
    )
    return Container(
        settings=settings,
        knowledge=knowledge,
        standard_answers=standard_answers,
        intent_classifier=intent_classifier,
        cache=cache,
        sessions=sessions,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        query_optimizer=query_optimizer,
        pipeline=pipeline,
        chunks_count=len(chunks),
    )
