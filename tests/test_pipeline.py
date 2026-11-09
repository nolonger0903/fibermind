import pytest

from app.config import Settings
from app.container import build_container


@pytest.mark.asyncio
async def test_standard_answer_then_cache_hit() -> None:
    container = build_container(Settings())

    first = await container.pipeline.run("LOS是什么意思？", "session-cache")
    second = await container.pipeline.run("LOS是什么意思？", "session-cache")

    assert first["route"] == "standard_knowledge"
    assert second["route"] == "cache"
    assert second["cached"] is True
    assert first["answer"] == second["answer"]


@pytest.mark.asyncio
async def test_complex_fault_runs_complete_rag_pipeline() -> None:
    container = build_container(Settings())

    response = await container.pipeline.run(
        "ONU频繁闪断，收光功率-29 dBm，昨天还报了LOS，应该怎么排查？",
        "session-rag",
    )

    assert response["route"] == "rag"
    assert response["diagnosis"]["grounded"] is True
    assert response["citations"]
    citation_ids = {citation["document_id"] for citation in response["citations"]}
    assert {"kb-onu-los-001", "kb-onu-flap-002"}.issubset(citation_ids)
    assert "1. 1." not in response["answer"]
    assert [step["name"] for step in response["trace"]] == [
        "intent_classification",
        "route_decision",
        "query_optimization",
        "hybrid_retrieval",
        "rerank",
        "grounded_generation",
    ]


@pytest.mark.asyncio
async def test_follow_up_expands_previous_user_context() -> None:
    container = build_container(Settings())
    session_id = "session-follow-up"
    await container.pipeline.run("ONU 报 LOS 而且收光功率低", session_id)

    response = await container.pipeline.run("那应该先检查哪里？", session_id)
    optimize_step = next(step for step in response["trace"] if step["name"] == "query_optimization")

    assert optimize_step["detail"]["context_expanded"] is True
    citation_ids = {citation["document_id"] for citation in response["citations"]}
    assert "kb-onu-los-001" in citation_ids


def test_parent_diversity_keeps_only_best_child_per_document() -> None:
    from app.domain.models import KnowledgeChunk, RetrievalResult
    from app.services.pipeline import DiagnosisPipeline

    def result(chunk_id: str, parent_id: str, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk=KnowledgeChunk(
                id=chunk_id,
                parent_id=parent_id,
                title=chunk_id,
                content=chunk_id,
                metadata={},
            ),
            rerank_score=score,
        )

    candidates = [
        result("a-1", "a", 0.99),
        result("a-2", "a", 0.95),
        result("b-1", "b", 0.90),
        result("c-1", "c", 0.80),
    ]

    selected = DiagnosisPipeline._select_parent_diverse(candidates, top_n=3)

    assert [item.chunk.id for item in selected] == ["a-1", "b-1", "c-1"]
