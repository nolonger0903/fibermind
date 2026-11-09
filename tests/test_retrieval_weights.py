import pytest
from pydantic import ValidationError

from app.config import Settings
from app.infrastructure.knowledge import JsonKnowledgeRepository
from app.services.chunking import ParentChildChunker
from app.services.retrieval import InMemoryHybridRetriever
from app.utils.retrieval_weights import normalize_fusion_weights


def test_weights_are_normalized_to_unit_sum() -> None:
    weights = normalize_fusion_weights(0.7, 0.3)
    assert weights.dense == pytest.approx(0.7)
    assert weights.sparse == pytest.approx(0.3)
    assert weights.dense + weights.sparse == pytest.approx(1.0)

    scaled = normalize_fusion_weights(7.0, 3.0)
    assert scaled.dense == pytest.approx(0.7)
    assert scaled.sparse == pytest.approx(0.3)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        normalize_fusion_weights(-0.1, 1.1)
    with pytest.raises(ValueError, match="must be >= 0"):
        normalize_fusion_weights(1.0, -0.2)


def test_all_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        normalize_fusion_weights(0.0, 0.0)


def test_settings_rejects_illegal_weight_combinations() -> None:
    with pytest.raises(ValidationError):
        Settings(dense_weight=-1, sparse_weight=2)
    with pytest.raises(ValidationError):
        Settings(dense_weight=0, sparse_weight=0)


@pytest.mark.asyncio
async def test_in_memory_retriever_honors_both_weights() -> None:
    settings = Settings(_env_file=None)
    documents = JsonKnowledgeRepository(settings.knowledge_path).list_documents()
    chunks = [chunk for document in documents for chunk in ParentChildChunker().split(document)]

    dense_only = InMemoryHybridRetriever(chunks, dense_weight=1.0, sparse_weight=0.0)
    sparse_only = InMemoryHybridRetriever(chunks, dense_weight=0.0, sparse_weight=1.0)
    balanced = InMemoryHybridRetriever(chunks, dense_weight=0.55, sparse_weight=0.45)

    query = "ONU 掉线 接收光功率 范围"  # 不含告警码，避免 exact bonus 干扰断言

    dense_results = await dense_only.search(query, 5)
    sparse_results = await sparse_only.search(query, 5)
    balanced_results = await balanced.search(query, 5)

    # 单权重退化时 fusion 应严格等于对应分量
    for item in dense_results:
        assert item.fusion_score == pytest.approx(item.dense_score)
    for item in sparse_results:
        assert item.fusion_score == pytest.approx(item.sparse_score)
    # 均衡权重时 fusion 等于两个分量的加权和
    for item in balanced_results:
        expected = 0.55 * item.dense_score + 0.45 * item.sparse_score
        assert item.fusion_score == pytest.approx(expected)


def test_default_settings_weights_are_valid() -> None:
    settings = Settings(_env_file=None)
    weights = normalize_fusion_weights(settings.dense_weight, settings.sparse_weight)
    assert 0 < weights.dense < 1
    assert 0 < weights.sparse < 1
