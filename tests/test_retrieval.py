import pytest

from app.config import Settings
from app.infrastructure.knowledge import JsonKnowledgeRepository
from app.services.chunking import ParentChildChunker
from app.services.retrieval import InMemoryHybridRetriever, LocalReranker


@pytest.mark.asyncio
async def test_los_query_retrieves_los_document_first() -> None:
    settings = Settings()
    documents = JsonKnowledgeRepository(settings.knowledge_path).list_documents()
    chunks = [chunk for document in documents for chunk in ParentChildChunker().split(document)]
    retriever = InMemoryHybridRetriever(chunks)

    candidates = await retriever.search("ONU 报 LOS 且接收光功率低", top_k=8)
    results = await LocalReranker().rerank("ONU 报 LOS 且接收光功率低", candidates, top_n=4)

    assert results[0].chunk.parent_id == "kb-onu-los-001"
    assert results[0].rerank_score >= results[-1].rerank_score
    assert results[0].dense_score > 0
    assert results[0].sparse_score > 0


@pytest.mark.asyncio
async def test_metadata_filter_limits_device_type() -> None:
    settings = Settings()
    documents = JsonKnowledgeRepository(settings.knowledge_path).list_documents()
    chunks = [chunk for document in documents for chunk in ParentChildChunker().split(document)]
    retriever = InMemoryHybridRetriever(chunks)

    results = await retriever.search("误码异常", top_k=10, filters={"device_type": "传输设备"})

    assert results
    assert all(item.chunk.metadata["device_type"] == "传输设备" for item in results)
