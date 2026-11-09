import json

import httpx
import pytest

from app.domain.models import KnowledgeChunk, RetrievalResult
from app.infrastructure.bge_api import BGEAPIHybridRetriever, BGEAPIReranker


def make_chunk(chunk_id: str, content: str, alarm_codes: list[str] | None = None) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        title=content,
        content=content,
        metadata={
            "tags": content.split(),
            "alarm_codes": alarm_codes or [],
            "device_type": "ONU",
        },
    )


@pytest.mark.asyncio
async def test_bge_api_retriever_embeds_chunks_once_and_honors_filters() -> None:
    chunks = [
        make_chunk("los", "ONU LOS 光功率", ["LOS"]),
        make_chunk("wifi", "无线网络 密码"),
    ]
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        batch_sizes.append(len(body["input"]))
        if len(body["input"]) == 2:
            vectors = [[1.0, 0.0], [0.0, 1.0]]
        else:
            vectors = [[1.0, 0.0]]
        return httpx.Response(
            200,
            json={
                "model": "BAAI/bge-m3",
                "data": [
                    {"index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
            },
        )

    retriever = BGEAPIHybridRetriever(
        chunks,
        base_url="https://api.example/v1",
        api_key="test-key",
        dense_weight=0.6,
        sparse_weight=0.4,
        transport=httpx.MockTransport(handler),
    )
    first = await retriever.search("ONU LOS", top_k=2)
    second = await retriever.search("LOS", top_k=2, filters={"device_type": "ONU"})

    assert first[0].chunk.id == "los"
    assert first[0].dense_score == pytest.approx(1.0)
    assert first[0].sparse_score > 0
    assert second
    assert batch_sizes == [2, 1, 1]


@pytest.mark.asyncio
async def test_bge_api_reranker_maps_remote_order_and_scores() -> None:
    candidates = [
        RetrievalResult(chunk=make_chunk("wifi", "无线网络 密码"), fusion_score=0.6),
        RetrievalResult(chunk=make_chunk("los", "ONU LOS 光功率"), fusion_score=0.5),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        body = json.loads(request.content)
        assert body["model"] == "BAAI/bge-reranker-v2-m3"
        assert body["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.97},
                    {"index": 0, "relevance_score": 0.03},
                ]
            },
        )

    reranker = BGEAPIReranker(
        base_url="https://api.example/v1",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    results = await reranker.rerank("ONU LOS", candidates, top_n=2)

    assert [item.chunk.id for item in results] == ["los", "wifi"]
    assert [item.rerank_score for item in results] == pytest.approx([0.97, 0.03])


def test_bge_api_backends_require_a_key() -> None:
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        BGEAPIHybridRetriever([], base_url="https://api.example/v1", api_key="")
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        BGEAPIReranker(base_url="https://api.example/v1", api_key="")
