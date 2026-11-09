from __future__ import annotations

import asyncio
import math
from collections import Counter
from typing import Any

import httpx

from app.domain.models import KnowledgeChunk, RetrievalResult
from app.utils.retrieval_weights import FusionWeights, normalize_fusion_weights
from app.utils.text import lexical_overlap, normalize_text, tokenize


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class BGEAPIHybridRetriever:
    """BGE-M3 API dense retrieval fused with a deterministic lexical sparse signal.

    OpenAI-compatible BGE embedding endpoints expose the 1024-dimensional dense
    vector but not BGE-M3 lexical weights. The true BGE-M3 dense+sparse Milvus
    backend remains ``MilvusHybridRetriever``; this backend provides a practical
    remote-model path without pretending the lexical score came from BGE-M3.
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        *,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-m3",
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
        timeout_seconds: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for the BGE API retriever")
        self.chunks = chunks
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.weights: FusionWeights = normalize_fusion_weights(dense_weight, sparse_weight)
        self._chunk_vectors: list[list[float]] | None = None
        self._embedding_lock = asyncio.Lock()
        self._sparse_features = {
            chunk.id: Counter(
                tokenize(f"{chunk.title}{chunk.content}{' '.join(chunk.metadata['tags'])}")
            )
            for chunk in chunks
        }

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        await self._ensure_chunk_vectors()
        query_vector = (await self._embed([query]))[0]
        query_tokens = tokenize(query)
        normalized_query = normalize_text(query)
        results: list[RetrievalResult] = []
        assert self._chunk_vectors is not None
        for chunk, chunk_vector in zip(self.chunks, self._chunk_vectors, strict=True):
            if filters and any(
                value and value.lower() not in str(chunk.metadata.get(key, "")).lower()
                for key, value in filters.items()
            ):
                continue
            dense = _cosine_similarity(query_vector, chunk_vector)
            document_tokens = list(self._sparse_features[chunk.id].elements())
            sparse = lexical_overlap(query_tokens, document_tokens)
            exact_bonus = 0.0
            if any(
                str(code).lower() in normalized_query
                for code in chunk.metadata.get("alarm_codes", [])
            ):
                exact_bonus = 0.22
            fusion = min(
                1.0,
                self.weights.dense * dense + self.weights.sparse * sparse + exact_bonus,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    dense_score=dense,
                    sparse_score=sparse,
                    fusion_score=fusion,
                )
            )
        results.sort(key=lambda item: item.fusion_score, reverse=True)
        return results[:top_k]

    async def _ensure_chunk_vectors(self) -> None:
        if self._chunk_vectors is not None:
            return
        async with self._embedding_lock:
            if self._chunk_vectors is None:
                texts = [f"{chunk.title}\n{chunk.content}" for chunk in self.chunks]
                self._chunk_vectors = await self._embed(texts)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, transport=self.transport
        ) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
        rows = sorted(payload["data"], key=lambda item: item.get("index", 0))
        vectors = [list(map(float, item["embedding"])) for item in rows]
        if len(vectors) != len(texts):
            raise ValueError(
                f"BGE embedding API returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class BGEAPIReranker:
    """Remote BGE cross-encoder reranker using a SiliconFlow-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout_seconds: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for the BGE API reranker")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def rerank(
        self, query: str, candidates: list[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        documents = [f"{item.chunk.title}\n{item.chunk.content}" for item in candidates]
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, transport=self.transport
        ) as client:
            response = await client.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(candidates)),
                    "return_documents": False,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        ranked: list[RetrievalResult] = []
        for row in payload.get("results", []):
            index = int(row["index"])
            if index < 0 or index >= len(candidates):
                raise ValueError(f"BGE rerank API returned invalid document index {index}")
            candidate = candidates[index]
            candidate.rerank_score = float(row["relevance_score"])
            ranked.append(candidate)
        if not ranked:
            raise ValueError("BGE rerank API returned no results")
        return ranked
