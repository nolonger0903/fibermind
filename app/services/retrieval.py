from __future__ import annotations

from collections import Counter

from app.domain.models import KnowledgeChunk, RetrievalResult
from app.utils.retrieval_weights import FusionWeights, normalize_fusion_weights
from app.utils.text import cosine_similarity, lexical_overlap, normalize_text, tokenize


class InMemoryHybridRetriever:
    """Deterministic dense/sparse baseline with the same contract as the Milvus adapter."""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
    ) -> None:
        self.chunks = chunks
        # 与 Milvus WeightedRanker 共用同一套校验/归一化逻辑，两个权重都真实生效。
        self.weights: FusionWeights = normalize_fusion_weights(dense_weight, sparse_weight)
        self._features = {
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
        query_tokens = tokenize(query)
        query_vector = Counter(query_tokens)
        normalized_query = normalize_text(query)
        results: list[RetrievalResult] = []
        for chunk in self.chunks:
            if filters and any(
                value and value.lower() not in str(chunk.metadata.get(key, "")).lower()
                for key, value in filters.items()
            ):
                continue
            document_tokens = list(self._features[chunk.id].elements())
            dense = cosine_similarity(query_vector, self._features[chunk.id])
            sparse = lexical_overlap(query_tokens, document_tokens)
            exact_bonus = 0.0
            for code in chunk.metadata.get("alarm_codes", []):
                if str(code).lower() in normalized_query:
                    exact_bonus = 0.22
                    break
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


class LocalReranker:
    """Feature-based reranker used for demo; BGE adapter can replace it without pipeline changes."""

    async def rerank(
        self, query: str, candidates: list[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        normalized_query = normalize_text(query)
        query_terms = set(tokenize(query))
        for rank, result in enumerate(candidates):
            title_terms = set(tokenize(result.chunk.title))
            title_score = len(query_terms & title_terms) / max(len(query_terms), 1)
            metadata_text = " ".join(
                [
                    str(result.chunk.metadata.get("device_type", "")),
                    str(result.chunk.metadata.get("fault_type", "")),
                    " ".join(result.chunk.metadata.get("tags", [])),
                ]
            )
            metadata_score = lexical_overlap(list(query_terms), tokenize(metadata_text))
            alarm_bonus = 0.0
            for code in result.chunk.metadata.get("alarm_codes", []):
                if str(code).lower() in normalized_query:
                    alarm_bonus = 0.2
                    break
            rank_prior = 1 / (rank + 1)
            result.rerank_score = min(
                1.0,
                0.58 * result.fusion_score
                + 0.16 * title_score
                + 0.12 * metadata_score
                + 0.04 * rank_prior
                + alarm_bonus,
            )
        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        return candidates[:top_n]
