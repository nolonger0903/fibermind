from __future__ import annotations

import asyncio

from app.domain.models import RetrievalResult


class BGEReranker:
    """BGE cross-encoder reranker; loaded only when RERANK_BACKEND=bge."""

    def __init__(self, model_name: str) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "Install FiberMind with the 'ml' extra to use BGE reranking"
            ) from exc
        self.model = FlagReranker(model_name, use_fp16=True)

    async def rerank(
        self, query: str, candidates: list[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        pairs = [[query, f"{item.chunk.title}\n{item.chunk.content}"] for item in candidates]
        scores = await asyncio.to_thread(self.model.compute_score, pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for result, score in zip(candidates, scores, strict=True):
            result.rerank_score = float(score)
        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        return candidates[:top_n]
