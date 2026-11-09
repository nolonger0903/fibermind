from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings
from app.domain.models import KnowledgeChunk, RetrievalResult
from app.utils.retrieval_weights import FusionWeights, normalize_fusion_weights


class MilvusHybridRetriever:
    """BGE-M3 dense/sparse retrieval backed by Milvus hybrid search."""

    def __init__(self, settings: Settings, chunks: list[KnowledgeChunk]) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
        except ImportError as exc:
            raise RuntimeError(
                "Install FiberMind with the 'infra' and 'ml' extras for Milvus"
            ) from exc

        self.client = MilvusClient(uri=settings.milvus_uri)
        self.collection = settings.milvus_collection
        # dense_weight 与 sparse_weight 都来自配置：统一校验后归一化，
        # 禁止 WeightedRanker(dense, 1 - dense) 这种忽略 sparse_weight 的写法。
        self.weights: FusionWeights = normalize_fusion_weights(
            settings.dense_weight, settings.sparse_weight
        )
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.encoder = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
        self.AnnSearchRequest = AnnSearchRequest
        self.WeightedRanker = WeightedRanker
        self.DataType = DataType
        self._ensure_collection(chunks)

    def _ensure_collection(self, chunks: list[KnowledgeChunk]) -> None:
        if not self.client.has_collection(self.collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", self.DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("parent_id", self.DataType.VARCHAR, max_length=128)
            schema.add_field("title", self.DataType.VARCHAR, max_length=512)
            schema.add_field("content", self.DataType.VARCHAR, max_length=4096)
            schema.add_field("metadata", self.DataType.JSON)
            schema.add_field("dense_vector", self.DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("sparse_vector", self.DataType.SPARSE_FLOAT_VECTOR)
            index = self.client.prepare_index_params()
            index.add_index(
                "dense_vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 32, "efConstruction": 200},
            )
            index.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            self.client.create_collection(self.collection, schema=schema, index_params=index)

        stats = self.client.get_collection_stats(self.collection)
        if int(stats.get("row_count", 0)) == 0:
            texts = [f"{chunk.title}\n{chunk.content}" for chunk in chunks]
            embeddings = self.encoder.encode(texts, return_dense=True, return_sparse=True)
            rows = []
            for index, chunk in enumerate(chunks):
                rows.append(
                    {
                        "id": chunk.id,
                        "parent_id": chunk.parent_id,
                        "title": chunk.title,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "dense_vector": embeddings["dense_vecs"][index],
                        "sparse_vector": self._sparse_row(embeddings["lexical_weights"][index]),
                    }
                )
            self.client.insert(self.collection, rows)

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        return await asyncio.to_thread(self._search_sync, query, top_k, filters)

    def _search_sync(
        self, query: str, top_k: int, filters: dict[str, str] | None
    ) -> list[RetrievalResult]:
        vectors = self.encoder.encode([query], return_dense=True, return_sparse=True)
        dense_request = self.AnnSearchRequest(
            [vectors["dense_vecs"][0]],
            "dense_vector",
            {"metric_type": "COSINE", "params": {"ef": 80}},
            limit=top_k,
            expr=self._filter_expr(filters),
        )
        sparse_request = self.AnnSearchRequest(
            [self._sparse_row(vectors["lexical_weights"][0])],
            "sparse_vector",
            {"metric_type": "IP"},
            limit=top_k,
            expr=self._filter_expr(filters),
        )
        raw = self.client.hybrid_search(
            self.collection,
            [dense_request, sparse_request],
            ranker=self.WeightedRanker(self.weights.dense, self.weights.sparse),
            limit=top_k,
            output_fields=["id", "parent_id", "title", "content", "metadata"],
        )[0]
        results: list[RetrievalResult] = []
        for hit in raw:
            entity = hit["entity"]
            chunk = self.chunks.get(entity["id"]) or KnowledgeChunk(
                id=entity["id"],
                parent_id=entity["parent_id"],
                title=entity["title"],
                content=entity["content"],
                metadata=entity["metadata"],
            )
            results.append(RetrievalResult(chunk=chunk, fusion_score=float(hit["distance"])))
        return results

    @staticmethod
    def _sparse_row(weights: Any) -> dict[int, float]:
        if isinstance(weights, dict):
            return {int(key): float(value) for key, value in weights.items()}
        return {
            int(key): float(value) for key, value in zip(weights.col, weights.data, strict=True)
        }

    @staticmethod
    def _filter_expr(filters: dict[str, str] | None) -> str:
        if not filters:
            return ""
        expressions = []
        for key, value in filters.items():
            safe_value = value.replace('"', '\\"')
            expressions.append(f'metadata["{key}"] == "{safe_value}"')
        return " and ".join(expressions)
