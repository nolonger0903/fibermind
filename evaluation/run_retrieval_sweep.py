from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from statistics import mean

from app.config import Settings
from app.infrastructure.knowledge import JsonKnowledgeRepository
from app.services.chunking import ParentChildChunker
from app.services.pipeline import DiagnosisPipeline
from app.services.retrieval import InMemoryHybridRetriever, LocalReranker

ROOT = Path(__file__).resolve().parent.parent


async def run_sweep() -> list[dict]:
    settings = Settings(_env_file=None)
    documents = JsonKnowledgeRepository(settings.knowledge_path).list_documents()
    cases = json.loads((ROOT / "evaluation" / "eval_dataset.json").read_text(encoding="utf-8"))
    rows = []
    for chunk_size in (120, 180, 240):
        chunks = [
            chunk
            for document in documents
            for chunk in ParentChildChunker(max_chars=chunk_size).split(document)
        ]
        for dense_weight, sparse_weight in ((0.7, 0.3), (0.55, 0.45), (0.4, 0.6)):
            retriever = InMemoryHybridRetriever(chunks, dense_weight, sparse_weight)
            for top_k in (4, 8, 12):
                started = time.perf_counter()
                recalls = []
                precisions = []
                reciprocal_ranks = []
                for case in cases:
                    candidates = await retriever.search(case["question"], top_k)
                    ranked = await LocalReranker().rerank(
                        case["question"], candidates, len(candidates)
                    )
                    selected = DiagnosisPipeline._select_parent_diverse(ranked, top_n=4)
                    retrieved = [item.chunk.parent_id for item in selected]
                    expected = set(case["expected_document_ids"])
                    hits = expected & set(retrieved)
                    recalls.append(len(hits) / len(expected))
                    precisions.append(len(hits) / len(retrieved) if retrieved else 0.0)
                    first_rank = next(
                        (
                            index
                            for index, parent in enumerate(retrieved, start=1)
                            if parent in expected
                        ),
                        None,
                    )
                    reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
                rows.append(
                    {
                        "chunk_size": chunk_size,
                        "top_k": top_k,
                        "dense_weight": dense_weight,
                        "sparse_weight": sparse_weight,
                        "context_recall": round(mean(recalls), 6),
                        "context_precision": round(mean(precisions), 6),
                        "mrr": round(mean(reciprocal_ranks), 6),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
    return rows


def write_reports(rows: list[dict]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            row["context_recall"],
            row["mrr"],
            row["context_precision"],
            -row["latency_ms"],
        ),
        reverse=True,
    )
    payload = {
        "method": "deterministic in-memory retrieval sweep (not RAGAS)",
        "cases": 6,
        "best": ranked[0],
        "runs": rows,
    }
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "retrieval_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# FiberMind Retrieval Sweep",
        "",
        "> Deterministic baseline sweep. These are retrieval metrics, not RAGAS scores.",
        "",
        "| Chunk | Top-K | Dense | Sparse | Recall | Precision | MRR | Latency ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {row['chunk_size']} | {row['top_k']} | {row['dense_weight']:.2f} | "
            f"{row['sparse_weight']:.2f} | {row['context_recall']:.4f} | "
            f"{row['context_precision']:.4f} | {row['mrr']:.4f} | {row['latency_ms']:.3f} |"
            for row in ranked
        ],
    ]
    (reports / "retrieval_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_reports(asyncio.run(run_sweep()))
