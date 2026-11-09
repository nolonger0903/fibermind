# FiberMind Project Description Alignment

Date: 2026-09-17

## Outcome

The repository now implements and has direct evidence for the main claims in the project
description. Claims that still depend on unavailable infrastructure or broader real-world data are
explicitly scoped below.

| Project claim | Implementation and evidence | Status |
| --- | --- | --- |
| 5000-query intent dataset and BERT fine-tuning | 5000 synthetic queries; cluster-level 3466/494/1040 split; `bert-base-chinese` trained for 3 epochs; test Macro-F1 0.8225 | Implemented and run; synthetic-data limitation retained |
| Intent Router and layered Cache/Standard/RAG response | `RuleIntentClassifier` / `BertIntentClassifier`, resilient fallback, configurable router, cache-first pipeline | Implemented; unit and API tested |
| BGE-M3 hybrid retrieval and BGE-Reranker | Local Milvus adapter uses BGE-M3 dense+sparse; remote adapter uses BGE-M3 dense + lexical sparse; remote BGE reranker | Remote path run; local Milvus path awaits Docker |
| Parent-child, Rewrite, HyDE and SubQuery | Child retrieval, parent-diverse Top-N, LLM/rule rewrite, HyDE and multi-query merge | Implemented and tested |
| RAGAS + LLM-as-a-Judge | Qwen2.5-7B-Instruct judge + BGE-M3 embeddings; four metrics, six valid samples each | Implemented and run |
| Chunk/Top-K/weight iteration | 27 deterministic sweep combinations saved in `retrieval_sweep.*` | Implemented and run; six-case sample is small |
| FastAPI/WebSocket multi-turn streaming | REST, WebSocket events, token streaming, session history and trace | Implemented and API tested |
| Redis/MySQL/Milvus/Docker production stack | Adapters and Compose definitions exist; Compose configuration validates | Not integration-tested because this machine has no Docker Engine |

## Verified Results

- Tests: 64 passed, 0 failed, 0 skipped.
- Intent test set: Accuracy 0.8356, Macro-F1 0.8225, Weighted-F1 0.8214.
- RAGAS: Faithfulness 0.7546, Answer Relevancy 0.7638, Context Precision 0.9583,
  Context Recall 0.8333.
- Retrieval sweep best deterministic baseline: chunk 240, Top-K 8, dense/sparse 0.7/0.3,
  Context Recall 1.0 and MRR 1.0 on six cases.

## Claims That Must Stay Qualified

- The intent metrics are from synthetic data, not a held-out real ticket dataset.
- The RAGAS and retrieval sets contain six cases and are regression smoke tests, not production
  quality estimates.
- The remote BGE API does not return BGE-M3 lexical weights; native BGE-M3 dense+sparse belongs to
  the local FlagEmbedding + Milvus path.
- Redis, MySQL and Milvus container integration remains unverified on this machine.
