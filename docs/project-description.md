# FiberMind | 基于 RAG 的智能故障诊断系统

Python · FastAPI · BERT · BGE-M3 · BGE-Reranker · Milvus · Redis · MySQL · Docker · RAGAS

## 项目概述

面向光网络运维故障诊断，针对知识分散、Query 口语化和信息不完整导致召回不稳的问题，
构建意图路由、分层响应、混合检索、二阶段精排与多轮对话 RAG 系统，并通过离线回归和
模型评测同时关注效果与响应延迟。

- 构建 5000 条合成用户 Query 意图分类数据集，采用语义簇级 train/val/test 切分和近重复
  泄露检查；完成 `bert-base-chinese` 三轮微调，在 1040 条独立合成测试集上取得 Accuracy
  0.8356、Macro-F1 0.8225。设计 Query Intent Router，将知识查询、故障诊断、闲聊、
  域外请求和信息不足请求动态路由至 Cache、标准知识、Direct、Clarify 或 RAG。
- 设计 Redis Cache → 高频标准知识 → RAG 分层响应链路；高频确定性问题返回标准答案并
  写入缓存，复杂和长尾 Query 进入混合检索，故障诊断答案默认不缓存，避免跨现场复用结论。
- 实现 BGE-M3 + Milvus 原生 Dense/Sparse Hybrid Retrieval 与 BGE-Reranker 本地适配器；
  同时实现并真实验证远程 BGE-M3 Dense + 词法 Sparse、远程 BGE-Reranker 后端。结合父子块
  索引、父文档多样性选择、Query Rewrite、HyDE 和 SubQuery 提升复杂 Query 的证据覆盖。
- 构建确定性 Proxy、参数消融与 RAGAS + LLM-as-a-Judge 三层离线评测。真实 6 条 RAGAS
  回归结果为 Faithfulness 0.7546、Answer Relevancy 0.7638、Context Precision 0.9583、
  Context Recall 0.8333；完成 27 组 Chunk、Top-K 和 Dense/Sparse 权重组合实验并保存记录。
- 基于 FastAPI / WebSocket 实现多轮会话与流式响应，使用 Redis 管理答案缓存、MySQL 管理
  Session 与历史上下文，形成“意图识别 → Query 优化 → 检索 → Rerank → 生成”完整链路；
  API、WebSocket 与核心业务由 64 个自动化测试覆盖。

## 证据边界

- BERT 指标来自合成数据，不能表述为真实工单测试结果。
- 远程 BGE embedding API 不返回 lexical weights，因此远程路径的 Sparse 是词法分量；
  BGE-M3 原生 Dense/Sparse 对应本地 FlagEmbedding + Milvus 路径。
- RAGAS 和参数消融目前只有 6 条固定回归用例，适合链路与回归验证，不代表线上总体质量。
- 当前机器没有 Docker Engine，Redis/MySQL/Milvus 适配器和 Compose 配置已实现并通过静态
  校验，但三者的容器级集成测试仍需在有 Docker Engine 的环境执行。
