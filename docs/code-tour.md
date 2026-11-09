# FiberMind 逐文件代码导读

本文只解释项目源文件。`.venv`、`__pycache__` 和 `*.egg-info` 是 Python 工具生成的目录，
不属于业务代码。

## 推荐阅读顺序

第一次读代码建议按以下顺序：

```text
app/main.py
  -> app/container.py
  -> app/services/pipeline.py
  -> intent.py / router.py / query_optimizer.py
  -> retrieval.py / generation.py
  -> infrastructure/*
  -> api/routes.py
```

`main.py` 回答“服务如何启动”，`container.py` 回答“当前启用了哪些实现”，`pipeline.py`
回答“一条 Query 如何流转”。其余文件是各步骤的具体算法和外部适配器。

## 根目录

### `README.md`

项目总入口。包含快速启动、演示问题、架构图、API、DeepSeek 配置、BERT 训练、真实后端切换、
测试和评测命令。

### `pyproject.toml`

Python 工程与依赖声明：

- 默认依赖负责 FastAPI、配置和 HTTP 调用。
- `infra` extra 安装 Redis、SQLAlchemy、PyMySQL 和 PyMilvus。
- `ml` extra 安装 Torch、Transformers、FlagEmbedding 等模型依赖。
- `eval` extra 安装 RAGAS；`dev` extra 安装 Pytest 和 Ruff。
- 同时保存 Pytest、Ruff 的项目级配置。

### `.env.example`

所有环境变量的完整模板。默认值是完全离线的 demo 模式，不含密钥。

### `.env.deepseek.example`

DeepSeek 专用最小模板。复制成 `.env` 并在本机填写 `LLM_API_KEY` 即可启用。

### `.gitignore`

排除虚拟环境、缓存、模型权重、真实 `.env` 和生成数据，主要作用是防止 API Key 与大型文件
进入版本控制。

### `Dockerfile`

API 服务镜像：复制应用与知识数据、安装运行依赖，然后用 Uvicorn 监听 `8000`。

### `docker-compose.yml`

编排 API、Redis、MySQL、Milvus、etcd 和 MinIO。Milvus 使用 etcd 保存元数据、MinIO 保存对象；
默认 API 容器先使用 Redis + MySQL + 本地检索，避免普通电脑启动时被大模型下载阻塞。

### `Makefile`

常用命令别名，包括安装、开发启动、测试、Lint、数据生成、评测和 Docker 启停。在 Windows
没有 `make` 时直接执行对应 Python/Docker 命令即可。

## `app/`：在线服务

### `app/main.py`

FastAPI 程序入口。`create_app()` 注册 lifespan、REST Router、WebSocket、静态文件和首页。
lifespan 启动时调用 `build_container()`，因此模型和数据库不会在模块 import 阶段被隐式创建。

### `app/config.py`

使用 Pydantic Settings 从 `.env` 和系统环境变量加载配置，并校验 Top-K、权重、超时等范围。
`resolved_llm_base_url` 和 `resolved_llm_model` 为 DeepSeek 提供默认地址和 `deepseek-chat` 模型，
但显式配置始终优先。

### `app/container.py`

依赖装配中心，也就是 Composition Root。它根据配置选择：

- Rule 或 BERT 意图分类器；
- Memory 或 Redis 缓存；
- Memory 或 MySQL 会话；
- 本地或 Milvus 检索；
- 本地或 BGE 精排；
- 本地模板、DeepSeek 或 OpenAI-compatible 生成器。

最后把这些对象注入 `DiagnosisPipeline`。切换后端不需要在 Pipeline 里写环境判断。

### `app/domain/models.py`

与框架无关的领域对象：

- `Intent`：七类意图枚举；
- `Route`：缓存、标准答案、RAG、直接回答和澄清；
- `KnowledgeDocument` / `KnowledgeChunk`：父文档与检索子块；
- `RetrievalResult`：保存 Dense、Sparse、Fusion、Rerank 四类分数；
- `Message`：会话消息；
- `TraceStep`：每个流水线阶段的状态、耗时和调试摘要。

使用 dataclass 是为了让业务层不依赖 HTTP/Pydantic。

### `app/schemas/api.py`

API 边界的 Pydantic Schema。负责请求长度、Top-K 范围等输入校验，也定义响应中的意图、诊断、
引用、Trace、消息和健康检查结构。领域 dataclass 与 API Schema 分开，避免传输格式侵入算法层。

## `app/api/`：接口层

### `app/api/routes.py`

集中声明 REST 和 WebSocket 接口：

- `/chat`、`/diagnose` 调用完整 Pipeline；
- `/intent/predict` 单独观察意图识别；
- `/knowledge/search` 检查召回和精排分数；
- `/sessions/...` 获取历史；
- `/health`、`/metrics` 查看运行模式与统计；
- `/ws/chat/{session_id}` 将流水线事件和答案 Token 实时推送给前端。

异常会转换为 WebSocket `error` 事件，而不会让连接无提示地断开。

## `app/services/`：核心业务逻辑

### `app/services/pipeline.py`

项目核心编排器。一次请求按以下顺序执行：

1. 保存用户消息并识别意图、实体；
2. 查询缓存和标准答案，交给 Router 选路径；
3. RAG 路径加载历史并优化 Query；
4. 对主 Query 和 Sub-Query 分别召回，再按 chunk id 合并；
5. 用原始 Query 精排候选；
6. 将父文档、近期历史交给生成器；
7. 流式发送 Token，保存助手消息，返回引用和完整 Trace。

这里还负责路由次数、缓存命中相关统计。复杂故障答案默认不缓存，避免跨现场复用结论。

### `app/services/intent.py`

包含三个实现：

- `RuleIntentClassifier` 是离线 fallback，可解释规则识别意图并抽取设备、告警码、光功率
  和症状；返回 `signals` 字段供 trace 与测试追溯；
- `BertIntentClassifier` 加载训练后的 Hugging Face Sequence Classification 模型，标签来自
  模型的 `id2label`，实体仍由领域规则抽取；
- `ResilientIntentClassifier` 是 BERT 优先 + Rule 降级的包装器：BERT 推理异常自动
  fallback，并把降级事件写入日志与 `fallback_events`（由 pipeline 透传到 trace，
  `IntentResult.classifier` 标注真实来源，例如 `rule(fallback:bert-runtime-error)`），
  连续失败达阈值后熔断，避免每个请求都重复付出失败推理的开销。

### `app/services/router.py`

把分类结果转成业务路径。优先级是 Cache → 闲聊/越界/历史直答 → ambiguous（首轮
无历史）或低置信度 → 标准答案 → 信息不足追问 → RAG。**多轮追问（`has_history=True`）
中的模糊指代优先交给 Query Rewrite 用会话历史消解**，避免破坏多轮演示链路。
所有阈值通过构造参数注入：`confidence_threshold`、`standard_min_confidence`，由
Settings 提供，配置化无硬编码。

### `app/services/query_optimizer.py`

两个实现，统一 async `optimize(query, intent, history)` 接口，输出 `OptimizedQuery`
（含 `backend` 与 `fallback_reason`，前端“Query 重构”面板展示全量明细）：

- `LLMQueryOptimizer`：按 `app/prompts/rewrite.py` 调 OpenAI-compatible 接口，做多轮
  指代消解、错别字纠正、术语归一、子查询拆分和 HyDE 假设文档；失败自动降级规则版并
  标注 `backend=rules(llm-fallback)`；
- `RuleQueryOptimizer`：确定性兜底，同义词映射 + 条件拆分 + 上下文拼接。

HyDE 只参与召回，最终回答仍必须来自知识库文档。

### `app/services/chunking.py`

实现父子块切分。子块控制在约 180 字用于检索，每个子块通过 `parent_id` 指向完整文档，并在
metadata 中保存父文内容。这样既提高局部命中率，又不会把残缺步骤交给生成模型。

### `app/services/retrieval.py`

离线 baseline：

- `InMemoryHybridRetriever` 用字符/二元词特征计算类 Dense 相似度、词项覆盖率作为 Sparse
  分数，再加告警码精确命中奖励；
- `LocalReranker` 综合初排、标题、元数据、告警码和名次先验做第二阶段排序。

它用于无模型环境的稳定演示，不冒充真实 BGE；真实实现位于 `infrastructure/`。

### `app/services/generation.py`

包含统一的 `GeneratedAnswer` 和两种生成器：

- `GroundedTemplateGenerator` 从已召回父文档抽取步骤，在完全离线时生成有引用回答；
- `OpenAICompatibleGenerator` 组装系统 Prompt、近期历史和编号知识上下文，调用 DeepSeek 或
  OpenAI-compatible `/chat/completions`。

WebSocket 场景设置 `stream=true` 并解析 SSE；每个增量 Token 立即回调 Pipeline。HTTP 状态、
超时、非法 JSON 和空流会产生带 provider 信息的明确错误。

## `app/infrastructure/`：外部系统适配器

### `app/infrastructure/knowledge.py`

`JsonKnowledgeRepository` 加载领域知识；`StandardAnswerRepository` 对归一化 Query 做确定性
模式匹配。标准答案不会经过 LLM，适合告警含义等高频稳定问题。

### `app/infrastructure/cache.py`

`MemoryTTLCache` 用于 demo，支持 TTL、命中/未命中计数；`RedisCache` 保持相同异步接口，值以
JSON 存储。Pipeline 不关心实际缓存介质。

### `app/infrastructure/sessions.py`

内存版 Session Repository。按 session id 保存消息，支持追加、读取最近 N 条和统计会话数；
进程重启后数据消失，适合测试和 demo。

### `app/infrastructure/mysql.py`

MySQL 持久化版本。内部定义 SQLAlchemy ORM 表，通过 `asyncio.to_thread` 隔离同步 PyMySQL
调用，防止阻塞 FastAPI 事件循环。实现与内存版相同的方法契约。

### `app/infrastructure/bge.py`

真实 `BAAI/bge-reranker-v2-m3` 适配器。把原始 Query 与候选文本组成 Pair，通过 Cross
Encoder 计算归一化分数，再返回 Top-N。

### `app/infrastructure/milvus.py`

真实 BGE-M3 + Milvus 混合检索：

1. 首次启动创建 Dense、Sparse 和标量字段；
2. 用 BGE-M3 同时编码 Dense Vector 与 Lexical Weights；
3. 建立 HNSW 与 Sparse Inverted Index；
4. 查询时发起两个 `AnnSearchRequest`，使用 `WeightedRanker` 融合；
5. 将 Milvus 结果还原为统一 `RetrievalResult`。

### `app/*/__init__.py`

标记 Python Package，并用简短 docstring 说明包职责。当前不包含业务行为。

## `app/static/`：演示前端

### `app/static/index.html`

三栏工作台结构：会话/环境、诊断对话、可观察链路。提供三个预设 Query 和无障碍标签。

### `app/static/styles.css`

控制配色、布局、Trace 时间线、引用卡片和响应式断点。宽屏为三栏；小于 850px 时隐藏侧栏，
诊断区与 Trace 上下排列。

### `app/static/app.js`

浏览器端状态与交互：建立 WebSocket、提交 Query、消费阶段事件和 Token、更新 Trace/引用/耗时、
切换面板并转义动态文本。前端只展示后端事实，不在浏览器里重新计算检索结果。

## `data/`：演示数据

### `data/knowledge.json`

光网络知识父文档，包含标题、设备类型、故障类型、告警码、来源、版本、标签和完整处理流程。

### `data/standard_answers.json`

高频确定性问答及其匹配表达，例如 LOS/LOF 含义、常见光功率范围、端面清洁规范。

### `data/generated/intent_queries.jsonl`

脚本生成的 5000 条训练样本。每行包含 id、文本、意图和实体槽位；该目录被 Git 忽略，可随时
用固定 seed 重建。

## `scripts/`：离线训练工具

### `scripts/generate_intent_dataset.py`

按固定类别数量生成 5000 条 Query，使用固定随机种子保证复现，并打印类别分布。合成数据只用
于打通训练链路，正式训练需要混入脱敏真实工单和人工难例。

### `scripts/train_intent_bert.py`

读取 JSONL、做分层训练/验证拆分，加载 `bert-base-chinese`，使用 Hugging Face Trainer 微调，
按验证集 Accuracy 选择最佳 checkpoint，并保存模型、Tokenizer 与标签映射。

## `evaluation/` 与 `reports/`

### `evaluation/eval_dataset.json`

小型回归集。每个问题记录应出现的答案关键词和期望召回文档 id。

### `evaluation/run_evaluation.py`

无需 LLM Judge 的确定性 smoke evaluation，计算 Faithfulness proxy、Answer Relevancy proxy、
Context Recall、首条 Context Relevancy proxy 和延迟，输出到 `reports/`。

### `evaluation/run_ragas.py`

收集系统真实回答和上下文，交给 RAGAS 计算 Faithfulness、Answer Relevancy、Context
Precision/Relevancy、Context Recall。需要另行配置评价模型和 Embedding。

### `reports/evaluation.json|csv|md`

同一次离线评测的机器可读、表格和人工阅读版本。它们是运行结果，不参与在线服务。

## `tests/`：自动化测试

### `tests/test_intent.py`

验证诊断意图、中文相邻告警码、实体抽取，以及“告警解释”优先于普通诊断规则。

### `tests/test_query_optimizer.py`

验证术语改写不丢失告警码、多条件 Query 会拆分、诊断 Query 会产生 HyDE。

### `tests/test_retrieval.py`

验证 LOS Query 能召回正确知识、分数顺序合理，以及设备类型过滤生效。

### `tests/test_pipeline.py`

验证标准答案到缓存、完整 RAG 六阶段 Trace、关键来源进入 Top-N、多轮追问继承前文。

### `tests/test_api.py`

使用 FastAPI TestClient 验证健康检查、Chat REST，以及 WebSocket 的阶段事件、Token 增量和最终
完成事件。

### `tests/test_config.py`

验证 DeepSeek 默认地址/模型解析、显式配置优先，以及外部 LLM 模式缺少 Key 时快速失败。

## 一句话理解依赖方向

```text
API/UI -> Pipeline -> 领域接口形状 -> Infrastructure
```

业务编排不反向依赖 FastAPI、Redis、MySQL、Milvus 或特定 LLM SDK。这是代码能在 demo 和
完整环境之间切换，同时仍然容易测试和讲解的关键。

