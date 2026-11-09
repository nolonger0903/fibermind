# FiberMind 工程整改报告

日期：2026-09-17 · 范围：意图数据与训练链路、Intent Router、Query 重构、混合检索权重、Prompt 管理、存储职责、前端展示、测试与评测

> 状态更新：本报告最初记录的“BERT/RAGAS/BGE 未运行”状态已于同日后续对齐工作中
> 被取代；以下 Evaluation 与限制章节已更新为最终真实执行结果。

---

## 1. Summary

本次整改覆盖 10 项需求，核心变化：

- **数据生成重构**：从“4 个固定模板 + 同义词槽填充”升级为“语义簇种子 × 多样性变换（错别字/口语化/简写/不完整 Query/模糊指代）”，新增 `ambiguous` 意图与 Hard Negative 设计；新增 **LLM+Prompt 生成脚本**（无模板兜底，未配置 LLM 直接退出）。
- **防数据泄露机制化**：新增 `app/evaluation/dataset_split.py`，实现“聚类 → 按意图分层簇级切分 → 同标签跨侧高相似对迭代合并（union-find 泄露守护）→ 泄露自检”。实测抓到并修复了旧版随机切分的泄露问题。
- **训练评测补全**：新增 `app/evaluation/intent_metrics.py` 纯函数指标库（Accuracy / Macro P-R-F1 / Weighted F1 / Per-class P-R-F1 / 混淆矩阵），训练脚本输出 `reports/intent_metrics.json`、`reports/intent_report.md`、可选混淆矩阵 PNG；**未运行训练则不产生任何指标文件**。
- **Intent Router 重构**：BERT 优先 + Rule 降级（`ResilientIntentClassifier`，降级写日志与 trace、结果标注真实来源、熔断保护）、`ambiguous` 意图、低置信度 clarify、阈值全部配置化。
- **Query 重构 LLM 化**：`LLMQueryOptimizer` 按集中管理的 Prompt 调用 OpenAI-compatible 接口完成改写/子查询/HyDE/多轮指代消解，失败自动降级规则版并显式标注；前端新增“Query 重构”面板完整展示改写明细。
- **权重修复**：`WeightedRanker(dense, 1 - dense)` 改为统一校验 + 归一化的 `normalize_fusion_weights(dense, sparse)`，内存检索器与 Milvus 共用同一实现，Settings 启动期校验非法组合。
- **Prompt 集中管理**：新增 `app/prompts/`（system / diagnosis / rewrite / hyde / clarify / dataset），Prompt 内置证据约束（只依据证据、不足即追问、禁止编造告警/设备/阈值、排障顺序、区分事实/可能原因/建议操作、追问关键参数清单）。
- **存储职责澄清**：README 明确“Redis 实现答案缓存，MySQL 持久化 Session 与多轮历史”。
- **Parent-Child Chunking 保留并强化**：子块检索 + 父块生成不变；生成 Prompt 与信息不足模板显式引用父文档上下文约束。

## 2. Critical Bugs Fixed

### P0
1. **Milvus 混合检索忽略 sparse_weight**：`WeightedRanker(self.dense_weight, 1 - self.dense_weight)` → 统一校验归一化后 `WeightedRanker(weights.dense, weights.sparse)`（`app/infrastructure/milvus.py`）；内存检索器同步修复（原先同样用 `1 - dense_weight`）。
2. **BERT 无降级保护，故障面大**：模型加载失败会让进程无法启动（`build_container` 直接抛错）；推理抛错会让整个请求 500。且原 `BertIntentClassifier` 内部持有 fallback 实例但从不降级。→ 新增 `ResilientIntentClassifier`：优先 BERT、异常降级 Rule、降级事件写 `logger.warning` 与 trace、`classifier` 字段标注真实来源（如 `rule(fallback:bert-runtime-error)`）、连续失败 3 次熔断；容器加载失败降级并记日志。
3. **训练/测试集数据泄露**：旧版 `train_test_split(test_size=0.15, stratify_by_column)` 随机切分同模板生成的近重复样本 → 测试指标虚高。→ 簇级切分 + 泄露守护（实测旧逻辑在自检下抓出 1996 对跨侧高相似样本，新逻辑归零）。
4. **测试套件被本地 `.env` 污染**：`.env` 中的 DeepSeek 配置使 `test_config` 失败，且存在测试误用真实 API 的风险。→ `tests/conftest.py` 固定全部后端选择环境变量（环境变量优先级高于 dotenv），涉及默认值的测试用 `Settings(_env_file=None)` 隔离。

### P1
5. **Query Rewrite 是固定同义词替换**：→ `LLMQueryOptimizer`（Prompt 驱动、JSON 输出、markdown 围栏容错），失败降级规则版并在结果/trace/前端标注 `rules(llm-fallback)` + 失败原因。
6. **训练只输出 Accuracy**：→ 完整指标集 + JSON/Markdown/图片报告 + `macro_f1` 选型。
7. **无 ambiguous / 低置信 clarify 策略**：默认分支曾武断归为 `knowledge_qa(0.58)`；0.75 等阈值硬编码。→ 新增 `Intent.AMBIGUOUS`（多信号冲突、弱信号无实体、无信号三类判定）、`INTENT_CONFIDENCE_THRESHOLD` / `ROUTER_STANDARD_MIN_CONFIDENCE` 配置化、clarify 按缺失槽位生成针对性追问（设备型号/告警码/Tx·Rx 光功率/端口状态/时间范围）。
8. **README 引用不存在的 `.env.example` / `.env.deepseek.example`**：→ 补齐两个模板文件（不含真实 Key）。
9. **多轮指代误伤**：整改中发现“那应该先检查哪里？”这类追问会被 ambiguous 分支误路由到 clarify，破坏多轮演示链路 → Router 感知 `has_history`，多轮模糊指代交给 Rewrite 消解，首轮信息不足才 clarify。

### P2
10. **Prompt 散落在代码里** → `app/prompts/` 集中管理，业务代码只引用。
11. **README / docs 描述与实现脱节**（存储职责、项目结构、训练流程）→ 同步更新 README、docs/code-tour.md。

## 3. Architecture Changes

- **意图层**：`BertIntentClassifier / RuleIntentClassifier` 二选一 → `ResilientIntentClassifier` 包装（primary + fallback + 熔断 + 可观测降级），`IntentResult` 增加 `signals` 字段增强可解释性。意图类别从 7 类扩为 8 类（+`ambiguous`）。
- **路由层**：`decide(query, intent, cache_hit, standard_hit)` → 增加 `has_history` 与构造注入阈值；路由顺序 Cache → DIRECT（闲聊/域外/历史）→ ambiguous/低置信（首轮 clarify、多轮 RAG）→ Standard → 缺证据 clarify → RAG。
- **Query 优化层**：同步规则器 → async 双实现（LLM 优先/规则兜底），多轮上下文消解职责从 pipeline 下沉到优化器；`OptimizedQuery` 增加 `backend` / `fallback_reason` / `context_expanded`，WebSocket `query_optimized` 事件与前端面板展示全量明细。
- **检索层**：权重校验/归一化收敛到 `app/utils/retrieval_weights.py`，内存与 Milvus 两个后端共用，Settings `model_validator` 启动期拒绝非法组合。
- **评测层**：新增 `app/evaluation/`（指标纯函数 + 数据切分），训练脚本与单元测试共用同一实现。
- **数据层**：生成产物从单一 jsonl 变为全量 + train/val/test 四文件，行内含 `cluster_id` / `split` / `transforms`，切分可复现（固定 seed）。

## 4. Files Changed

主要修改（新增标 ✚）：

| 文件 | 用途 |
| --- | --- |
| ✚ `app/prompts/{__init__,system,diagnosis,rewrite,hyde,clarify,dataset}.py` | Prompt 集中管理 |
| ✚ `app/evaluation/__init__.py` / `intent_metrics.py` / `dataset_split.py` | 指标纯函数；簇级防泄露切分 |
| ✚ `app/utils/retrieval_weights.py` | 权重校验 + 归一化 |
| `app/config.py` | 权重 validator；路由阈值/降级开关/rewrite 开关配置化 |
| `app/domain/models.py` | `Intent.AMBIGUOUS`；`IntentResult.signals` |
| `app/services/intent.py` | Rule 分类器 ambiguous 判定；`ResilientIntentClassifier`（BERT 优先/降级/熔断/日志） |
| `app/services/router.py` | 阈值注入、ambiguous/低置信/多轮分支 |
| `app/services/query_optimizer.py` | 全部重写：`LLMQueryOptimizer` + `RuleQueryOptimizer`（async） |
| `app/services/pipeline.py` | await 优化器、历史前置、fallback 事件入 trace、clarify 槽位追问 |
| `app/services/retrieval.py` / `app/infrastructure/milvus.py` | 双权重检索与 WeightedRanker 修复 |
| `app/services/generation.py` | 引用集中 Prompt；信息不足回答带关键参数追问 |
| `app/container.py` | BERT 加载失败降级、LLM 优化器组装、阈值注入 |
| `app/static/{index.html,app.js,styles.css}` | “Query 重构”面板（改写引擎/原文/改写后/子查询/HyDE/降级原因） |
| `scripts/generate_intent_dataset.py` | 重写：语义簇种子 × 多样性变换 + 泄露守护切分 + 自检报告 |
| ✚ `scripts/generate_intent_dataset_llm.py` | LLM+Prompt 生成（无兜底、dry-run、重试、泄露守护） |
| `scripts/train_intent_bert.py` | 重写：簇切分数据、全指标、reports 三件套、smoke 参数 |
| `README.md` / `docs/code-tour.md` | 存储职责、结构、训练评测流程同步 |
| ✚ `.env.example` / `.env.deepseek.example` | 配置模板（不含真实 Key） |
| `tests/conftest.py` + 8 个测试文件（✚4：router/retrieval_weights/dataset_split/intent_metrics） | 56 个测试 |
| `data/generated/intent_queries*.jsonl` | 真实再生成（5000 条 + train/val/test） |

## 5. Tests（真实执行结果）

环境：WSL2 Ubuntu 22.04，Python 3.10.12，独立 venv，`pip install -e ".[dev]"`。

```text
pytest:          64 passed, 0 failed, 0 skipped   （整改前基线：14 passed, 1 failed）
ruff check .:    All checks passed!
compileall:      passed（app / scripts / evaluation / tests）
node --check app.js: OK（前端语法）
```

新增测试覆盖：Router 全分支与阈值配置化、ambiguous 有/无历史、BERT 优先/降级/熔断/容器降级、LLM 改写解析（含 markdown 围栏）与 HTTP/JSON 失败降级、权重非法组合拒绝与归一化、检索器双权重生效、指标手算对照、泄露守护收敛性、数据切分分层。

## 6. Evaluation（真实运行情况）

**已真实运行：**
- `scripts/generate_intent_dataset.py`：5000 条，96 簇，unique 58.1%，train/val/test = 3466/494/1040，同标签跨侧泄露对 = 0（阈值 0.65）。
- `scripts/train_intent_bert.py`：`bert-base-chinese`、3 epochs、FP16、有效 batch 16；独立测试集 1040 条，Accuracy 0.8356、Macro-F1 0.8225、Weighted-F1 0.8214。
- 远程 BGE API：`BAAI/bge-m3` 1024 维 Dense 与 `BAAI/bge-reranker-v2-m3` 精排均通过真实调用；新增正式适配器和 Mock 回归测试。
- `evaluation/run_evaluation.py`（离线确定性 smoke 评测，6 用例）：faithfulness_proxy 1.0，answer_relevancy_proxy 0.75，context_recall 0.8333，context_relevancy_proxy 0.8333。
- `evaluation/run_ragas.py`：使用 Qwen2.5-7B-Instruct Judge + BGE-M3 Embedding 完成 6 条真实评测，Faithfulness 0.7546、Answer Relevancy 0.7638、Context Precision 0.9583、Context Recall 0.8333，全部指标 6/6 有效。
- `evaluation/run_retrieval_sweep.py`：完成 Chunk / Top-K / Dense-Sparse 权重共 27 组确定性消融；小样本最优为 240 / 8 / 0.7:0.3。
- `scripts/generate_intent_dataset_llm.py`：无 LLM 配置时退出守卫、`--dry-run` Prompt 预览逻辑验证通过（未消耗真实 API）。

**仍未运行（环境限制）：**
- **LLM 批量数据生成**：未消耗真实 DeepSeek API；脚本经 MockTransport 级单测验证。
- **Milvus / Redis / MySQL 容器联调**：本机没有 Docker Engine；远程 BGE 链路已运行，本地 FlagEmbedding + Milvus 路径仍待 Docker 环境验证。

## 7. Remaining Limitations

1. 训练数据是合成数据，只有 58.1% 文本唯一；0.8225 Macro-F1 不能替代真实工单外部测试。
2. 泄露守护基于字符 bigram Jaccard（阈值 0.65），不能覆盖所有语义近重复。
3. RAGAS 只有 6 条样本，适合链路验证，不足以形成稳定的生产质量结论。
4. 熔断的 BERT 不会进程内自动恢复（重启恢复）。
5. `.env` 中的历史真实 Key 应轮换；本次新增的 BGE Key 未写入任何仓库文件。
6. `REWRITE_USE_LLM` 打开时每条 RAG 请求多一次 LLM 调用，仍缺少结果缓存和请求级总超时预算。

## 8. How to Run

```powershell
# 轻量模式（零依赖 demo）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Full 模式（Redis/MySQL/Milvus/BGE/LLM 按需在 .env 打开；模板见 .env.example）
docker compose up -d --build
python -m pip install -e ".[infra,ml]"

# 数据生成
.\.venv\Scripts\python.exe scripts\generate_intent_dataset.py            # 离线确定性（含防泄露切分）
python scripts\generate_intent_dataset_llm.py --per-intent 200           # LLM+Prompt（需 .env 配置 LLM）
python scripts\generate_intent_dataset_llm.py --dry-run                  # 只预览 Prompt

# 训练 BERT（真实训练后才会在 reports/ 生成指标）
python -m pip install -e ".[ml]"
python scripts\train_intent_bert.py --epochs 3            # 完整训练
python scripts\train_intent_bert.py --max-samples 200 --max-steps 10   # CPU 冒烟验证链路

# 建立 Milvus Index（RETRIEVAL_BACKEND=milvus 时服务启动自动建集合/灌数据）
# .env: RETRIEVAL_BACKEND=milvus, MILVUS_URI=http://localhost:19530

# 运行 Test
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q

# 运行 Evaluation
.\.venv\Scripts\python.exe evaluation\run_evaluation.py    # 离线确定性（已验证可运行）
python -m pip install -e ".[eval]"
python evaluation\run_ragas.py                             # 需 evaluator LLM
```

## 9. Resume-safe Claims

以下描述均能被当前代码直接证明（附证据位置）：

- 构建意图路由（8 类意图、BERT 优先 + Rule 降级熔断、置信度阈值配置化、clarify 追问策略）——`app/services/intent.py`、`app/services/router.py`、`app/services/clarify.py`
- 分层响应（Cache / Standard / RAG / Direct / Clarify 五路由）——`app/services/router.py`、`app/services/pipeline.py`
- Dense/Sparse Hybrid Retrieval（双权重校验归一化，内存与 Milvus 双后端）——`app/utils/retrieval_weights.py`、`app/services/retrieval.py`、`app/infrastructure/milvus.py`
- BGE-Reranker 接入——`app/infrastructure/bge.py`
- Parent-Child Retrieval（子块召回、父块生成）——`app/services/chunking.py`、`app/services/generation.py`
- 多轮 RAG（会话历史持久化、指代消解、上下文扩展）——`app/services/pipeline.py`、`app/infrastructure/mysql.py`、`app/services/query_optimizer.py`
- Redis 答案缓存 / MySQL Session 持久化（职责分离）——`app/infrastructure/cache.py`、`app/infrastructure/mysql.py`
- LLM Query Rewrite / HyDE / Sub-Query（Prompt 集中管理、失败降级标注、前端完整展示）——`app/prompts/`、`app/services/query_optimizer.py`、`app/static/`
- FastAPI/WebSocket Streaming（SSE token 级转发 + 全链路 trace 事件）——`app/api/routes.py`、`app/services/generation.py`
- 离线评测链路（确定性 smoke、参数消融、真实 RAGAS）——`evaluation/`、`reports/`
- 意图数据工程（LLM+Prompt 生成与离线变换双模式、簇级防泄露切分、泄露自检）——`scripts/generate_intent_dataset*.py`、`app/evaluation/dataset_split.py`
- 分类评测指标库（Accuracy/Macro/Weighted/Per-class/混淆矩阵，含手算对照单测）——`app/evaluation/intent_metrics.py`

**仍不能写的：**
- “线上延迟降低 xx%”或“相对基线提升 xx%”——当前没有线上对照实验。
- “真实工单 Macro-F1 0.8225”——0.8225 来自合成数据独立切分，不是真实工单。
- “LLM 生成 5 万/5 千条训练数据”——LLM 生成脚本未消耗真实 API 批量运行（当前仓库内 5000 条为离线确定性生成）。
