# FiberMind 面试讲解稿

## 5 分钟讲解顺序

### 1. 先讲问题，不先报技术栈

光网络现场 Query 往往口语化、信息不完整，而且同一个告警在不同影响范围下排查路径不同。
如果所有请求都直接向量检索再调用 LLM，会同时遇到简单问题延迟高、领域术语召回不稳和无证据
生成的问题。因此系统先判断“这个请求应该走哪条路径”，再优化复杂 Query。

### 2. 打开主编排代码

从 `app/services/pipeline.py` 讲一遍：

1. 意图识别同时输出置信度和实体，不只输出标签。
2. Router 依次判断缓存、标准答案、直接回答、澄清和 RAG。
3. 复杂问题做术语归一化、上下文补全、HyDE 和 Sub-Query。
4. 每个 Sub-Query 独立召回，再按 chunk id 合并最高分。
5. 对 Top-K 使用 Reranker，最终只把 Top-N 父文档交给生成器。
6. 返回答案时保留引用、各阶段分数、路由原因和耗时。

### 3. 用两次 LOS 查询证明分层路由

第一次问 `LOS是什么意思？`，界面显示 `standard_knowledge`；再问一次显示 `cache`。说明缓存键
经过 Query 标准化，并强调复杂故障答案不会默认缓存，因为现场条件可能不同。

### 4. 演示复杂 Query

输入包含 ONU、-29 dBm、闪断和 LOS 的问题。右侧依次指出：

- Intent 为 `fault_diagnosis`，同时抽取设备、告警、指标和症状。
- Query Rewrite 把“闪断”等口语映射成领域表达，Sub-Query 拆解多条件。
- Hybrid Retrieval 保留 Dense、Sparse 和 Fusion 分数。
- Reranker 用原始 Query 精排，避免改写 Query 改变用户真实意图。
- 生成结果附知识来源，证据不足时会追问。

### 5. 最后讲工程边界

默认模式是可离线运行的 baseline。真实环境将相同接口切换为 BERT、BGE-M3、Milvus、
BGE-Reranker、Redis、MySQL 和在线 LLM。这样既能在普通电脑稳定演示，也保留真实部署代码，
而且不会把规则算法冒充已训练模型。

## 常见追问

**为什么使用混合检索？**

Dense 对同义表达更稳，但告警码、设备型号和精确数值可能被语义表示稀释；Sparse 对精确术语
强，但口语化表达召回差。BGE-M3 一次推理输出两类表示，融合后兼顾二者。

**为什么还需要 Reranker？**

向量检索强调召回速度，Query 与文档的交互有限。Cross Encoder 联合编码 Query-Document，
排序更准但成本高，所以只处理初排 Top-K。`retrieval_top_k` 和 `rerank_top_n` 是主要延迟旋钮。

**HyDE 会不会引入幻觉？**

HyDE 只用于扩展检索，不作为最终事实来源。最终上下文仍来自知识库，告警码和数值保留在原始
Query 中，Reranker 也使用原始 Query，因此假设文本不会直接进入答案。

**多轮对话怎么避免历史污染？**

只在短追问或出现指代词时拼接上一条用户 Query；生成最多使用最近 6 条历史。生产环境还应
增加会话摘要、实体状态和话题切换检测。

**怎么评价检索和生成？**

CI 使用确定性 proxy 快速发现召回回归；正式实验用 RAGAS 的 Faithfulness、Answer
Relevancy、Context Precision/Relevancy、Context Recall，并按 Chunk、Top-K、融合权重做消融。
失败样例应进入固定回归集，而不是只看平均分。

**当前项目有什么诚实的限制？**

- 已完成 `bert-base-chinese` 三轮微调（测试集 Macro-F1 0.8225），但训练数据是合成数据，
  仍需真实脱敏工单外部验证；模型权重位于本地 `models/intent-bert`，不进入版本控制。
- demo 检索器是可解释 baseline；`bge_api` 使用远程 BGE-M3 Dense + 词法 Sparse，只有
  Milvus + 本地 FlagEmbedding 路径使用 BGE-M3 原生 Dense/Sparse。
- 示例知识库规模小，证明的是链路与工程设计，不能代表生产知识覆盖度。
- 已完成 6 条真实 RAGAS 评测，但评测集仍小；当前 Context Recall 为 0.8333，需要继续补充
  难例并提升多条件故障的上下文覆盖。
- 当前机器没有 Docker Engine，因此 Redis/MySQL/Milvus 的容器级联调仍需在有 Docker 的
  环境执行；远程 BGE-M3/BGE-Reranker API 链路已经真实跑通。

这些限制可以主动说明。面试重点是展示清楚的接口边界、实验方法和真实可运行的链路，而不是
声称没有实际测过的线上规模或准确率。
