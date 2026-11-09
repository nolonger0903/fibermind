"""Prompt for LLM-based query rewriting, sub-query decomposition."""

from app.domain.models import Intent, IntentResult

INTENT_HINTS = {
    Intent.FAULT_DIAGNOSIS: "故障诊断类，改写时保留现象、告警、数值等定位线索",
    Intent.KNOWLEDGE_QA: "知识问答类，改写为明确的术语/参数/规范问题",
    Intent.OPERATION_GUIDE: "操作指导类，改写为明确的操作步骤请求",
    Intent.ALARM_EXPLANATION: "告警解释类，改写为对告警码含义/影响的提问",
    Intent.HISTORY_QUERY: "历史查询类，改写为对会话历史的明确引用",
    Intent.SMALL_TALK: "闲聊类，保持原文即可",
    Intent.UNSUPPORTED: "域外请求，保持原文即可",
    Intent.AMBIGUOUS: "意图模糊类，尽量保留原文信息，不要自行补全意图",
}

SYSTEM_PROMPT = (
    "你是光网络运维检索查询改写引擎。输入是一线运维人员的原始提问（可能口语化、"
    "包含错别字、简写、缩略、指代不清或多意图混排），输出改写结果供检索引擎使用。\n"
    "改写规则：\n"
    "1. 纠正明显的错别字与口语化表达，映射为标准运维术语"
    "（如“光猫/猫”→ONU、“闪断”→频繁掉线、“红灯”→LOS 告警）。\n"
    "2. 结合多轮对话历史消解模糊指代（“它/这个/刚才那个”等），把指代还原为具体对象。\n"
    "3. 不得增删事实：不编造告警码、设备、数值；原文没有的信息不能凭空补。\n"
    "4. 若提问包含多个可独立检索的子问题，拆分为 sub_queries（每条一个明确问题，"
    "最多 3 条）；单一问题则 sub_queries 为空数组。\n"
    "5. rewritten 是一条完整的、可直接用于检索的查询。\n"
    "6. hyde 是一段假设性答案文档（60 字以内），描述理想知识库文档会包含的内容，"
    "用于 HyDE 向量召回；无必要时可为 null。\n"
    "只输出一个 JSON 对象，不要输出任何解释文字，格式：\n"
    '{"rewritten": "...", "sub_queries": ["...", "..."], "hyde": "... 或 null"}'
)


def build_user_prompt(query: str, intent: IntentResult, history: list[dict[str, str]]) -> str:
    history_text = "\n".join(
        f"{item['role']}: {item['content']}" for item in history[-6:]
    ) or "（无历史，首轮提问）"
    hint = INTENT_HINTS.get(intent.intent, "未知意图，忠实改写")
    return (
        f"意图识别结果：{intent.intent.value}（{hint}），置信度 {intent.confidence:.2f}\n"
        f"抽取到的实体：{intent.entities or '无'}\n\n"
        f"多轮对话历史：\n{history_text}\n\n"
        f"原始提问：{query}\n\n"
        "请输出 JSON。"
    )
