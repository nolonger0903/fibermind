"""Prompts for grounded diagnosis / QA generation."""

from app.prompts.system import SYSTEM_ROLE

SYSTEM_PROMPT = SYSTEM_ROLE + (
    "\n\n本次任务的输出要求：\n"
    "- 先用一两句话给出结论性判断，并注明依据来源编号，例如[1]。\n"
    "- 故障诊断按【排障顺序】输出编号步骤，顺序体现从公共故障点到单点、"
    "从物理层到设备层的排查优先级。\n"
    "- 可能原因与确定事实分开陈述，不确定的措辞用“可能”“建议排查”。\n"
    "- 引用证据时使用[序号]格式，只引用给定上下文中存在的条目。\n"
    "- 若上下文不足以支撑结论，明确说明信息不足并按系统约束追问关键参数。"
)


def build_user_prompt(query: str, context_text: str) -> str:
    return (
        f"知识上下文（唯一事实来源，禁止使用上下文之外的知识）：\n{context_text}\n\n"
        f"用户问题：{query}\n"
        "请依据以上上下文回答；不足处明确说明并追问关键参数。"
    )
