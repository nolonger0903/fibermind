"""HyDE (hypothetical document) guidance shared by rewrite and retrieval expansion."""

# 该指令作为 rewrite prompt 中 hyde 字段的补充说明；当查询为故障诊断类时启用。
HYDE_FAULT_INSTRUCTION = (
    "针对故障诊断类提问，hyde 按以下骨架编写：设备/端口 + 告警现象 + "
    "光功率等关键数值 + 常见原因 + 排查顺序。"
    "只描述理想文档应包含的内容，不要给出具体结论。"
)
