"""Templates for the clarify strategy when intent is ambiguous or low-confidence."""

CLARIFY_OPENING = "当前描述还不足以准确定位问题，请补充以下关键信息："

SLOT_QUESTIONS: dict[str, str] = {
    "device_type": "设备型号或类型（如 ONU / OLT / ONT / 光模块）？",
    "alarm_codes": "具体告警码（如 LOS / LOF / CRC_ERROR / AUTH_FAIL）？",
    "power_values": "当前 Tx / Rx 光功率各是多少 dBm？",
    "symptoms": "具体故障现象（掉线 / 闪断 / 误码 / 离线 / 网速慢）？",
}

DEFAULT_QUESTIONS = [
    SLOT_QUESTIONS["device_type"],
    SLOT_QUESTIONS["alarm_codes"],
    SLOT_QUESTIONS["power_values"],
    "端口状态与指示灯现象？",
    "故障发生的时间范围与影响范围？",
]


def build_clarify_reply(missing_slots: list[str]) -> str:
    questions = [SLOT_QUESTIONS[slot] for slot in missing_slots if slot in SLOT_QUESTIONS]
    for question in DEFAULT_QUESTIONS:
        if len(questions) >= 3:
            break
        if question not in questions:
            questions.append(question)
    lines = "\n".join(f"{index}. {question}" for index, question in enumerate(questions, 1))
    return f"{CLARIFY_OPENING}\n{lines}"
