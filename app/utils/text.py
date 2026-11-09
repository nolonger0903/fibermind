from __future__ import annotations

import math
import re
from collections import Counter

DOMAIN_SYNONYMS = {
    "频繁闪断": "频繁掉线",
    "闪断": "频繁掉线",
    "断网": "业务中断",
    "红灯": "LOS告警",
    "光不好": "接收光功率异常",
    "弱光": "接收光功率低",
    "强光": "接收光功率高",
    "猫": "ONU",
    "光猫": "ONU",
}


def normalize_text(text: str) -> str:
    value = text.strip()
    for source, target in DOMAIN_SYNONYMS.items():
        value = value.replace(source, target)
    value = re.sub(r"\s+", "", value)
    return value.lower()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    latin = re.findall(r"[a-z]+(?:_[a-z]+)*|\d+(?:\.\d+)?(?:dbm|db)?", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chars: list[str] = []
    for run in chinese_runs:
        chars.extend(run)
        chars.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin + chars


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def lexical_overlap(query_tokens: list[str], document_tokens: list[str]) -> float:
    query_set = set(query_tokens)
    document_set = set(document_tokens)
    if not query_set:
        return 0.0
    return len(query_set & document_set) / len(query_set)
