"""LLM-driven intent dataset generation (llm + prompt mode).

与离线版 generate_intent_dataset.py 的区别：样本文本由 OpenAI-compatible LLM
（DeepSeek 等）按 app/prompts/dataset.py 中的指令逐批生成，覆盖口语化、错别字、
简写、不完整 Query、多意图边界、Hard Negative、同词不同意图、模糊指代等真实
运维表达；本脚本不提供任何模板兜底——没有配置 LLM 时直接退出并给出指引，
禁止静默用模板数据冒充 LLM 生成结果。

生成完成后同样执行“聚类 → 分层簇切分 → 泄露自检”，防止近重复样本跨集。

用法：
    python scripts/generate_intent_dataset_llm.py --per-intent 200
    python scripts/generate_intent_dataset_llm.py --dry-run   # 只校验配置与提示词
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from app.config import get_settings
from app.evaluation.dataset_split import split_with_leakage_guard, verify_no_leakage
from app.prompts import dataset as dataset_prompts
from app.services.intent import RuleIntentClassifier

ROOT = Path(__file__).resolve().parent.parent

INTENTS = [
    "fault_diagnosis",
    "knowledge_qa",
    "operation_guide",
    "alarm_explanation",
    "history_query",
    "small_talk",
    "unsupported",
    "ambiguous",
]


def _extract_json_array(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response contains no JSON array: {content[:160]}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("LLM response JSON is not an array")
    rows: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text_value = str(item.get("text") or "").strip()
        label = str(item.get("intent") or "").strip()
        if text_value and label in INTENTS:
            rows.append({"text": text_value, "intent": label})
    if not rows:
        raise ValueError("LLM response contained no valid rows")
    return rows


class LLMClient:
    """Minimal synchronous OpenAI-compatible chat client for dataset generation."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.httpx = httpx

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.9) -> str:
        with self.httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])


def generate_batch(
    client: LLMClient, intent: str, count: int, avoid: list[str]
) -> list[dict]:
    messages = [
        {"role": "system", "content": dataset_prompts.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": dataset_prompts.build_user_prompt(intent, count, avoid),
        },
    ]
    content = client.chat(messages)
    return _extract_json_array(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate intent dataset via LLM + prompt (no template fallback)"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "generated" / "intent_queries_llm.jsonl"
    )
    parser.add_argument("--per-intent", type=int, default=200, help="target rows per intent")
    parser.add_argument("--batch-size", type=int, default=25, help="rows per LLM request")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--dry-run", action="store_true", help="print prompts and exit without calling the LLM"
    )
    args = parser.parse_args()

    settings = get_settings()
    if settings.llm_backend not in {"openai", "deepseek"} or not settings.llm_api_key:
        raise SystemExit(
            "LLM 数据生成需要真实 LLM 配置（不提供模板兜底）：\n"
            "  1. 复制 .env.example 为 .env；\n"
            "  2. 设置 LLM_BACKEND=deepseek（或 openai）、LLM_API_KEY=<key>；\n"
            "  3. 重新运行本脚本。离线确定性数据请使用 scripts/generate_intent_dataset.py。"
        )

    client = LLMClient(
        base_url=settings.resolved_llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.resolved_llm_model,
        timeout=settings.llm_timeout_seconds,
    )
    preview = dataset_prompts.build_user_prompt("fault_diagnosis", args.batch_size, [])
    print(f"LLM: {settings.llm_backend} / {settings.resolved_llm_model}")
    print(f"per-intent={args.per_intent} batch={args.batch_size}")
    if args.dry_run:
        print("--- system prompt preview ---")
        print(dataset_prompts.SYSTEM_PROMPT[:600])
        print("--- user prompt preview ---")
        print(preview[:600])
        return

    rng = random.Random(args.seed)
    rows: list[dict] = []
    entity_extractor = RuleIntentClassifier()
    for intent in INTENTS:
        collected: list[str] = []
        remaining = args.per_intent
        while remaining > 0:
            batch_count = min(args.batch_size, remaining)
            for attempt in range(args.retries + 1):
                try:
                    batch = generate_batch(client, intent, batch_count, collected)
                    break
                except Exception as exc:  # noqa: BLE001 - 生成失败重试，最终失败如实报错
                    if attempt == args.retries:
                        raise SystemExit(
                            f"LLM generation failed for intent={intent}: {exc}"
                        ) from exc
                    print(f"  retry {attempt + 1} for {intent}: {exc}")
            fresh = [
                row
                for row in batch
                if row["text"] not in set(collected)  # 简单去重；相似度去重在切分阶段
            ]
            for row in fresh:
                collected.append(row["text"])
                rows.append(
                    {
                        "id": f"llm-intent-{len(rows):05d}",
                        "text": row["text"],
                        "intent": row["intent"],
                        "slots": entity_extractor._extract_entities(row["text"]),
                        "source": f"llm:{settings.resolved_llm_model}",
                    }
                )
            remaining -= len(fresh)
        print(f"{intent}: {len(collected)} rows")

    rng.shuffle(rows)
    cluster_of_row, cluster_split, residual_leaks = split_with_leakage_guard(
        rows,
        threshold=0.65,
        ratios=(0.7, 0.1, 0.2),
        seed=args.seed,
        stratify_by="intent",
    )
    for row, cluster_id in zip(rows, cluster_of_row, strict=True):
        row["cluster_id"] = cluster_id
        row["split"] = cluster_split[cluster_id]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("train", "val", "test"):
        path = args.output.with_name(f"{args.output.stem}_{split}.jsonl")
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                if row["split"] == split:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

    texts = [row["text"] for row in rows]
    print(f"wrote {len(rows)} rows to {args.output}")
    print(
        f"clusters: {len({row['cluster_id'] for row in rows})} | "
        f"unique: {len(set(texts))}/{len(texts)}"
    )
    print("splits:", json.dumps(dict(Counter(row["split"] for row in rows)), ensure_ascii=False))
    print(
        "intents:",
        json.dumps(dict(Counter(row["intent"] for row in rows)), ensure_ascii=False),
    )
    if residual_leaks:
        print(f"WARNING: {len(residual_leaks)} same-label cross-split pairs remain")
    else:
        print("leakage check: no same-label cross-split pair above threshold")
    # 追加独立复核（跨意图对也统计一次，仅提示不阻断）
    cross_label = verify_no_leakage(
        rows, [row["split"] for row in rows], threshold=0.65, same_label_only=False
    )
    print(
        f"info: {len(cross_label)} cross-label similar pairs kept across splits "
        "(hard negatives by design)"
    )


if __name__ == "__main__":
    main()
