from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from app.domain.models import Intent, IntentResult
from app.prompts import hyde as hyde_prompts
from app.prompts import rewrite as rewrite_prompts
from app.utils.text import DOMAIN_SYNONYMS

logger = logging.getLogger("fibermind.rewrite")


@dataclass(slots=True)
class OptimizedQuery:
    original: str
    rewritten: str
    sub_queries: list[str] = field(default_factory=list)
    hyde: str | None = None
    strategies: list[str] = field(default_factory=list)
    backend: str = "rules"  # "llm" | "rules" | "rules(llm-fallback)"
    fallback_reason: str | None = None
    context_expanded: bool = False


CONTEXT_MARKERS = ["那", "这个", "它", "还要", "然后", "接着", "刚才"]


def _needs_context_expansion(query: str, history: list[dict[str, str]] | None) -> bool:
    """Follow-up detection shared by the rule and LLM optimizers."""
    if not history:
        return False
    has_previous_user = any(item["role"] == "user" for item in history)
    return has_previous_user and (
        len(query.strip()) < 10 or any(marker in query for marker in CONTEXT_MARKERS)
    )


class RuleQueryOptimizer:
    """Deterministic rewrite used offline and as the fallback of the LLM optimizer."""

    async def optimize(
        self,
        query: str,
        intent: IntentResult,
        history: list[dict[str, str]] | None = None,
    ) -> OptimizedQuery:
        rewritten = query.strip()
        strategies: list[str] = []
        for source, target in DOMAIN_SYNONYMS.items():
            if source in rewritten:
                rewritten = rewritten.replace(source, target)
                strategies.append("domain_rewrite")

        # 多轮上下文消解：短问句或包含指代词时，拼接上一轮用户问题（规则版兜底）。
        context_expanded = _needs_context_expansion(query, history)
        if context_expanded:
            previous_user_queries = [item["content"] for item in history if item["role"] == "user"]
            rewritten = f"{previous_user_queries[-1]}；追问：{rewritten}"

        clauses = [
            item.strip(" ，,。;；")
            for item in re.split(r"并且|同时|而且|，|,|；|;", rewritten)
            if len(item.strip()) >= 3
        ]
        sub_queries = clauses if len(clauses) > 1 else []
        if sub_queries:
            strategies.append("sub_query")

        hyde = None
        if intent.intent == Intent.FAULT_DIAGNOSIS:
            device = intent.entities.get("device_type", "光网络设备")
            alarms = "、".join(intent.entities.get("alarm_codes", [])) or "相关"
            hyde = (
                f"{device}出现{alarms}告警或异常，需检查光功率、告警关联、物理链路、"
                "供电和配置。"
            )
            strategies.append("hyde")

        if context_expanded:
            strategies.append("context_expand")

        return OptimizedQuery(
            original=query,
            rewritten=rewritten,
            sub_queries=sub_queries,
            hyde=hyde,
            strategies=list(dict.fromkeys(strategies)),
            backend="rules",
            context_expanded=context_expanded,
        )


class LLMQueryOptimizer:
    """LLM-driven rewrite / sub-query / HyDE with a rule-based fallback.

    - 走 OpenAI-compatible /chat/completions 接口（与生成器同协议，支持 DeepSeek 等）；
    - 结合多轮历史做指代消解、错别字纠错、术语规范化、多意图拆解；
    - LLM 调用失败时降级到 RuleQueryOptimizer，并在结果里显式标注
      backend="rules(llm-fallback)" 与 fallback_reason，禁止静默假装 LLM 结果。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "openai",
        timeout_seconds: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
        fallback: RuleQueryOptimizer | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError(f"LLM_API_KEY is required when rewrite backend=llm ({provider})")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.fallback = fallback or RuleQueryOptimizer()

    async def optimize(
        self,
        query: str,
        intent: IntentResult,
        history: list[dict[str, str]] | None = None,
    ) -> OptimizedQuery:
        system = rewrite_prompts.SYSTEM_PROMPT
        if intent.intent == Intent.FAULT_DIAGNOSIS:
            system = f"{system}\n{hyde_prompts.HYDE_FAULT_INSTRUCTION}"
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": rewrite_prompts.build_user_prompt(query, intent, history or []),
            },
        ]
        try:
            result = await self._request(messages)
            parsed = self._parse_payload(result, query)
            parsed.backend = "llm"
            parsed.context_expanded = _needs_context_expansion(query, history)
            if parsed.context_expanded:
                parsed.strategies.append("context_expand")
            return parsed
        except Exception as exc:  # noqa: BLE001 - LLM 任何失败都必须降级而不是打断请求
            logger.warning("LLM rewrite failed, falling back to rules: %s", exc)
            fallback_result = await self.fallback.optimize(query, intent, history)
            fallback_result.backend = "rules(llm-fallback)"
            fallback_result.fallback_reason = f"{type(exc).__name__}: {exc}"
            fallback_result.strategies.append("llm_fallback")
            return fallback_result

    async def _request(self, messages: list[dict[str, str]]) -> str:
        request_body = {"model": self.model, "messages": messages, "temperature": 0.2}
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, transport=self.transport
        ) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"])

    @staticmethod
    def _parse_payload(content: str, original: str) -> OptimizedQuery:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rewrite LLM returned non-JSON content: {content[:120]}") from exc

        rewritten = str(payload.get("rewritten") or "").strip() or original
        sub_queries = [
            str(item).strip()
            for item in payload.get("sub_queries") or []
            if str(item).strip() and str(item).strip() != rewritten
        ][:3]
        hyde_value = payload.get("hyde")
        hyde = str(hyde_value).strip() if hyde_value else None

        strategies = []
        if rewritten != original:
            strategies.append("llm_rewrite")
        if sub_queries:
            strategies.append("sub_query")
        if hyde:
            strategies.append("hyde")
        if not strategies:
            strategies.append("llm_noop")
        return OptimizedQuery(
            original=original,
            rewritten=rewritten,
            sub_queries=sub_queries,
            hyde=hyde,
            strategies=strategies,
        )
