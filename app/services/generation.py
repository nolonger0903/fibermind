from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from app.domain.models import Intent, RetrievalResult
from app.prompts import diagnosis as diagnosis_prompts
from app.prompts.system import INSUFFICIENT_CONTEXT_TEMPLATE

TokenCallback = Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    possible_causes: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = True


class GroundedTemplateGenerator:
    """Offline generator that synthesizes only retrieved, source-backed instructions."""

    native_streaming = False

    async def generate(
        self,
        query: str,
        intent: Intent,
        contexts: list[RetrievalResult],
        history: list[dict[str, str]],
        on_token: TokenCallback | None = None,
    ) -> GeneratedAnswer:
        if not contexts or contexts[0].rerank_score < 0.08:
            return GeneratedAnswer(
                answer=INSUFFICIENT_CONTEXT_TEMPLATE,
                confidence=0.2,
                grounded=False,
            )

        parent_contexts: list[RetrievalResult] = []
        seen_parents: set[str] = set()
        for result in contexts:
            if result.chunk.parent_id not in seen_parents:
                seen_parents.add(result.chunk.parent_id)
                parent_contexts.append(result)

        primary = parent_contexts[0]
        causes = [
            f"可能与“{item.chunk.metadata['fault_type']}”相关（依据：{item.chunk.title}）"
            for item in parent_contexts[:3]
        ]
        steps = self._extract_steps(primary.chunk.metadata["parent_content"])
        source_lines = "；".join(
            f"[{index}] {item.chunk.title}"
            for index, item in enumerate(parent_contexts[:3], start=1)
        )
        if intent == Intent.FAULT_DIAGNOSIS:
            opening = "根据告警和现象，建议先按影响范围判断公共故障点，再从光路到设备逐层排查。"
        else:
            opening = f"与该问题最相关的是“{primary.chunk.title}”。"
        answer = f"{opening}\n\n排查建议：\n" + "\n".join(
            f"{index}. {step}" for index, step in enumerate(steps[:5], start=1)
        )
        answer += f"\n\n参考知识：{source_lines}。"
        confidence = min(0.94, 0.48 + primary.rerank_score * 0.55)
        return GeneratedAnswer(answer, causes, steps[:5], confidence, True)

    @staticmethod
    def _extract_steps(content: str) -> list[str]:
        if "排查步骤：" in content:
            body = content.split("排查步骤：", maxsplit=1)[1]
            raw_steps = [part.strip(" 。；") for part in body.replace("；", "。").split("。")]
        else:
            raw_steps = [part.strip(" 。；") for part in content.replace("；", "。").split("。")]
        cleaned = [re.sub(r"^\d+[.、]\s*", "", step) for step in raw_steps]
        return [step for step in cleaned if len(step) >= 8]

    async def stream(self, text: str, chunk_size: int = 18) -> AsyncIterator[str]:
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]


class OpenAICompatibleGenerator:
    native_streaming = True

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "openai",
        timeout_seconds: float = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError(f"LLM_API_KEY is required when LLM_BACKEND={provider}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def generate(
        self,
        query: str,
        intent: Intent,
        contexts: list[RetrievalResult],
        history: list[dict[str, str]],
        on_token: TokenCallback | None = None,
    ) -> GeneratedAnswer:
        context_text = "\n\n".join(
            f"[{index}] {item.chunk.title}\n{item.chunk.metadata['parent_content']}"
            for index, item in enumerate(contexts, start=1)
        )
        messages = [
            {"role": "system", "content": diagnosis_prompts.SYSTEM_PROMPT},
            *history[-6:],
            {"role": "user", "content": diagnosis_prompts.build_user_prompt(query, context_text)},
        ]
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                if on_token:
                    answer = await self._stream_completion(client, request_body, on_token)
                else:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=request_body,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    answer = payload["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:300]
            raise RuntimeError(
                f"{self.provider} LLM request failed with HTTP {status}: {detail}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{self.provider} LLM returned an invalid response: {exc}") from exc
        confidence = min(0.9, 0.45 + (contexts[0].rerank_score if contexts else 0) * 0.5)
        return GeneratedAnswer(answer=answer, confidence=confidence, grounded=bool(contexts))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        request_body: dict,
        on_token: TokenCallback,
    ) -> str:
        parts: list[str] = []
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={**request_body, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                payload = json.loads(data)
                token = payload["choices"][0].get("delta", {}).get("content") or ""
                if token:
                    parts.append(token)
                    await on_token(token)
        if not parts:
            raise ValueError("stream completed without content")
        return "".join(parts)
