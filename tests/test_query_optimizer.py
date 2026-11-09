import json

import httpx
import pytest

from app.domain.models import Intent, IntentResult
from app.services.query_optimizer import LLMQueryOptimizer, RuleQueryOptimizer


def fault_intent() -> IntentResult:
    return IntentResult(
        intent=Intent.FAULT_DIAGNOSIS,
        confidence=0.93,
        entities={"device_type": "ONU", "alarm_codes": ["LOS"]},
    )


@pytest.mark.asyncio
async def test_rule_optimizer_rewrites_synonyms_and_splits_sub_queries() -> None:
    result = await RuleQueryOptimizer().optimize("光猫一直闪断，同时昨天报了LOS", fault_intent())

    assert "ONU" in result.rewritten
    assert "LOS" in result.rewritten
    assert len(result.sub_queries) == 2
    assert result.hyde is not None
    assert result.backend == "rules"
    assert set(result.strategies) >= {"domain_rewrite", "sub_query", "hyde"}


@pytest.mark.asyncio
async def test_rule_optimizer_expands_followup_with_history() -> None:
    history = [
        {"role": "user", "content": "ONU 报 LOS 而且收光功率低"},
        {"role": "assistant", "content": "建议先检查光功率。"},
    ]

    result = await RuleQueryOptimizer().optimize("那应该先检查哪里？", fault_intent(), history)

    assert result.context_expanded is True
    assert "ONU" in result.rewritten
    assert "追问" in result.rewritten


def llm_optimizer(handler, **kwargs) -> LLMQueryOptimizer:
    return LLMQueryOptimizer(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
        provider="deepseek",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_llm_optimizer_parses_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "deepseek-chat"
        assert "意图识别结果" in body["messages"][1]["content"]
        payload = {
            "rewritten": "ONU 频繁掉线且接收光功率 -29 dBm 伴随 LOS 告警的排查方法",
            "sub_queries": ["ONU 掉线排查步骤", "LOS 告警处理"],
            "hyde": "ONU 出现 LOS 告警时的光功率标准与排查顺序文档。",
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    result = await llm_optimizer(handler).optimize(
        "光猫老掉线，收光-29，还报LOS咋整", fault_intent()
    )

    assert result.backend == "llm"
    assert result.rewritten.startswith("ONU 频繁掉线")
    assert len(result.sub_queries) == 2
    assert result.hyde is not None
    assert "llm_rewrite" in result.strategies and "hyde" in result.strategies
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_llm_optimizer_accepts_markdown_fenced_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = '```json\n{"rewritten": "ONU LOS 排查", "sub_queries": [], "hyde": null}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    result = await llm_optimizer(handler).optimize("光猫掉线报LOS", fault_intent())

    assert result.backend == "llm"
    assert result.rewritten == "ONU LOS 排查"


@pytest.mark.asyncio
async def test_llm_optimizer_falls_back_to_rules_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    result = await llm_optimizer(handler).optimize("光猫掉线报LOS", fault_intent())

    assert result.backend == "rules(llm-fallback)"
    assert result.fallback_reason and "503" in result.fallback_reason
    assert "llm_fallback" in result.strategies
    # 降级后仍然产出可用的规则改写结果
    assert "ONU" in result.rewritten


@pytest.mark.asyncio
async def test_llm_optimizer_falls_back_on_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "这不是 JSON"}}]}
        )

    result = await llm_optimizer(handler).optimize("光猫掉线报LOS", fault_intent())

    assert result.backend == "rules(llm-fallback)"
    assert result.fallback_reason


def test_llm_optimizer_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        LLMQueryOptimizer(
            base_url="https://api.deepseek.com", api_key="", model="deepseek-chat"
        )
