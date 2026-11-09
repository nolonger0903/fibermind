import json

import httpx
import pytest

from app.domain.models import Intent, KnowledgeChunk, RetrievalResult
from app.services.generation import OpenAICompatibleGenerator


@pytest.mark.asyncio
async def test_deepseek_stream_is_forwarded_and_collected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert body["model"] == "deepseek-chat"
        assert body["stream"] is True
        content = (
            'data: {"choices":[{"delta":{"content":"先检查"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"光功率"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=content)

    chunk = KnowledgeChunk(
        id="chunk-1",
        parent_id="doc-1",
        title="ONU LOS 处理",
        content="检查光功率。",
        metadata={"parent_content": "先检查接收光功率。"},
    )
    context = RetrievalResult(chunk=chunk, rerank_score=0.8)
    tokens: list[str] = []

    async def on_token(token: str) -> None:
        tokens.append(token)

    generator = OpenAICompatibleGenerator(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
        provider="deepseek",
        transport=httpx.MockTransport(handler),
    )
    result = await generator.generate(
        query="ONU 报 LOS 怎么办？",
        intent=Intent.FAULT_DIAGNOSIS,
        contexts=[context],
        history=[],
        on_token=on_token,
    )

    assert tokens == ["先检查", "光功率"]
    assert result.answer == "先检查光功率"
    assert result.grounded is True
