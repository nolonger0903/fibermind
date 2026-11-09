import pytest

from app.config import Settings
from app.container import build_container


def test_deepseek_uses_provider_defaults() -> None:
    settings = Settings(llm_backend="deepseek", llm_api_key="test-key")

    assert settings.resolved_llm_base_url == "https://api.deepseek.com"
    assert settings.resolved_llm_model == "deepseek-chat"


def test_explicit_deepseek_endpoint_and_model_win() -> None:
    settings = Settings(
        llm_backend="deepseek",
        llm_api_key="test-key",
        llm_base_url="https://gateway.example/v1",
        llm_model="deepseek-reasoner",
    )

    assert settings.resolved_llm_base_url == "https://gateway.example/v1"
    assert settings.resolved_llm_model == "deepseek-reasoner"


def test_deepseek_requires_api_key_at_startup() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY.*deepseek"):
        build_container(Settings(llm_backend="deepseek", llm_api_key=""))


def test_bge_api_backends_require_embedding_key() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        Settings(_env_file=None, retrieval_backend="bge_api")
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        Settings(_env_file=None, rerank_backend="bge_api")


def test_bge_api_backend_accepts_explicit_key() -> None:
    settings = Settings(
        _env_file=None,
        retrieval_backend="bge_api",
        rerank_backend="bge_api",
        embedding_backend="bge_api",
        embedding_api_key="test-key",
    )

    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
