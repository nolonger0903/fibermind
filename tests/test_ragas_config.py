import pytest

from app.config import Settings
from evaluation.run_ragas import _evaluator_config


def test_ragas_uses_dedicated_judge_and_embedding_credentials() -> None:
    settings = Settings(
        _env_file=None,
        ragas_llm_base_url="https://judge.example/v1",
        ragas_llm_api_key="judge-key",
        ragas_llm_model="judge-model",
        ragas_embedding_base_url="https://embed.example/v1",
        ragas_embedding_api_key="embed-key",
        ragas_embedding_model="BAAI/bge-m3",
    )

    llm, embeddings = _evaluator_config(settings)

    assert llm == {
        "base_url": "https://judge.example/v1",
        "api_key": "judge-key",
        "model": "judge-model",
    }
    assert embeddings["api_key"] == "embed-key"
    assert embeddings["model"] == "BAAI/bge-m3"


def test_ragas_requires_both_judge_and_embedding_credentials() -> None:
    with pytest.raises(SystemExit, match="RAGAS_LLM_API_KEY"):
        _evaluator_config(Settings(_env_file=None))

    settings = Settings(
        _env_file=None,
        ragas_llm_api_key="judge-key",
        ragas_embedding_api_key="",
        embedding_api_key="",
    )
    with pytest.raises(SystemExit, match="RAGAS_EMBEDDING_API_KEY"):
        _evaluator_config(settings)
