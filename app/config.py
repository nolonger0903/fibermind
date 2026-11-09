from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.retrieval_weights import normalize_fusion_weights

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FiberMind"
    app_env: str = "demo"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    cache_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 600

    session_backend: str = "memory"
    mysql_dsn: str = "mysql+pymysql://fibermind:fibermind@localhost:3306/fibermind"

    retrieval_backend: str = "memory"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "fiber_knowledge_chunks"
    dense_weight: float = Field(default=0.55, ge=0, le=1)
    sparse_weight: float = Field(default=0.45, ge=0, le=1)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    rerank_top_n: int = Field(default=4, ge=1, le=20)

    intent_backend: str = "rule"
    intent_model_path: str = "models/intent-bert"
    embedding_backend: str = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_timeout_seconds: float = Field(default=60, gt=0, le=300)
    rerank_backend: str = "local"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    llm_backend: str = "grounded"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)

    ragas_llm_base_url: str = "https://api.openai.com/v1"
    ragas_llm_api_key: str = ""
    ragas_llm_model: str = "gpt-4o-mini"
    ragas_embedding_base_url: str = "https://api.siliconflow.cn/v1"
    ragas_embedding_api_key: str = ""
    ragas_embedding_model: str = "BAAI/bge-m3"
    ragas_timeout_seconds: float = Field(default=300, gt=0, le=600)
    ragas_max_retries: int = Field(default=3, ge=0, le=10)

    # 路由 / 降级 / 澄清策略阈值（全部配置化，避免硬编码）
    intent_confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    router_standard_min_confidence: float = Field(default=0.75, ge=0, le=1)
    intent_fallback_on_error: bool = True

    rewrite_use_llm: bool = True
    rewrite_offline_max_sub_queries: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def _validate_fusion_weights(self) -> Settings:
        # 启动期即校验 dense/sparse 权重组合，非法配置直接失败，禁止静默默认值。
        normalize_fusion_weights(self.dense_weight, self.sparse_weight)
        if self.retrieval_backend == "bge_api" and not self.embedding_api_key:
            raise ValueError(
                "EMBEDDING_API_KEY is required when RETRIEVAL_BACKEND=bge_api"
            )
        if self.rerank_backend == "bge_api" and not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY is required when RERANK_BACKEND=bge_api")
        return self

    @property
    def resolved_llm_base_url(self) -> str:
        if self.llm_backend == "deepseek" and self.llm_base_url == "https://api.openai.com/v1":
            return "https://api.deepseek.com"
        return self.llm_base_url

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_backend == "deepseek" and self.llm_model == "gpt-4o-mini":
            return "deepseek-chat"
        return self.llm_model

    @property
    def knowledge_path(self) -> Path:
        return ROOT_DIR / "data" / "knowledge.json"

    @property
    def standard_answers_path(self) -> Path:
        return ROOT_DIR / "data" / "standard_answers.json"

    @property
    def static_path(self) -> Path:
        return ROOT_DIR / "app" / "static"


@lru_cache
def get_settings() -> Settings:
    return Settings()
