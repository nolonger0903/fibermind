"""Keep the test suite deterministic and free of external LLM calls.

本地 `.env`（可能包含真实 LLM Key 与 DeepSeek 配置）不允许影响测试：
pydantic-settings 中环境变量优先级高于 dotenv，因此这里把全部后端选择
固定为轻量实现，测试永远不会访问真实 API / Redis / MySQL / Milvus。
"""

import os

os.environ["LLM_BACKEND"] = "grounded"
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_MODEL"] = "gpt-4o-mini"
os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"
os.environ["CACHE_BACKEND"] = "memory"
os.environ["SESSION_BACKEND"] = "memory"
os.environ["RETRIEVAL_BACKEND"] = "memory"
os.environ["RERANK_BACKEND"] = "local"
os.environ["INTENT_BACKEND"] = "rule"
