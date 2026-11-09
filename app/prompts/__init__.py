"""Centralized prompt templates for FiberMind.

所有 LLM 使用的 prompt 统一在本包管理，业务代码只允许引用，不允许内联长文本。
每个模块暴露一个模块级 docstring 风格的常量或 build_* 函数。
"""

from app.prompts import clarify, dataset, diagnosis, hyde, rewrite, system

__all__ = ["system", "diagnosis", "rewrite", "hyde", "clarify", "dataset"]
PROMPT_VERSION = "2026-09-17-evidence-v2"
