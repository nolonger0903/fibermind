"""Run model-based RAGAS evaluation with explicit judge and embedding providers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from app.config import Settings, get_settings
from app.container import build_container
from app.prompts import PROMPT_VERSION

ROOT = Path(__file__).resolve().parent.parent


async def collect_samples() -> list[dict]:
    container = build_container(get_settings())
    dataset = json.loads((ROOT / "evaluation" / "eval_dataset.json").read_text(encoding="utf-8"))
    samples = []
    for index, row in enumerate(dataset):
        response = await container.pipeline.run(row["question"], f"ragas-{index}")
        samples.append(
            {
                "user_input": row["question"],
                "response": response["answer"],
                "retrieved_contexts": [citation["excerpt"] for citation in response["citations"]],
                "reference": "；".join(row["ground_truth_keywords"]),
            }
        )
    return samples


def _evaluator_config(settings: Settings) -> tuple[dict[str, str], dict[str, str]]:
    if settings.ragas_llm_api_key:
        llm = {
            "base_url": settings.ragas_llm_base_url,
            "api_key": settings.ragas_llm_api_key,
            "model": settings.ragas_llm_model,
        }
    elif settings.llm_api_key and settings.llm_backend in {"openai", "deepseek"}:
        llm = {
            "base_url": settings.resolved_llm_base_url,
            "api_key": settings.llm_api_key,
            "model": settings.resolved_llm_model,
        }
    else:
        raise SystemExit(
            "Configure RAGAS_LLM_API_KEY (recommended) or an application LLM_API_KEY first"
        )

    embedding_key = settings.ragas_embedding_api_key or settings.embedding_api_key
    if not embedding_key:
        raise SystemExit(
            "Configure RAGAS_EMBEDDING_API_KEY or EMBEDDING_API_KEY for RAGAS"
        )
    embeddings = {
        "base_url": settings.ragas_embedding_base_url,
        "api_key": embedding_key,
        "model": settings.ragas_embedding_model,
    }
    return llm, embeddings


def main() -> None:
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise SystemExit("Install evaluation dependencies first: pip install -e '.[eval]'") from exc

    settings = get_settings()
    llm_config, embedding_config = _evaluator_config(settings)
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            temperature=0,
            timeout=settings.ragas_timeout_seconds,
            max_retries=settings.ragas_max_retries,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            base_url=embedding_config["base_url"],
            api_key=embedding_config["api_key"],
            model=embedding_config["model"],
            check_embedding_ctx_length=False,
        )
    )
    dataset = EvaluationDataset.from_list(asyncio.run(collect_samples()))
    run_config = RunConfig(
        timeout=settings.ragas_timeout_seconds,
        max_retries=settings.ragas_max_retries,
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )
    output = ROOT / "reports" / "ragas.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(result.to_pandas().to_json(force_ascii=False, orient="records"))
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_without_reference",
        "context_recall",
    ]
    summary = {}
    for metric in metric_names:
        valid = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary[metric] = {
            "mean": round(mean(valid), 6) if valid else None,
            "valid_samples": len(valid),
            "failed_samples": len(rows) - len(valid),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge": {
            "base_url": llm_config["base_url"],
            "model": llm_config["model"],
        },
        "embeddings": {
            "base_url": embedding_config["base_url"],
            "model": embedding_config["model"],
        },
        "run_config": {
            "timeout_seconds": settings.ragas_timeout_seconds,
            "max_retries": settings.ragas_max_retries,
        },
        "prompt_version": PROMPT_VERSION,
        "summary": summary,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# FiberMind RAGAS Evaluation",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        f"- Judge: `{llm_config['model']}`",
        f"- Embeddings: `{embedding_config['model']}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        f"- Samples: `{len(rows)}`",
        "",
        "| Metric | Mean | Valid | Failed |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {metric} | {values['mean']} | {values['valid_samples']} | "
            f"{values['failed_samples']} |"
            for metric, values in summary.items()
        ],
        "",
        "> Model-based scores on six fixed regression cases. This is a reproducible smoke "
        "evaluation, not a production-scale benchmark.",
    ]
    (ROOT / "reports" / "ragas.md").write_text("\n".join(markdown), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
