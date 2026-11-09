from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.container import build_container
from app.utils.text import normalize_text

ROOT = Path(__file__).resolve().parent.parent


def keyword_recall(answer: str, keywords: list[str]) -> float:
    normalized = normalize_text(answer)
    return sum(normalize_text(word) in normalized for word in keywords) / len(keywords)


async def evaluate(dataset_path: Path) -> list[dict]:
    # Offline evaluation must never spend an external LLM budget implicitly.
    container = build_container(get_settings().model_copy(update={"llm_backend": "grounded"}))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = []
    for index, sample in enumerate(dataset, start=1):
        response = await container.pipeline.run(sample["question"], session_id=f"eval-{index}")
        retrieved_ids = [item["document_id"] for item in response["citations"]]
        expected = set(sample["expected_document_ids"])
        context_recall = len(expected & set(retrieved_ids)) / len(expected)
        rows.append(
            {
                "question": sample["question"],
                "answer": response["answer"],
                "faithfulness_proxy": 1.0 if response["diagnosis"]["grounded"] else 0.0,
                "answer_relevancy_proxy": keyword_recall(
                    response["answer"], sample["ground_truth_keywords"]
                ),
                "context_recall": context_recall,
                "context_relevancy_proxy": 1.0
                if retrieved_ids and retrieved_ids[0] in expected
                else 0.0,
                "latency_ms": response["total_latency_ms"],
                "route": response["route"],
                "retrieved_ids": retrieved_ids,
            }
        )
    return rows


def write_reports(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "evaluation.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[key for key in rows[0] if key != "retrieved_ids"])
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "retrieved_ids"} for row in rows
        )

    metric_names = [
        "faithfulness_proxy",
        "answer_relevancy_proxy",
        "context_recall",
        "context_relevancy_proxy",
        "latency_ms",
    ]
    summary = {name: round(mean(row[name] for row in rows), 4) for name in metric_names}
    lines = [
        "# FiberMind Offline Evaluation",
        "",
        "> Proxy metrics are deterministic smoke-test metrics. "
        "Use `run_ragas.py` for model-based RAGAS scores.",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
        *[f"| {name} | {value} |" for name, value in summary.items()],
        "",
        "## Cases",
        "",
        *[
            f"- `{row['context_recall']:.2f}` recall / "
            f"`{row['latency_ms']:.1f} ms`: {row['question']}"
            for row in rows
        ],
    ]
    (output_dir / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic FiberMind offline evaluation")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "eval_dataset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    rows = asyncio.run(evaluate(args.dataset))
    write_reports(rows, args.output)


if __name__ == "__main__":
    main()
