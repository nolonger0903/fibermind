"""Fine-tune bert-base-chinese for FiberMind intent routing, with full evaluation.

训练/验证/测试默认使用生成器产出的“簇级切分”文件（intent_queries_{train,val,test}.jsonl），
同簇样本永远同侧，杜绝近重复样本跨集导致指标虚高；若只有全量文件，会自动执行
聚类 + 分层簇切分 + 泄露自检后再训练。

评测输出：
  * reports/intent_metrics.json     —— Accuracy / Macro P-R-F1 / Weighted F1 /
                                        per-class P-R-F1 / 混淆矩阵
  * reports/intent_report.md        —— 人读报告（含混淆矩阵表格）
  * reports/intent_confusion_matrix.png —— 需要 matplotlib（可选，缺失时跳过并提示）
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.dataset_split import split_with_leakage_guard
from app.evaluation.intent_metrics import (
    compute_classification_metrics,
    render_metrics_markdown,
)

ROOT = Path(__file__).resolve().parent.parent

LABELS = [
    "knowledge_qa",
    "fault_diagnosis",
    "operation_guide",
    "alarm_explanation",
    "history_query",
    "small_talk",
    "unsupported",
    "ambiguous",
]


def load_split_rows(data_path: Path) -> tuple[list[dict], list[dict], list[dict], str]:
    """Prefer pre-clustered train/val/test files; fall back to cluster-splitting the full file."""
    stem = data_path.with_suffix("")
    train_path = Path(f"{stem}_train.jsonl")
    val_path = Path(f"{stem}_val.jsonl")
    test_path = Path(f"{stem}_test.jsonl")
    if train_path.exists() and test_path.exists():
        def read(path: Path) -> list[dict]:
            lines = path.read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]

        source = f"{train_path.name} / {val_path.name} / {test_path.name} (cluster-level split)"
        return read(train_path), read(val_path), read(test_path), source

    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found: {data_path}. Run scripts/generate_intent_dataset.py first."
        )
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print("train/val/test files not found; running cluster-aware split on full dataset...")
    cluster_of_row, cluster_split, leaks = split_with_leakage_guard(rows, threshold=0.65)
    if leaks:
        print(f"WARNING: {len(leaks)} same-label cross-split pairs remain after guard")
    for row, cluster_id in zip(rows, cluster_of_row, strict=True):
        row["split"] = cluster_split[cluster_id]
    source = f"{data_path.name} (cluster-aware split computed at training time)"
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    test = [row for row in rows if row["split"] == "test"]
    return train, val, test, source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune bert-base-chinese with cluster-split data and full metrics"
    )
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "generated" / "intent_queries.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "intent-bert")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="smoke-test mode: cap samples per split (0 = use all)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="override total training steps (smoke tests)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="enable CUDA mixed precision (recommended for GPUs with limited memory)",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install ML dependencies first: pip install -e '.[ml]'") from exc

    train_rows, val_rows, test_rows, data_source = load_split_rows(args.data)
    if args.max_samples:
        train_rows, val_rows, test_rows = (
            rows[: args.max_samples] for rows in (train_rows, val_rows, test_rows)
        )
    print(
        f"rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} | {data_source}"
    )

    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}

    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list(
            [
                {"text": row["text"], "label": label2id[row["intent"]]}
                for row in rows
                if row["intent"] in label2id
            ]
        )

    train_dataset = to_dataset(train_rows)
    val_dataset = to_dataset(val_rows)
    test_dataset = to_dataset(test_rows)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    def metrics(prediction) -> dict[str, float]:
        predictions = np.argmax(prediction.predictions, axis=-1)
        labels = [id2label[index] for index in prediction.label_ids]
        preds = [id2label[int(index)] for index in predictions]
        full = compute_classification_metrics(labels, preds, LABELS)
        # Trainer 只接受 float 值作为优化/选型指标；完整指标在训练结束后单独输出。
        return {
            "accuracy": full["accuracy"],
            "macro_precision": full["macro_precision"],
            "macro_recall": full["macro_recall"],
            "macro_f1": full["macro_f1"],
            "weighted_f1": full["weighted_f1"],
        }

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=args.fp16,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # ---- 在独立测试集上计算完整指标并写入 reports/ ----
    print("evaluating on held-out test split...")
    test_output = trainer.predict(tokenized_test)
    predictions = np.argmax(test_output.predictions, axis=-1)
    y_true = [id2label[int(index)] for index in test_output.label_ids]
    y_pred = [id2label[int(index)] for index in predictions]
    report = compute_classification_metrics(y_true, y_pred, LABELS)

    args.reports.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "bert-base-chinese",
        "data_source": data_source,
        "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": args.fp16,
        "seed": args.seed,
        **report,
    }
    (args.reports / "intent_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    markdown = render_metrics_markdown(
        report,
        title="FiberMind Intent Classifier Evaluation",
        source=(
            f"{data_source}; model=bert-base-chinese; epochs={args.epochs}; "
            f"rows(train/val/test)={len(train_rows)}/{len(val_rows)}/{len(test_rows)}; "
            f"generated_at={metrics_payload['generated_at']}"
        ),
        extra_notes=(
            "指标由 `scripts/train_intent_bert.py` 在独立测试集上真实计算生成；"
            "数据按簇切分并通过同标签泄露自检（阈值 0.65）。"
        ),
    )
    (args.reports / "intent_report.md").write_text(markdown, encoding="utf-8")

    # ---- 混淆矩阵图片（可选依赖）----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = report["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        for row_index in range(len(LABELS)):
            for col_index in range(len(LABELS)):
                axis.text(
                    col_index,
                    row_index,
                    str(matrix[row_index][col_index]),
                    ha="center",
                    va="center",
                    color="white"
                    if matrix[row_index][col_index] > max(map(max, matrix)) / 2
                    else "black",
                    fontsize=8,
                )
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_title("Intent Confusion Matrix (test split)")
        figure.colorbar(image)
        figure.tight_layout()
        figure.savefig(args.reports / "intent_confusion_matrix.png", dpi=150)
        plt.close(figure)
        print(f"confusion matrix image: {args.reports / 'intent_confusion_matrix.png'}")
    except ImportError:
        print(
            "matplotlib not installed; skipped intent_confusion_matrix.png "
            "(markdown table is available in reports/intent_report.md)"
        )

    print(json.dumps({key: report[key] for key in
                      ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                     ensure_ascii=False, indent=2))
    print(f"reports written to {args.reports}/intent_metrics.json and intent_report.md")


if __name__ == "__main__":
    main()
