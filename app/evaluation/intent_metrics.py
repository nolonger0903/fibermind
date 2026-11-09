"""Pure-Python classification metrics for the intent classifier.

不依赖 sklearn/numpy/torch：同一份实现既被 HuggingFace Trainer 的
compute_metrics 调用，也被单元测试直接验证，保证“训练脚本输出的指标”
与“测试断言的指标”是同一套代码。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """rows = true label, cols = predicted label."""
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred, strict=True):
        if true in index and pred in index:
            matrix[index[true]][index[pred]] += 1
    return matrix


def compute_classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict:
    """Accuracy / Macro P-R-F1 / Weighted F1 / per-class P-R-F1 / confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("cannot compute metrics on empty sequences")

    matrix = confusion_matrix(y_true, y_pred, labels)
    support = Counter(y_true)
    total = len(y_true)

    per_class: dict[str, dict[str, float]] = {}
    macro_precision = macro_recall = macro_f1 = 0.0
    weighted_f1 = 0.0
    correct = 0

    for row_index, label in enumerate(labels):
        row = matrix[row_index]
        predicted = sum(matrix[other][row_index] for other in range(len(labels)))
        true_total = sum(row)
        correct += row[row_index]

        precision = _safe_div(row[row_index], predicted)
        recall = _safe_div(row[row_index], true_total)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        label_support = support.get(label, 0)

        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": label_support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_f1 += f1 * label_support

    label_count = len(labels)
    return {
        "accuracy": round(correct / total, 6),
        "macro_precision": round(macro_precision / label_count, 6),
        "macro_recall": round(macro_recall / label_count, 6),
        "macro_f1": round(macro_f1 / label_count, 6),
        "weighted_f1": round(weighted_f1 / total, 6),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": list(labels),
        "samples": total,
    }


def render_confusion_matrix_markdown(
    labels: Sequence[str], matrix: list[list[int]]
) -> str:
    """Render a confusion matrix as a GitHub-flavored markdown table."""
    header = "| true \\ pred | " + " | ".join(labels) + " |"
    separator = "| --- | " + " | ".join("---" for _ in labels) + " |"
    rows = [
        "| " + label + " | " + " | ".join(str(cell) for cell in matrix[index]) + " |"
        for index, label in enumerate(labels)
    ]
    return "\n".join([header, separator, *rows])


def render_metrics_markdown(
    metrics: dict,
    *,
    title: str = "Intent Classifier Evaluation",
    source: str = "",
    extra_notes: str = "",
) -> str:
    """Render the full metric report as markdown (reports/intent_report.md)."""
    lines = [
        f"# {title}",
        "",
        f"- 数据来源：{source or '未注明'}",
        f"- 样本数：{metrics['samples']}",
        f"- 标签数：{len(metrics['labels'])}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro Precision | {metrics['macro_precision']:.4f} |",
        f"| Macro Recall | {metrics['macro_recall']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        f"| Weighted F1 | {metrics['weighted_f1']:.4f} |",
        "",
        "## Per-class Metrics",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, values in metrics["per_class"].items():
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} "
            f"| {values['f1']:.4f} | {values['support']} |"
        )
    lines += [
        "",
        "## Confusion Matrix",
        "",
        render_confusion_matrix_markdown(metrics["labels"], metrics["confusion_matrix"]),
    ]
    if extra_notes:
        lines += ["", "## Notes", "", extra_notes]
    return "\n".join(lines) + "\n"
