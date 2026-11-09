import pytest

from app.evaluation.intent_metrics import (
    compute_classification_metrics,
    confusion_matrix,
    render_confusion_matrix_markdown,
    render_metrics_markdown,
)

LABELS = ["a", "b", "c"]


def test_confusion_matrix_counts_rows_and_columns() -> None:
    y_true = ["a", "a", "b", "c"]
    y_pred = ["a", "b", "b", "c"]
    matrix = confusion_matrix(y_true, y_pred, LABELS)
    assert matrix == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_metrics_hand_computed_case() -> None:
    y_true = ["a", "a", "a", "a", "b", "b", "c", "c", "c", "c"]
    y_pred = ["a", "a", "b", "a", "b", "c", "c", "c", "c", "c"]
    metrics = compute_classification_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == pytest.approx(8 / 10)
    # class a: P=3/3, R=3/4, F1=6/7 ; class b: P=1/2, R=1/2, F1=0.5 ; class c: P=4/5, R=4/4, F1=8/9
    assert metrics["per_class"]["a"]["precision"] == pytest.approx(1.0)
    assert metrics["per_class"]["a"]["recall"] == pytest.approx(0.75)
    assert metrics["per_class"]["a"]["f1"] == pytest.approx(6 / 7, rel=1e-4)
    assert metrics["per_class"]["b"]["precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["b"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["b"]["f1"] == pytest.approx(0.5)
    assert metrics["per_class"]["c"]["precision"] == pytest.approx(0.8)
    assert metrics["per_class"]["c"]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["c"]["f1"] == pytest.approx(8 / 9, rel=1e-4)

    macro_p = (1.0 + 0.5 + 0.8) / 3
    macro_r = (0.75 + 0.5 + 1.0) / 3
    macro_f1 = (6 / 7 + 0.5 + 8 / 9) / 3
    assert metrics["macro_precision"] == pytest.approx(macro_p, rel=1e-4)
    assert metrics["macro_recall"] == pytest.approx(macro_r, rel=1e-4)
    assert metrics["macro_f1"] == pytest.approx(macro_f1, rel=1e-4)
    weighted = (4 * (6 / 7) + 2 * 0.5 + 4 * (8 / 9)) / 10
    assert metrics["weighted_f1"] == pytest.approx(weighted, rel=1e-4)
    assert metrics["samples"] == 10
    assert metrics["confusion_matrix"][0][1] == 1  # a 误判为 b 一次


def test_perfect_predictions() -> None:
    y = ["a", "b", "c", "a"]
    metrics = compute_classification_metrics(y, list(y), LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_empty_and_mismatched_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_classification_metrics([], [], LABELS)
    with pytest.raises(ValueError, match="mismatch"):
        compute_classification_metrics(["a"], ["a", "b"], LABELS)


def test_markdown_rendering_contains_all_sections() -> None:
    y_true = ["a", "b"]
    y_pred = ["a", "a"]
    metrics = compute_classification_metrics(y_true, y_pred, LABELS)
    markdown = render_metrics_markdown(
        metrics, title="FiberMind Intent Classifier Evaluation", source="unit-test"
    )

    assert "# FiberMind Intent Classifier Evaluation" in markdown
    assert "| Accuracy |" in markdown
    assert "| Macro F1 |" in markdown
    assert "| Weighted F1 |" in markdown
    assert "## Per-class Metrics" in markdown
    assert "## Confusion Matrix" in markdown
    table = render_confusion_matrix_markdown(LABELS, metrics["confusion_matrix"])
    assert "| true \\ pred |" in table
