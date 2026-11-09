import pytest

from app.config import Settings
from app.container import build_container
from app.domain.models import Intent
from app.services.intent import (
    BertIntentClassifier,
    ResilientIntentClassifier,
    RuleIntentClassifier,
)


def test_fault_intent_extracts_domain_entities() -> None:
    result = RuleIntentClassifier().predict("ONU 收光 -29 dBm 并报 LOS，频繁掉线")

    assert result.intent == Intent.FAULT_DIAGNOSIS
    assert result.confidence >= 0.9
    assert result.entities["device_type"] == "ONU"
    assert result.entities["alarm_codes"] == ["LOS"]
    assert result.entities["power_values"] == ["-29 dBm"]


def test_alarm_explanation_has_priority_over_alarm_keyword() -> None:
    result = RuleIntentClassifier().predict("LOF 是什么意思？")

    assert result.intent == Intent.ALARM_EXPLANATION


def test_alarm_code_is_detected_next_to_chinese_text() -> None:
    result = RuleIntentClassifier().predict("设备昨天报了LOS，现在业务中断")

    assert result.entities["alarm_codes"] == ["LOS"]
    assert result.confidence >= 0.9


def test_no_signal_query_is_ambiguous_not_force_classified() -> None:
    result = RuleIntentClassifier().predict("这个怎么办")

    assert result.intent == Intent.AMBIGUOUS
    assert result.confidence < 0.55


def test_multi_intent_query_is_ambiguous() -> None:
    # 同时命中“解释类”与“排查类”强信号 → ambiguous，交给 clarify/rewrite
    result = RuleIntentClassifier().predict("LOS是什么意思？现场一直闪断要怎么排查处理？")

    assert result.intent == Intent.AMBIGUOUS


class FakeBert:
    """BERT 适配器的测试替身：可编程抛错或返回固定结果。"""

    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls = 0

    def predict(self, query: str):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results.pop(0)


def test_bert_is_preferred_when_available() -> None:
    from app.domain.models import IntentResult

    bert = FakeBert(
        results=[IntentResult(Intent.FAULT_DIAGNOSIS, 0.99, {}, "bert-base-chinese")]
    )
    classifier = ResilientIntentClassifier(bert)  # type: ignore[arg-type]

    result = classifier.predict("ONU 掉线")

    assert result.classifier == "bert-base-chinese"
    assert classifier.backend == "bert"
    assert classifier.fallback_events == []


def test_bert_runtime_error_falls_back_to_rule_with_trace_event() -> None:
    bert = FakeBert(error=RuntimeError("cuda oom"))
    classifier = ResilientIntentClassifier(bert)  # type: ignore[arg-type]

    result = classifier.predict("ONU 收光 -29 dBm 并报 LOS，频繁掉线")

    assert result.classifier == "rule(fallback:bert)"
    assert result.intent == Intent.FAULT_DIAGNOSIS
    events = classifier.pending_fallback_events()
    assert len(events) == 1
    assert events[0]["code"] == "bert-runtime-error"
    assert "cuda oom" in str(events[0]["message"])
    # 事件被取走后不再重复上报
    assert classifier.pending_fallback_events() == []


def test_bert_circuit_opens_after_consecutive_failures() -> None:
    bert = FakeBert(error=RuntimeError("model corrupted"))
    classifier = ResilientIntentClassifier(bert, circuit_threshold=3)  # type: ignore[arg-type]

    for _ in range(3):
        classifier.predict("ONU 掉线")

    assert classifier.circuit_open is True
    assert classifier.backend == "rule"
    # 熔断后不再调用 BERT
    calls_after_open = bert.calls
    classifier.predict("ONU 掉线")
    assert bert.calls == calls_after_open


def test_fallback_on_error_disabled_raises() -> None:
    bert = FakeBert(error=RuntimeError("boom"))
    classifier = ResilientIntentClassifier(
        bert, fallback_on_error=False  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="boom"):
        classifier.predict("ONU 掉线")


def test_missing_bert_model_degrades_to_rule_in_container() -> None:
    settings = Settings(
        _env_file=None, intent_backend="bert", intent_model_path="models/does-not-exist"
    )
    container = build_container(settings)

    result = container.intent_classifier.predict("ONU 收光 -29 dBm 并报 LOS，频繁掉线")

    assert result.intent == Intent.FAULT_DIAGNOSIS
    assert result.classifier == "rule(fallback:bert-unavailable)"


def test_bert_classifier_constructor_reports_missing_model_clearly() -> None:
    with pytest.raises(Exception, match="does-not-exist|ml' extra|No such file"):
        BertIntentClassifier("models/does-not-exist")
