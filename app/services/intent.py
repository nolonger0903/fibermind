from __future__ import annotations

import logging
import re
from pathlib import Path

from app.domain.models import Intent, IntentResult

logger = logging.getLogger("fibermind.intent")


class RuleIntentClassifier:
    """Explainable fallback classifier used when the fine-tuned BERT model is unavailable."""

    DEVICE_PATTERNS = ["ONU", "ONT", "OLT", "PON", "光模块", "光纤", "传输设备"]
    ALARM_PATTERN = re.compile(
        r"(?<![A-Z0-9_])(LOS|LOF|OOF|AIS|BER_HIGH|CRC_ERROR|DYING_GASP|AUTH_FAIL|"
        r"RX_POWER_(?:LOW|HIGH))(?![A-Z0-9_])",
        re.IGNORECASE,
    )

    WEAK_FAULT_MARKERS = {"怎么办", "处理", "定位", "异常"}

    def predict(self, query: str) -> IntentResult:
        normalized = query.upper()
        entities = self._extract_entities(query)
        signals = self._intent_signals(query, normalized, entities)

        # 混合意图 1：解释类信号 + 动作类信号同时出现
        # （如“LOS是什么意思，顺便问下怎么排查处理”）→ ambiguous。
        action_hits = signals["fault_diagnosis"] + signals["operation_guide"]
        if signals["alarm_explanation"] >= 1 and action_hits >= 2:
            return IntentResult(Intent.AMBIGUOUS, 0.52, entities, "rule", signals=signals)

        # 混合意图 2：多个意图各有 >=2 次强信号命中 → ambiguous。
        strong_intents = [name for name, hits in signals.items() if hits >= 2]
        if len(strong_intents) >= 2:
            return IntentResult(Intent.AMBIGUOUS, 0.52, entities, "rule", signals=signals)

        # 单一弱信号（如只有“怎么办”）且没有任何设备/告警/数值实体：
        # 信息不足，ambiguous，交给 clarify 追问。
        weak_only_fault = signals["fault_diagnosis"] == 1 and any(
            marker in query for marker in self.WEAK_FAULT_MARKERS
        )
        if weak_only_fault and not entities:
            return IntentResult(Intent.AMBIGUOUS, 0.45, entities, "rule", signals=signals)

        if signals["small_talk"]:
            return IntentResult(Intent.SMALL_TALK, 0.97, entities, "rule", signals=signals)
        if signals["history_query"]:
            return IntentResult(Intent.HISTORY_QUERY, 0.88, entities, "rule", signals=signals)
        if signals["unsupported"]:
            return IntentResult(Intent.UNSUPPORTED, 0.96, entities, "rule", signals=signals)
        if signals["alarm_explanation"]:
            return IntentResult(Intent.ALARM_EXPLANATION, 0.94, entities, "rule", signals=signals)
        if signals["operation_guide"]:
            return IntentResult(Intent.OPERATION_GUIDE, 0.87, entities, "rule", signals=signals)
        if signals["fault_diagnosis"]:
            confidence = 0.93 if entities.get("alarm_codes") else 0.84
            return IntentResult(
                Intent.FAULT_DIAGNOSIS, confidence, entities, "rule", signals=signals
            )
        if signals["knowledge_qa"]:
            return IntentResult(Intent.KNOWLEDGE_QA, 0.82, entities, "rule", signals=signals)

        # 没有任何领域信号：信息不足，判为 ambiguous 交给 clarify 追问，
        # 不再武断落到 knowledge_qa。
        return IntentResult(Intent.AMBIGUOUS, 0.4, entities, "rule", signals=signals)

    def _intent_signals(
        self, query: str, normalized: str, entities: dict[str, object]
    ) -> dict[str, int]:
        has_alarm_code = bool(self.ALARM_PATTERN.search(normalized))
        fault_markers = [
            "故障",
            "掉线",
            "闪断",
            "中断",
            "异常",
            "告警",
            "红灯",
            "误码",
            "丢包",
            "上不去",
            "离线",
            "怎么办",
            "排查",
            "定位",
            "处理",
        ]
        return {
            "small_talk": sum(word in query for word in ["你好", "谢谢", "你是谁", "再见", "在吗"]),
            "history_query": sum(
                word in query for word in ["上次", "之前问", "历史记录", "聊天记录", "刚才问"]
            ),
            "unsupported": sum(
                word in query for word in ["天气", "电影", "写一首诗", "股票", "菜谱"]
            ),
            "alarm_explanation": sum(
                word in query
                for word in ["是什么意思", "什么是", "含义", "解释", "区别", "代表什么"]
            ),
            "operation_guide": sum(
                word in query
                for word in [
                    "怎么操作", "如何配置", "清洁", "怎么更换", "步骤", "怎么做", "如何测试",
                ]
            ),
            "fault_diagnosis": sum(word in query for word in fault_markers)
            + (1 if has_alarm_code else 0),
            "knowledge_qa": sum(
                word in query for word in ["光功率", "范围", "规范", "原理", "多少", "正常值"]
            ),
        }

    def _extract_entities(self, query: str) -> dict[str, object]:
        upper_query = query.upper()
        devices = [item for item in self.DEVICE_PATTERNS if item.upper() in upper_query]
        alarms = [match.upper() for match in self.ALARM_PATTERN.findall(upper_query)]
        power_values = re.findall(r"-?\d+(?:\.\d+)?\s*dBm", query, flags=re.IGNORECASE)
        symptoms = [
            item
            for item in ["掉线", "闪断", "中断", "弱光", "强光", "误码", "丢包", "红灯", "离线"]
            if item in query
        ]
        result: dict[str, object] = {}
        if devices:
            result["device_type"] = devices[0]
        if alarms:
            result["alarm_codes"] = list(dict.fromkeys(alarms))
        if power_values:
            result["power_values"] = power_values
        if symptoms:
            result["symptoms"] = symptoms
        return result


class BertIntentClassifier:
    """Optional adapter for a fine-tuned Hugging Face sequence-classification model."""

    def __init__(self, model_path: str) -> None:
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(__file__).resolve().parents[2] / resolved_path
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"BERT intent model directory does not exist: {resolved_path}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install FiberMind with the 'ml' extra to use BERT intent routing"
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolved_path)
        self.model.eval()
        self.fallback = RuleIntentClassifier()

    def predict(self, query: str) -> IntentResult:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        confidence, label_id = self.torch.max(probabilities, dim=-1)
        label = self.model.config.id2label[int(label_id)]
        entities = self.fallback._extract_entities(query)
        return IntentResult(Intent(label), float(confidence), entities, "bert-base-chinese")


class ResilientIntentClassifier:
    """BERT-first intent classifier with an explicit, observable rule fallback.

    契约：
      * BERT 可用时永远优先使用 BERT；
      * BERT 推理抛错时降级到 Rule 分类器（可配置关闭），系统绝不整体崩溃；
      * 每次降级都写入日志（logger.warning）并记录到 ``fallback_events``，
        由 pipeline 透传到 trace，禁止静默 fallback 后伪装 BERT 正常工作；
      * 结果的 ``classifier`` 字段标注真实来源（例如 ``rule(fallback:bert-runtime-error)``）；
      * 熔断：连续推理失败达到 ``circuit_threshold`` 后不再尝试 BERT，
        避免每个请求都重复付出失败的推理开销。
    """

    def __init__(
        self,
        primary: BertIntentClassifier | None,
        fallback: RuleIntentClassifier | None = None,
        *,
        fallback_on_error: bool = True,
        circuit_threshold: int = 3,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or RuleIntentClassifier()
        self.fallback_on_error = fallback_on_error
        self.circuit_threshold = circuit_threshold
        self.consecutive_failures = 0
        self.circuit_open = primary is None
        self.fallback_events: list[dict[str, object]] = []
        if primary is None:
            self._record("bert-unavailable", "primary classifier was not constructed")

    @property
    def backend(self) -> str:
        return "bert" if (self.primary is not None and not self.circuit_open) else "rule"

    def _record(self, code: str, message: str) -> None:
        import time

        event = {"code": code, "message": message, "at": round(time.time(), 3)}
        self.fallback_events.append(event)
        logger.warning("intent fallback [%s]: %s", code, message)

    def predict(self, query: str) -> IntentResult:
        if self.primary is not None and not self.circuit_open:
            try:
                result = self.primary.predict(query)
                self.consecutive_failures = 0
                return result
            except Exception as exc:  # noqa: BLE001 - 任何推理错误都必须降级而不是崩溃
                self.consecutive_failures += 1
                self._record(
                    "bert-runtime-error",
                    f"{type(exc).__name__}: {exc} (consecutive failures="
                    f"{self.consecutive_failures})",
                )
                if self.consecutive_failures >= self.circuit_threshold:
                    self.circuit_open = True
                    self._record(
                        "bert-circuit-opened",
                        f"BERT disabled after {self.consecutive_failures} consecutive failures",
                    )
                if not self.fallback_on_error:
                    raise
        result = self.fallback.predict(query)
        result.classifier = (
            "rule(fallback:bert-unavailable)" if self.primary is None else "rule(fallback:bert)"
        )
        return result

    def pending_fallback_events(self) -> list[dict[str, object]]:
        """Return and clear fallback events so the pipeline can attach them to the trace."""
        events, self.fallback_events = self.fallback_events, []
        return events
