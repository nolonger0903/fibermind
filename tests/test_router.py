from app.domain.models import Intent, IntentResult, Route
from app.services.router import IntentRouter


def result(intent: Intent, confidence: float = 0.9, entities: dict | None = None) -> IntentResult:
    return IntentResult(intent=intent, confidence=confidence, entities=entities or {})


def test_cache_hit_wins_over_everything() -> None:
    decision = IntentRouter().decide(
        "ONU 报 LOS", result(Intent.FAULT_DIAGNOSIS), cache_hit=True, standard_hit=True
    )
    assert decision.route == Route.CACHE


def test_small_talk_unsupported_history_go_direct() -> None:
    router = IntentRouter()
    assert router.decide("你好", result(Intent.SMALL_TALK), False, False).route == Route.DIRECT
    assert router.decide("天气", result(Intent.UNSUPPORTED), False, False).route == Route.DIRECT
    history_decision = router.decide("上次问了啥", result(Intent.HISTORY_QUERY), False, False)
    assert history_decision.route == Route.DIRECT


def test_ambiguous_without_history_clarifies() -> None:
    decision = IntentRouter().decide("这个怎么办", result(Intent.AMBIGUOUS, 0.52), False, False)
    assert decision.route == Route.CLARIFY
    assert "ambiguous" in decision.reason


def test_ambiguous_with_history_goes_rag_for_rewrite_resolution() -> None:
    decision = IntentRouter().decide(
        "那应该先检查哪里？", result(Intent.AMBIGUOUS, 0.52), False, False, has_history=True
    )
    assert decision.route == Route.RAG
    assert "multi-turn" in decision.reason


def test_low_confidence_clarifies_and_threshold_is_configurable() -> None:
    intent = result(Intent.FAULT_DIAGNOSIS, confidence=0.5, entities={"device_type": "ONU"})

    strict = IntentRouter(confidence_threshold=0.55)
    lenient = IntentRouter(confidence_threshold=0.4)

    assert strict.decide("ONU 掉线怎么办", intent, False, False).route == Route.CLARIFY
    assert "below threshold" in strict.decide("ONU 掉线怎么办", intent, False, False).reason
    assert lenient.decide("ONU 掉线怎么办", intent, False, False).route == Route.RAG


def test_standard_route_requires_configured_min_confidence() -> None:
    router = IntentRouter(standard_min_confidence=0.75)
    low = result(Intent.ALARM_EXPLANATION, confidence=0.7)

    assert router.decide("LOS是什么意思", low, False, True).route == Route.RAG

    high = result(Intent.ALARM_EXPLANATION, confidence=0.9)
    decision = router.decide("LOS是什么意思", high, False, True)
    assert decision.route == Route.STANDARD


def test_short_fault_query_without_entities_clarifies() -> None:
    decision = IntentRouter().decide("坏了", result(Intent.FAULT_DIAGNOSIS, 0.9), False, False)
    assert decision.route == Route.CLARIFY


def test_normal_domain_query_goes_rag() -> None:
    decision = IntentRouter().decide(
        "ONU频繁闪断，收光功率-29 dBm，昨天还报了LOS，应该怎么排查？",
        result(Intent.FAULT_DIAGNOSIS, 0.93, {"alarm_codes": ["LOS"]}),
        False,
        False,
    )
    assert decision.route == Route.RAG
