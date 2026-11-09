import pytest

from app.evaluation.dataset_split import (
    assign_clusters_by_similarity,
    jaccard_similarity,
    split_clusters,
    split_with_leakage_guard,
    text_bigrams,
    verify_no_leakage,
)


def test_text_bigrams_captures_cjk_pairs_and_latin_tokens() -> None:
    signature = text_bigrams("ONU掉线")
    assert "onu" in signature
    assert "掉线" in signature


def test_jaccard_similarity_bounds() -> None:
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a"}, {"b"}) == 0.0
    assert jaccard_similarity(set(), {"a"}) == 0.0


def test_similar_texts_cluster_together() -> None:
    texts = [
        "ONU频繁掉线还报LOS告警怎么办",
        "ONU频繁掉线还报LOS告警咋办",  # 与上一条高度相似
        "帮我查询今天天气怎么样",
    ]
    assignment = assign_clusters_by_similarity(texts, threshold=0.65)
    assert assignment.cluster_of_row[0] == assignment.cluster_of_row[1]
    assert assignment.cluster_of_row[2] != assignment.cluster_of_row[0]


def test_split_clusters_stratifies_by_label() -> None:
    cluster_ids = [f"i{index}" for index in range(10)] * 2  # 10 clusters, 2 rows each
    strata = ["a"] * 10 + ["b"] * 10
    split_map = split_clusters(cluster_ids, ratios=(0.7, 0.1, 0.2), seed=1, strata=strata)
    assert set(split_map.values()) >= {"train", "test"}
    for stratum in ("a", "b"):
        stratum_clusters = [
            cluster
            for cluster, label in zip(cluster_ids, strata, strict=False)
            if label == stratum
        ]
        splits = {split_map[cluster] for cluster in stratum_clusters}
        assert "train" in splits and "test" in splits


def test_verify_no_leakage_ignores_cross_label_pairs() -> None:
    rows = [
        {"text": "ONU频繁掉线还报LOS告警怎么办", "intent": "fault_diagnosis"},
        {"text": "ONU频繁掉线还报LOS告警怎么办呀", "intent": "knowledge_qa"},
    ]
    # 跨标签近重复对是 hard negative：same_label_only=True（默认）不算泄露
    assert not verify_no_leakage(rows, ["train", "test"], threshold=0.5)
    # 关闭同标签过滤后按原样报告
    assert verify_no_leakage(rows, ["train", "test"], threshold=0.5, same_label_only=False)


def test_leakage_guard_merges_clusters_until_clean() -> None:
    rows = [
        {"text": "CRC告警的触发条件是什么", "intent": "alarm_explanation", "cluster_id": "s1"},
        {"text": "CRC告警是什么原因触发的呢", "intent": "alarm_explanation", "cluster_id": "s2"},
        {"text": "CRC告警的触发条件是啥", "intent": "alarm_explanation", "cluster_id": "s3"},
        {"text": "帮我查一下明天天气如何", "intent": "unsupported", "cluster_id": "s4"},
        {"text": "明天天气怎么样适合干活吗", "intent": "unsupported", "cluster_id": "s5"},
        {"text": "天气查询", "intent": "unsupported", "cluster_id": "s6"},
    ]
    cluster_of_row, cluster_split, leaks = split_with_leakage_guard(
        rows, threshold=0.6, stratify_by="intent"
    )

    assert leaks == []
    split_of_row = [cluster_split[cluster] for cluster in cluster_of_row]
    # 同标签的高相似对必须同侧
    assert split_of_row[0] == split_of_row[1] or split_of_row[0] == split_of_row[2]


def test_leakage_guard_collapses_fully_similar_pool_into_one_cluster() -> None:
    # 阈值极低时 12 条互为高相似的样本应合并为同一簇并整体落在同一侧
    rows = [
        {"text": f"ONU掉线了{i}号设备还闪断要处理下", "intent": "fault_diagnosis"}
        for i in range(12)
    ]
    cluster_of_row, _, leaks = split_with_leakage_guard(
        rows, threshold=0.05, ratios=(0.5, 0.25, 0.25), max_rounds=2
    )

    assert len(set(cluster_of_row)) == 1
    assert leaks == []


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="ratios"):
        split_clusters(["a", "b"], ratios=(0.5, 0.5, 0.5))
