"""Cluster-aware dataset splitting to prevent train/test leakage.

背景：意图数据由模板/LLM 批量生成时，同一“语义簇”的样本文本高度相似。
如果随机切分，训练集和测试集会出现近重复句对，测试指标虚高（数据泄露）。

做法：
  * 离线生成器在构造样本时就记录 cluster_id（每个语义种子一个簇）；
  * LLM 生成的数据没有 cluster_id，用字符 bigram Jaccard 相似度做贪心聚类，
    相似度超过阈值的样本归入同一簇；
  * 切分以“簇”为单位分配到 train/val/test，同簇样本永远落在同一侧。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[一-鿿]|[a-z0-9]+(?:_[a-z0-9]+)*")


def text_bigrams(text: str) -> set[str]:
    """CJK character bigrams + whole latin tokens, forming the similarity signature."""
    tokens = _TOKEN_RE.findall(text.lower())
    signature: set[str] = set(tokens)
    cjk_run: list[str] = []
    for token in tokens:
        if len(token) == 1:  # CJK 单字符：累积成 run 生成相邻字 bigram
            cjk_run.append(token)
        else:
            if len(cjk_run) > 1:
                signature.update(
                    "".join(cjk_run[i : i + 2]) for i in range(len(cjk_run) - 1)
                )
            cjk_run = []
        if len(token) >= 2:
            signature.update(token[i : i + 2] for i in range(len(token) - 1))
    if len(cjk_run) > 1:
        signature.update("".join(cjk_run[i : i + 2]) for i in range(len(cjk_run) - 1))
    return signature


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


@dataclass(slots=True)
class ClusterAssignment:
    clusters: dict[str, list[int]] = field(default_factory=dict)  # cluster_id -> row indexes
    cluster_of_row: list[str] = field(default_factory=list)


def assign_clusters_by_similarity(
    texts: list[str],
    *,
    threshold: float = 0.55,
    max_cluster_representatives: int = 8,
) -> ClusterAssignment:
    """Greedy clustering: each new text joins the first similar cluster, else forms a new one.

    每个簇保留最多 ``max_cluster_representatives`` 个代表文本用于比较，
    控制复杂度约为 O(n * clusters * representatives)。
    """
    assignment = ClusterAssignment()
    representatives: dict[str, list[set[str]]] = {}
    for index, text in enumerate(texts):
        signature = text_bigrams(text)
        matched_cluster = None
        for cluster_id, reps in representatives.items():
            if any(jaccard_similarity(signature, rep) >= threshold for rep in reps):
                matched_cluster = cluster_id
                break
        if matched_cluster is None:
            matched_cluster = f"auto-{len(assignment.clusters):05d}"
            representatives[matched_cluster] = []
        assignment.clusters.setdefault(matched_cluster, []).append(index)
        assignment.cluster_of_row.append(matched_cluster)
        reps = representatives[matched_cluster]
        if len(reps) < max_cluster_representatives:
            reps.append(signature)
    return assignment


def split_clusters(
    cluster_ids: list[str],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 2025,
    strata: list[str] | None = None,
) -> dict[str, str]:
    """Map every cluster to one of train/val/test (cluster-level split, no leakage).

    ``strata`` 与 ``cluster_ids`` 平行（如每行的 intent）：给出时按层独立切分，
    保证每个意图在 train/val/test 都有覆盖（分层簇切分）。
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    if strata is None:
        groups = {"__all__": sorted(set(cluster_ids))}
    else:
        groups: dict[str, list[str]] = {}
        for cluster_id, stratum in zip(cluster_ids, strata, strict=True):
            groups.setdefault(stratum, [])
            if cluster_id not in groups[stratum]:
                groups[stratum].append(cluster_id)
    for clusters in groups.values():
        clusters = sorted(clusters)
        rng.shuffle(clusters)
        total = len(clusters)
        train_end = max(1, int(round(total * ratios[0])))
        val_end = min(
            total,
            train_end + max(1 if ratios[1] > 0 else 0, int(round(total * ratios[1]))),
        )
        for position, cluster_id in enumerate(clusters):
            if position < train_end:
                assignment[cluster_id] = "train"
            elif position < val_end:
                assignment[cluster_id] = "val"
            else:
                assignment[cluster_id] = "test"
    return assignment


def verify_no_leakage(
    rows: list[dict],
    split_of_row: list[str],
    *,
    threshold: float = 0.65,
    same_label_only: bool = True,
) -> list[dict]:
    """Cross-check: no high-similarity text pair may straddle different splits.

    ``same_label_only=True`` 时只把“同意图”的近重复对视为泄露：跨意图的相似
    句对（如“X是什么意思” vs “X怎么排查”）是天然 hard negative，保留跨侧
    反而有助于防止模型背文本。用倒排索引 + 大小剪枝避免 O(n²) 全对比较。
    """
    signatures = [text_bigrams(row["text"]) for row in rows]
    sizes = [len(signature) for signature in signatures]
    posting: dict[str, list[int]] = {}
    for index, signature in enumerate(signatures):
        for token in signature:
            posting.setdefault(token, []).append(index)

    def same_label(i: int, j: int) -> bool:
        if not same_label_only:
            return True
        return rows[i].get("intent") == rows[j].get("intent")

    checked: set[tuple[int, int]] = set()
    leaks: list[dict] = []
    for indexes in posting.values():
        if len(indexes) < 2:
            continue
        for position, i in enumerate(indexes):
            for j in indexes[position + 1 :]:
                if split_of_row[i] == split_of_row[j] or not same_label(i, j):
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in checked:
                    continue
                checked.add(pair)
                # Jaccard 上界剪枝：min/max 大小比低于阈值时不可能达到阈值
                upper = min(sizes[i], sizes[j]) / max(sizes[i], sizes[j])
                if upper < threshold:
                    continue
                similarity = jaccard_similarity(signatures[i], signatures[j])
                if similarity >= threshold:
                    leaks.append(
                        {
                            "similarity": round(similarity, 4),
                            "left": {
                                "index": pair[0],
                                "split": split_of_row[pair[0]],
                                "text": rows[pair[0]]["text"],
                            },
                            "right": {
                                "index": pair[1],
                                "split": split_of_row[pair[1]],
                                "text": rows[pair[1]]["text"],
                            },
                        }
                    )
    return leaks


def stratified_split_with_clusters(
    rows: list[dict],
    *,
    threshold: float = 0.55,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 2025,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Full pipeline: cluster (or reuse cluster_id) → split clusters → return split rows.

    若行内自带 ``cluster_id``（离线生成器写入），直接复用；否则按文本相似度聚类。
    返回 (train, val, test, diagnostics)，diagnostics 含簇数、每侧簇数与泄露检查结果。
    """
    if rows and all(row.get("cluster_id") for row in rows):
        cluster_of_row = [str(row["cluster_id"]) for row in rows]
    else:
        assignment = assign_clusters_by_similarity(
            [row["text"] for row in rows], threshold=threshold
        )
        cluster_of_row = assignment.cluster_of_row

    cluster_split = split_clusters(cluster_of_row, ratios=ratios, seed=seed)
    split_of_row = [cluster_split[cluster_id] for cluster_id in cluster_of_row]

    train = [
        dict(row, split="train")
        for row, split in zip(rows, split_of_row, strict=True)
        if split == "train"
    ]
    val = [
        dict(row, split="val")
        for row, split in zip(rows, split_of_row, strict=True)
        if split == "val"
    ]
    test = [
        dict(row, split="test")
        for row, split in zip(rows, split_of_row, strict=True)
        if split == "test"
    ]
    diagnostics = {
        "clusters_total": len(set(cluster_of_row)),
        "clusters_train": sum(1 for split in cluster_split.values() if split == "train"),
        "clusters_val": sum(1 for split in cluster_split.values() if split == "val"),
        "clusters_test": sum(1 for split in cluster_split.values() if split == "test"),
        "rows_train": len(train),
        "rows_val": len(val),
        "rows_test": len(test),
        "leaks": verify_no_leakage(rows, split_of_row, threshold=threshold),
    }
    return train, val, test, diagnostics


def split_with_leakage_guard(
    rows: list[dict],
    *,
    threshold: float = 0.65,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 2025,
    max_rounds: int = 8,
    stratify_by: str | None = "intent",
) -> tuple[list[str], dict[str, str], list[dict]]:
    """Cluster-level split that iteratively merges clusters until no leakage remains.

    单纯按“构造簇”切分挡不住不同种子之间的语义重叠（例如
    “X的触发条件是什么”与“X是什么原因触发的”）。本函数每轮：
      1. 按当前簇分层切分 train/val/test（每层独立，保证意图覆盖均衡）；
      2. 运行同标签跨侧高相似对检查（跨意图相似对是 hard negative，不算泄露）；
      3. 将泄露对两端的簇 union 合并，进入下一轮重切。
    收敛后任何同标签、相似度 >= threshold 的样本对必然落在同一侧。

    返回 (cluster_of_row, cluster_split, remaining_leaks)；
    ``remaining_leaks`` 非空表示达到 max_rounds 仍未收敛（如实上报，不静默）。
    """
    if rows and all(row.get("cluster_id") for row in rows):
        cluster_of_row = [str(row["cluster_id"]) for row in rows]
    else:
        assignment = assign_clusters_by_similarity(
            [row["text"] for row in rows], threshold=threshold
        )
        cluster_of_row = assignment.cluster_of_row
    strata = (
        [str(row.get(stratify_by, "__all__")) for row in rows] if stratify_by else None
    )

    parent: dict[str, str] = {}

    def find(cluster: str) -> str:
        root = cluster
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(cluster, cluster) != cluster:
            parent[cluster], cluster = root, parent[cluster]
        return root

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for cluster in set(cluster_of_row):
        parent.setdefault(cluster, cluster)

    remaining_leaks: list[dict] = []
    cluster_split: dict[str, str] = {}
    for round_index in range(max_rounds):
        current_clusters = [find(cluster) for cluster in cluster_of_row]
        cluster_split = split_clusters(
            current_clusters, ratios=ratios, seed=seed + round_index, strata=strata
        )
        split_of_row = [cluster_split[cluster] for cluster in current_clusters]
        remaining_leaks = verify_no_leakage(rows, split_of_row, threshold=threshold)
        if not remaining_leaks:
            return current_clusters, cluster_split, []
        for leak in remaining_leaks:
            union(
                cluster_of_row[leak["left"]["index"]],
                cluster_of_row[leak["right"]["index"]],
            )
    current_clusters = [find(cluster) for cluster in cluster_of_row]
    cluster_split = split_clusters(
        current_clusters, ratios=ratios, seed=seed + max_rounds, strata=strata
    )
    split_of_row = [cluster_split[cluster] for cluster in current_clusters]
    remaining_leaks = verify_no_leakage(rows, split_of_row, threshold=threshold)
    return current_clusters, cluster_split, remaining_leaks
