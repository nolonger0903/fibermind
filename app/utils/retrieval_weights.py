from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """Normalized dense/sparse weights used by every hybrid retrieval backend."""

    dense: float
    sparse: float


def normalize_fusion_weights(dense_weight: float, sparse_weight: float) -> FusionWeights:
    """Validate and normalize dense/sparse hybrid weights.

    Rules:
      * both weights must be >= 0;
      * their sum must be > 0 (otherwise fusion is undefined);
      * the returned pair always sums to 1.0.

    Raises:
        ValueError: on negative weights or an all-zero combination.
    """
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError(
            "dense_weight and sparse_weight must be >= 0, got "
            f"dense_weight={dense_weight}, sparse_weight={sparse_weight}"
        )
    total = dense_weight + sparse_weight
    if total == 0:
        raise ValueError(
            "dense_weight + sparse_weight must be > 0, otherwise hybrid fusion is undefined"
        )
    return FusionWeights(dense=dense_weight / total, sparse=sparse_weight / total)
