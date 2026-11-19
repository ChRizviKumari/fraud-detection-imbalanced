"""Cost-sensitive decision threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Monetary (or arbitrary-unit) costs of each outcome.

    Attributes:
        false_positive: Cost of wrongly flagging/declining a legitimate txn.
        false_negative: Cost of a missed fraud.
        review: Cost of sending one transaction to manual review (used by the
            three-way policy only).
    """

    false_positive: float = 10.0
    false_negative: float = 150.0
    review: float = 2.0

    def __post_init__(self) -> None:
        if min(self.false_positive, self.false_negative, self.review) < 0:
            raise ValueError("costs must be non-negative")


@dataclass(frozen=True)
class ThresholdResult:
    """Outcome of a threshold search (``flag if score >= threshold``)."""

    threshold: float
    total_cost: float
    fp: int
    fn: int
    tp: int
    tn: int


def total_cost(y: np.ndarray, scores: np.ndarray, threshold: float, costs: CostModel) -> float:
    """Total cost of flagging ``scores >= threshold``."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(scores) >= threshold
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return fp * costs.false_positive + fn * costs.false_negative


def cost_optimal_threshold(y: np.ndarray, scores: np.ndarray, costs: CostModel) -> ThresholdResult:
    """Exhaustively find the threshold minimising ``fp*C_fp + fn*C_fn``.

    Every distinct score is a candidate, plus "flag nothing". Ties between
    equal-cost thresholds resolve to the higher threshold (fewer alerts).
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    tp_c = np.cumsum(yy)
    fp_c = np.cumsum(~yy)
    # Valid cut points: after the last element of each run of equal scores.
    last_of_run = np.r_[ss[1:] != ss[:-1], True]
    tp, fp = tp_c[last_of_run], fp_c[last_of_run]
    thr = ss[last_of_run]
    pos = int(y.sum())
    fn = pos - tp
    cost = fp * costs.false_positive + fn * costs.false_negative
    # Include "flag nothing" as a candidate.
    cost_all = np.r_[pos * costs.false_negative, cost]
    best = int(np.argmin(cost_all))  # argmin picks the first (= flag nothing / higher thr)
    if best == 0:
        return ThresholdResult(float("inf"), float(cost_all[0]), 0, pos, 0, int((~y).sum()))
    j = best - 1
    return ThresholdResult(
        threshold=float(thr[j]),
        total_cost=float(cost[j]),
        fp=int(fp[j]),
        fn=int(fn[j]),
        tp=int(tp[j]),
        tn=int((~y).sum() - fp[j]),
    )
