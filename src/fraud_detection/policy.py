"""Score -> decision policy (approve / review / block)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .threshold import CostModel

Decision = Literal["approve", "review", "block"]


@dataclass(frozen=True)
class DecisionPolicy:
    """Two thresholds splitting the score range into three actions.

    ``score < review_threshold`` -> approve; ``review_threshold <= score <
    block_threshold`` -> manual review; ``score >= block_threshold`` -> block.
    """

    review_threshold: float
    block_threshold: float

    def __post_init__(self) -> None:
        if self.review_threshold > self.block_threshold:
            raise ValueError("review_threshold must be <= block_threshold")

    def decide(self, score: float) -> Decision:
        """Map one score to an action."""
        if score >= self.block_threshold:
            return "block"
        if score >= self.review_threshold:
            return "review"
        return "approve"

    def decide_many(self, scores: np.ndarray) -> np.ndarray:
        """Vectorised ``decide`` returning an array of strings."""
        s = np.asarray(scores)
        out = np.full(s.shape, "approve", dtype=object)
        out[s >= self.review_threshold] = "review"
        out[s >= self.block_threshold] = "block"
        return out

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def policy_cost(y: np.ndarray, scores: np.ndarray, policy: DecisionPolicy, costs: CostModel) -> float:
    """Expected total cost of a policy on labelled data.

    Assumptions (explicit simplifications): every reviewed transaction costs
    ``costs.review`` and the reviewer is *perfect* (fraud reviewed = caught,
    legit reviewed = approved); blocked legit pays ``false_positive``; blocked
    fraud is free; approved fraud pays ``false_negative``.
    """
    y = np.asarray(y).astype(bool)
    d = policy.decide_many(scores)
    approve, review, block = d == "approve", d == "review", d == "block"
    return float(
        (approve & y).sum() * costs.false_negative
        + review.sum() * costs.review
        + (block & ~y).sum() * costs.false_positive
    )


def fit_policy(
    y: np.ndarray, scores: np.ndarray, costs: CostModel, n_candidates: int = 200
) -> DecisionPolicy:
    """Pick (review, block) thresholds minimising policy cost on labelled data.

    Searches all pairs from ``n_candidates`` score quantiles (plus ``inf`` for
    "never"), vectorised with prefix counts over the sorted scores.
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="stable")
    ss, yy = s[order], y[order]
    cand = np.unique(np.quantile(ss, np.linspace(0, 1, n_candidates)))
    cand = np.r_[cand, np.inf]
    idx = np.searchsorted(ss, cand, side="left")  # rows with score < cand
    fraud_below = np.r_[0, np.cumsum(yy)][idx]
    n = len(ss)
    fraud_total = int(yy.sum())
    best = (np.inf, cand[0], cand[0])
    for a, (ia, fa) in enumerate(zip(idx, fraud_below, strict=True)):  # review threshold
        approve_fraud = fa
        ib, fb = idx[a:], fraud_below[a:]  # block thresholds >= review threshold
        n_review = ib - ia
        n_block = n - ib
        fraud_block = fraud_total - fb
        legit_block = n_block - fraud_block
        cost = (
            approve_fraud * costs.false_negative
            + n_review * costs.review
            + legit_block * costs.false_positive
        )
        j = int(np.argmin(cost))
        if cost[j] < best[0]:
            best = (float(cost[j]), float(cand[a]), float(cand[a + j]))
    return DecisionPolicy(review_threshold=best[1], block_threshold=best[2])
