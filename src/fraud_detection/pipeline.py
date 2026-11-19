"""End-to-end training pipeline and the serialisable model artifact."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data import generate_transactions
from .evaluation import evaluate_scores
from .features import FEATURE_COLUMNS, build_features
from .models import make_models, positive_scores
from .policy import DecisionPolicy, fit_policy, policy_cost
from .split import TimeSplit, time_based_split
from .threshold import CostModel, ThresholdResult, cost_optimal_threshold, total_cost


@dataclass
class ModelArtifact:
    """Everything the scoring service needs, saved/loaded with pickle."""

    model: Any
    model_name: str
    feature_columns: tuple[str, ...]
    policy: DecisionPolicy
    binary_threshold: float
    costs: CostModel
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Serialise with pickle (only load artifacts you trust)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> ModelArtifact:
        with open(path, "rb") as fh:
            return pickle.load(fh)


@dataclass
class PipelineResult:
    """Everything produced by :func:`run_pipeline` (for reporting and tests)."""

    artifact: ModelArtifact
    split: TimeSplit
    features: dict[str, pd.DataFrame]
    val_metrics: dict[str, dict[str, float]]
    test_metrics: dict[str, dict[str, float]]
    binary_threshold: ThresholdResult
    test_costs: dict[str, float]


def run_pipeline(
    n_transactions: int = 150_000,
    n_users: int = 3_000,
    seed: int = 42,
    costs: CostModel | None = None,
    gap_seconds: int = 0,
) -> PipelineResult:
    """Generate data, engineer features, split by time, train, tune, evaluate.

    Model selection uses validation average precision; thresholds and the
    approve/review/block policy are fitted on *validation* scores only, then
    every number reported for the test partition is a pure hold-out.
    """
    costs = costs or CostModel()
    df = generate_transactions(n_transactions=n_transactions, n_users=n_users, seed=seed)
    feats = build_features(df)  # causal => safe to compute before splitting
    split = time_based_split(df, gap_seconds=gap_seconds)
    X = {name: feats.loc[part.index] for name, part in zip(("train", "val", "test"), (split.train, split.val, split.test), strict=True)}
    y = {"train": split.train["is_fraud"], "val": split.val["is_fraud"], "test": split.test["is_fraud"]}

    models = make_models(seed)
    val_metrics: dict[str, dict[str, float]] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X["train"], y["train"])
        val_metrics[name] = evaluate_scores(y["val"], positive_scores(model, X["val"]))
        test_metrics[name] = evaluate_scores(y["test"], positive_scores(model, X["test"]))

    best = max(val_metrics, key=lambda n: val_metrics[n]["average_precision"])
    model = models[best]
    val_scores = positive_scores(model, X["val"])
    test_scores = positive_scores(model, X["test"])
    thr = cost_optimal_threshold(y["val"].to_numpy(), val_scores, costs)
    policy = fit_policy(y["val"].to_numpy(), val_scores, costs)

    yt = y["test"].to_numpy()
    test_costs = {
        "approve_everything": float(yt.sum() * costs.false_negative),
        "threshold_0.5": total_cost(yt, test_scores, 0.5, costs),
        "cost_optimal_threshold": total_cost(yt, test_scores, thr.threshold, costs),
        "three_way_policy": policy_cost(yt, test_scores, policy, costs),
    }
    artifact = ModelArtifact(
        model=model,
        model_name=best,
        feature_columns=FEATURE_COLUMNS,
        policy=policy,
        binary_threshold=thr.threshold,
        costs=costs,
        metadata={"seed": seed, "synthetic_data": True, "n_transactions": n_transactions},
    )
    return PipelineResult(artifact, split, X, val_metrics, test_metrics, thr, test_costs)
