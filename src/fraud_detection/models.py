"""Model factories and a simple rule baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(seed: int = 0) -> Pipeline:
    """Standardised logistic regression with ``class_weight='balanced'``."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ]
    )


def make_hgb(seed: int = 0, balanced: bool = False) -> HistGradientBoostingClassifier:
    """Histogram gradient boosting; optionally with balanced class weights.

    Early stopping is disabled because sklearn's internal validation split is
    random (not chronological) and would reintroduce temporal leakage.
    """
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        class_weight="balanced" if balanced else None,
        random_state=seed,
    )


class AmountZScoreBaseline:
    """Naive rule: the larger the per-user amount z-score, the riskier.

    Exposes ``fit`` / ``predict_proba`` so it can be evaluated like a model.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> AmountZScoreBaseline:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-X["amount_z_user"].to_numpy()))
        return np.column_stack([1 - s, s])


def make_models(seed: int = 0) -> dict[str, Any]:
    """All candidate models keyed by name (the rule baseline is first)."""
    return {
        "rule_amount_z": AmountZScoreBaseline(),
        "logreg_balanced": make_logreg(seed),
        "hgb_plain": make_hgb(seed, balanced=False),
        "hgb_balanced": make_hgb(seed, balanced=True),
    }


def positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(fraud) (or a monotone risk score) for each row."""
    return np.asarray(model.predict_proba(X)[:, 1])
