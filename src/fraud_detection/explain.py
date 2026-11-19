"""Model-agnostic explainability via permutation importance."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop in average precision when each feature is shuffled.

    Computed on held-out data. A large positive ``importance_mean`` means the
    model relies on that feature; values near zero (or negative) mean it does
    not. Correlated features share credit, so read results as a guide, not as
    causal attribution.

    Returns:
        DataFrame sorted by descending mean importance with columns
        ``feature``, ``importance_mean``, ``importance_std``.
    """
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
