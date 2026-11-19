"""Precision-recall driven evaluation (accuracy is meaningless at 1% prevalence)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def precision_at_recall(y: np.ndarray, scores: np.ndarray, min_recall: float) -> float:
    """Best precision achievable while keeping recall >= ``min_recall``."""
    p, r, _ = precision_recall_curve(y, scores)
    ok = r >= min_recall
    return float(p[ok].max()) if ok.any() else 0.0


def recall_at_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Best recall achievable while keeping precision >= ``min_precision``."""
    p, r, _ = precision_recall_curve(y, scores)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def top_fraction_stats(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    """Precision / recall when only the top ``fraction`` of scores is flagged."""
    k = max(1, int(round(len(y) * fraction)))
    top = np.argsort(-scores, kind="stable")[:k]
    tp = float(np.asarray(y)[top].sum())
    return {"precision": tp / k, "recall": tp / max(1.0, float(np.sum(y)))}


def evaluate_scores(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Headline metrics for one scored set.

    Returns average precision (PR-AUC), ROC-AUC, the prevalence (= AP of a
    random ranker), precision at 80% recall, recall at 50% precision and
    precision/recall when flagging the top 1% of transactions.
    """
    y = np.asarray(y).astype(int)
    top = top_fraction_stats(y, scores, 0.01)
    return {
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision_at_80_recall": precision_at_recall(y, scores, 0.80),
        "recall_at_50_precision": recall_at_precision(y, scores, 0.50),
        "precision_top1pct": top["precision"],
        "recall_top1pct": top["recall"],
    }


def confusion_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Confusion counts when flagging ``scores >= threshold``."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(scores) >= threshold
    return {
        "tp": int((pred & y).sum()),
        "fp": int((pred & ~y).sum()),
        "fn": int((~pred & y).sum()),
        "tn": int((~pred & ~y).sum()),
    }
