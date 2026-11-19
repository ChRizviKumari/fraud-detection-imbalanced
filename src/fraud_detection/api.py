"""Flask scoring service: raw transaction in, score + decision out.

The service keeps an in-memory per-user history (``HistoryStore``) so that
velocity and z-score features are computed with *exactly* the same code as in
training. In production this store would be a low-latency feature store
(e.g. Redis/DynamoDB) rather than process memory.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError, field_validator

from .data import CATEGORIES
from .features import build_features
from .models import positive_scores
from .pipeline import ModelArtifact

_RAW = ("user_id", "timestamp", "amount", "merchant_category", "is_online", "country_mismatch")
Category = Literal[CATEGORIES]  # type: ignore[valid-type]


class TransactionRequest(BaseModel):
    """Validated request body for ``POST /score``."""

    user_id: int = Field(ge=0)
    timestamp: datetime
    amount: float = Field(gt=0, lt=1e7)
    merchant_category: Category
    is_online: bool
    country_mismatch: bool = False
    record: bool = True  # append to the user's history after scoring

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v

    def epoch(self) -> int:
        return int(self.timestamp.timestamp())


class HistoryStore:
    """Thread-safe in-memory transaction history keyed by user."""

    def __init__(self) -> None:
        self._rows: dict[int, list[tuple[Any, ...]]] = {}
        self._lock = threading.Lock()

    def seed(self, df: pd.DataFrame) -> None:
        """Bulk-load historical raw transactions."""
        with self._lock:
            for row in df[list(_RAW)].itertuples(index=False, name=None):
                self._rows.setdefault(int(row[0]), []).append(row)

    def frame(self, user_id: int) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows.get(user_id, []))
        return pd.DataFrame(rows, columns=list(_RAW))

    def add(self, row: tuple[Any, ...]) -> None:
        with self._lock:
            self._rows.setdefault(int(row[0]), []).append(row)


def score_transaction(
    artifact: ModelArtifact, store: HistoryStore, txn: TransactionRequest
) -> dict[str, Any]:
    """Compute features from history + this transaction, score and decide."""
    row = (
        txn.user_id,
        txn.epoch(),
        txn.amount,
        txn.merchant_category,
        txn.is_online,
        txn.country_mismatch,
    )
    hist = store.frame(txn.user_id)
    frame = pd.concat([hist, pd.DataFrame([row], columns=list(_RAW))], ignore_index=True)
    feats = build_features(frame).iloc[[-1]][list(artifact.feature_columns)]
    score = float(positive_scores(artifact.model, feats)[0])
    if txn.record:
        store.add(row)
    return {
        "score": score,
        "decision": artifact.policy.decide(score),
        "model": artifact.model_name,
        "thresholds": artifact.policy.to_dict(),
        "features": {k: float(v) for k, v in feats.iloc[0].items() if not k.startswith("cat_")},
    }


def create_app(artifact: ModelArtifact, history: pd.DataFrame | None = None) -> Flask:
    """Application factory.

    Args:
        artifact: Trained model + policy.
        history: Optional raw transactions used to pre-populate user history.
    """
    app = Flask(__name__)
    store = HistoryStore()
    if history is not None:
        store.seed(history)

    @app.get("/health")
    def health() -> Any:
        return jsonify(status="ok", model=artifact.model_name, synthetic_data=True)

    @app.get("/policy")
    def policy() -> Any:
        return jsonify(
            thresholds=artifact.policy.to_dict(),
            costs=vars(artifact.costs),
            model=artifact.model_name,
        )

    @app.post("/score")
    def score() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="body must be a JSON object"), 400
        try:
            txn = TransactionRequest.model_validate(payload)
        except ValidationError as exc:
            return jsonify(error="invalid request", details=json.loads(exc.json(include_url=False, include_context=False))), 400
        return jsonify(score_transaction(artifact, store, txn))

    return app
