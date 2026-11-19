"""Time-based train / validation / test splitting (no random shuffling)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Chronologically ordered partitions of the transaction table."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict[str, dict[str, float]]:
        """Row counts and fraud rates per partition."""
        return {
            name: {"rows": len(part), "fraud_rate": float(part["is_fraud"].mean())}
            for name, part in (("train", self.train), ("val", self.val), ("test", self.test))
        }


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    gap_seconds: int = 0,
) -> TimeSplit:
    """Split by wall-clock time: train < validation < test.

    Cut points are placed at fractions of the *time span* (not the row count).
    An optional embargo ``gap_seconds`` drops rows near each boundary, which
    models label delay (fraud is usually confirmed days after the event).

    Raises:
        ValueError: on invalid fractions or an empty partition.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be >= 0")
    t_min, t_max = int(df["timestamp"].min()), int(df["timestamp"].max())
    span = t_max - t_min
    c1 = t_min + train_frac * span
    c2 = t_min + (train_frac + val_frac) * span
    ts = df["timestamp"]
    split = TimeSplit(
        train=df[ts < c1],
        val=df[(ts >= c1 + gap_seconds) & (ts < c2)],
        test=df[ts >= c2 + gap_seconds],
    )
    if min(len(split.train), len(split.val), len(split.test)) == 0:
        raise ValueError("empty partition; adjust fractions or gap")
    assert_no_temporal_leakage(split)
    return split


def assert_no_temporal_leakage(split: TimeSplit) -> None:
    """Raise ``AssertionError`` unless train < val < test strictly in time."""
    assert split.train["timestamp"].max() < split.val["timestamp"].min(), "train/val overlap"
    assert split.val["timestamp"].max() < split.test["timestamp"].min(), "val/test overlap"
    ids = [set(p["txn_id"]) for p in (split.train, split.val, split.test)]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2]), "shared rows"
