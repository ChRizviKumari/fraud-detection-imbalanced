"""Leakage-safe feature engineering.

Every feature for a transaction is computed from that transaction and the
*earlier* transactions of the same user only. Consequently, features can be
computed on the full table before a time-based split without leaking the
future (and a test verifies that truncating the future does not change any
feature). The same function is reused by the online scoring service, which
guarantees train/serve parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORIES

VELOCITY_WINDOWS: dict[str, int] = {"10m": 600, "1h": 3_600, "24h": 86_400}
MIN_HISTORY_FOR_Z = 3
STD_FLOOR = 0.25
NO_HISTORY_GAP_SECONDS = 30 * 86_400

NUMERIC_FEATURES: tuple[str, ...] = (
    "log_amount",
    "amount_z_user",
    "log_history_len",
    "log_secs_since_prev",
    "txn_count_10m",
    "txn_count_1h",
    "txn_count_24h",
    "log_amount_sum_24h",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_online",
    "country_mismatch",
)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + tuple(f"cat_{c}" for c in CATEGORIES)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features for every row of ``df``.

    Args:
        df: Raw transactions with ``user_id``, ``timestamp`` (epoch seconds),
            ``amount``, ``merchant_category``, ``is_online``,
            ``country_mismatch``. Row order is arbitrary; ties in timestamp are
            broken by row position.

    Returns:
        DataFrame indexed like ``df`` with exactly ``FEATURE_COLUMNS``.
    """
    n = len(df)
    user = df["user_id"].to_numpy()
    ts = df["timestamp"].to_numpy().astype(np.int64)
    amount = df["amount"].to_numpy().astype(float)
    log_amt = np.log1p(amount)

    out = {name: np.zeros(n) for name in NUMERIC_FEATURES}
    order = np.lexsort((np.arange(n), ts, user))
    su, st, sl, sa = user[order], ts[order], log_amt[order], amount[order]
    bounds = np.flatnonzero(np.r_[True, su[1:] != su[:-1], True])

    sorted_out = {name: np.zeros(n) for name in out}
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        t, la, a = st[lo:hi], sl[lo:hi], sa[lo:hi]
        m = hi - lo
        i = np.arange(m)

        # Velocity: number / amount of *prior* txns within each window.
        a_cum = np.r_[0.0, np.cumsum(a)]
        for label, w in VELOCITY_WINDOWS.items():
            first = np.searchsorted(t, t - w, side="left")
            sorted_out[f"txn_count_{label}"][lo:hi] = i - first
            if label == "24h":
                sorted_out["log_amount_sum_24h"][lo:hi] = np.log1p(a_cum[i] - a_cum[first])

        # Causal per-user z-score of log-amount (expanding mean/std).
        c1 = np.r_[0.0, np.cumsum(la)][:-1]
        c2 = np.r_[0.0, np.cumsum(la**2)][:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = c1 / i
            var = np.maximum(c2 / i - mean**2, 0.0)
        std = np.maximum(np.sqrt(var), STD_FLOOR)
        z = np.where(i >= MIN_HISTORY_FOR_Z, (la - np.nan_to_num(mean)) / std, 0.0)
        sorted_out["amount_z_user"][lo:hi] = z

        gap = np.r_[NO_HISTORY_GAP_SECONDS, np.diff(t)]
        sorted_out["log_secs_since_prev"][lo:hi] = np.log1p(gap)
        sorted_out["log_history_len"][lo:hi] = np.log1p(i)

    for name, arr in sorted_out.items():
        out[name][order] = arr
    out["log_amount"] = log_amt

    hour = (ts % 86_400) / 3_600.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = (hour < 6).astype(float)
    out["is_online"] = df["is_online"].to_numpy().astype(float)
    out["country_mismatch"] = df["country_mismatch"].to_numpy().astype(float)

    feats = pd.DataFrame(out, index=df.index)
    cats = df["merchant_category"].to_numpy()
    for c in CATEGORIES:
        feats[f"cat_{c}"] = (cats == c).astype(float)
    return feats[list(FEATURE_COLUMNS)]
