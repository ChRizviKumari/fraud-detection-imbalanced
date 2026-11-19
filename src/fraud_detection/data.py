"""Seeded synthetic transaction generator.

IMPORTANT: all data produced here is SYNTHETIC. It encodes the author's
assumptions about how fraud differs from legitimate behaviour (night-time
bursts, unusually large amounts relative to a user's history, risky merchant
categories, country mismatch). Metrics measured on it say nothing about
performance on real payment traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH_START = 1_704_067_200  # 2024-01-01T00:00:00Z
SECONDS_PER_DAY = 86_400

CATEGORIES: tuple[str, ...] = (
    "grocery",
    "restaurant",
    "fuel",
    "retail",
    "travel",
    "electronics",
    "gift_cards",
    "entertainment",
)
# Categories the synthetic fraudster over-indexes on.
_FRAUD_CATEGORY_P = np.array([0.03, 0.04, 0.03, 0.10, 0.15, 0.35, 0.25, 0.05])

RAW_COLUMNS: tuple[str, ...] = (
    "txn_id",
    "user_id",
    "timestamp",
    "amount",
    "merchant_category",
    "is_online",
    "country_mismatch",
    "is_fraud",
)


def generate_transactions(
    n_transactions: int = 150_000,
    n_users: int = 3_000,
    n_days: int = 90,
    fraud_rate: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic, time-ordered transaction table.

    Legitimate traffic: each user has a log-normal spending profile, a
    preferred hour of day and category preferences. Fraud arrives in
    *episodes* (1-5 transactions a few minutes apart on one compromised
    user). About 30% of episodes mimic the victim's normal spending so the
    problem is not trivially separable.

    Args:
        n_transactions: Total rows (legitimate + fraud).
        n_users: Number of distinct users.
        n_days: Length of the observation window.
        fraud_rate: Target fraction of fraudulent rows (exact up to rounding).
        seed: RNG seed; the output is fully determined by the arguments.

    Returns:
        DataFrame with ``RAW_COLUMNS``, sorted by ``timestamp`` (``txn_id`` is
        the row's position after sorting). ``timestamp`` is int epoch seconds.
    """
    if not 0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be in (0, 0.5)")
    if n_transactions < 100 or n_users < 2 or n_days < 3:
        raise ValueError("n_transactions>=100, n_users>=2, n_days>=3 required")

    rng = np.random.default_rng(seed)
    n_fraud = max(1, round(n_transactions * fraud_rate))
    n_legit = n_transactions - n_fraud

    # ---- per-user profiles -------------------------------------------------
    mu = rng.normal(3.5, 0.5, n_users)  # mean log-amount (~$33)
    sigma = rng.uniform(0.4, 0.8, n_users)
    pref_hour = rng.uniform(8, 21, n_users)
    online_p = rng.beta(2, 5, n_users)
    cat_pref = rng.dirichlet(np.full(len(CATEGORIES), 1.5), n_users)
    activity = rng.gamma(2.0, 1.0, n_users)
    activity /= activity.sum()

    # ---- legitimate transactions -------------------------------------------
    users = rng.choice(n_users, size=n_legit, p=activity)
    hour = (rng.normal(pref_hour[users], 3.5) % 24).astype(float)
    uniform_mask = rng.random(n_legit) < 0.05  # occasional off-pattern hour
    hour[uniform_mask] = rng.uniform(0, 24, uniform_mask.sum())
    day = rng.integers(0, n_days, n_legit)
    ts = EPOCH_START + day * SECONDS_PER_DAY + (hour * 3600).astype(np.int64)
    log_amt = rng.normal(mu[users], sigma[users])
    big = rng.random(n_legit) < 0.01  # legit large purchases = hard negatives
    log_amt[big] += rng.normal(1.5, 0.3, big.sum())
    amount = np.exp(log_amt)
    cum = cat_pref[users].cumsum(axis=1)
    cat = (rng.random(n_legit)[:, None] > cum).sum(axis=1).clip(0, len(CATEGORIES) - 1)
    legit = pd.DataFrame(
        {
            "user_id": users,
            "timestamp": ts,
            "amount": amount,
            "merchant_category": np.array(CATEGORIES)[cat],
            "is_online": rng.random(n_legit) < online_p[users],
            "country_mismatch": rng.random(n_legit) < 0.02,
            "is_fraud": False,
        }
    )

    # ---- fraud episodes -----------------------------------------------------
    rows: list[tuple[int, int, float, str, bool, bool]] = []
    horizon = EPOCH_START + n_days * SECONDS_PER_DAY - 1
    while len(rows) < n_fraud:
        u = int(rng.integers(0, n_users))
        k = int(rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.25, 0.2, 0.12, 0.08]))
        d = int(rng.integers(0, n_days))
        h = rng.uniform(0, 5.5) if rng.random() < 0.55 else rng.uniform(0, 24)
        t = EPOCH_START + d * SECONDS_PER_DAY + int(h * 3600)
        mimic = rng.random() < 0.30
        for _ in range(k):
            if len(rows) >= n_fraud:
                break
            la = rng.normal(mu[u], sigma[u]) if mimic else mu[u] + rng.normal(2.0, 0.6)
            c = rng.choice(len(CATEGORIES), p=_FRAUD_CATEGORY_P)
            rows.append(
                (
                    u,
                    min(t, horizon),
                    float(np.exp(la)),
                    CATEGORIES[c],
                    bool(rng.random() < 0.85),
                    bool(rng.random() < 0.5),
                )
            )
            t += int(rng.exponential(240)) + 1
    fraud = pd.DataFrame(
        rows,
        columns=["user_id", "timestamp", "amount", "merchant_category", "is_online", "country_mismatch"],
    )
    fraud["is_fraud"] = True

    df = pd.concat([legit, fraud], ignore_index=True)
    df["amount"] = df["amount"].clip(lower=0.5).round(2)
    df = df.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))
    df["user_id"] = df["user_id"].astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[list(RAW_COLUMNS)]
