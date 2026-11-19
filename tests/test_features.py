import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import generate_transactions
from fraud_detection.features import FEATURE_COLUMNS, build_features


def _txns(ts, amounts, user=1):
    n = len(ts)
    return pd.DataFrame(
        {
            "user_id": [user] * n,
            "timestamp": ts,
            "amount": amounts,
            "merchant_category": ["retail"] * n,
            "is_online": [False] * n,
            "country_mismatch": [False] * n,
        }
    )


def test_columns_and_no_nans(small_df):
    f = build_features(small_df)
    assert list(f.columns) == list(FEATURE_COLUMNS)
    assert len(f) == len(small_df)
    assert np.isfinite(f.to_numpy()).all()


def test_velocity_counts_only_prior_txns_in_window():
    t0 = 1_704_067_200
    df = _txns([t0, t0 + 60, t0 + 120, t0 + 700, t0 + 4_000], [10] * 5)
    f = build_features(df)
    assert f["txn_count_10m"].tolist() == [0, 1, 2, 1, 0]  # 60s txn aged out of the 10m window at +700
    assert f["txn_count_1h"].tolist() == [0, 1, 2, 3, 1]
    assert f["txn_count_24h"].tolist() == [0, 1, 2, 3, 4]


def test_amount_zscore_flags_outlier_and_is_zero_without_history():
    t0 = 1_704_067_200
    df = _txns([t0 + i * 1000 for i in range(8)], [20, 22, 19, 21, 20, 23, 21, 900])
    z = build_features(df)["amount_z_user"]
    assert (z.iloc[:3] == 0).all()  # not enough history yet
    assert z.iloc[-1] > 5
    assert abs(z.iloc[4]) < 3


def test_zscore_is_per_user():
    t0 = 1_704_067_200
    a = _txns([t0 + i * 100 for i in range(6)], [10] * 5 + [10], user=1)
    b = _txns([t0 + i * 100 for i in range(6)], [1000] * 5 + [1000], user=2)
    f = build_features(pd.concat([a, b], ignore_index=True))
    assert f["amount_z_user"].abs().max() < 1  # 1000 is normal for user 2


def test_time_of_day_features():
    day = 1_704_067_200
    df = _txns([day + 3 * 3600, day + 12 * 3600], [5, 5])
    f = build_features(df)
    assert f["is_night"].tolist() == [1.0, 0.0]
    assert f["hour_cos"].iloc[1] == pytest.approx(-1.0)


def test_features_do_not_use_the_future(small_df):
    """Truncating all rows after a cut must not change earlier rows' features."""
    cut = small_df["timestamp"].quantile(0.5)
    full = build_features(small_df)
    past = small_df[small_df["timestamp"] < cut]
    trunc = build_features(past)
    pd.testing.assert_frame_equal(full.loc[past.index], trunc)


def test_features_independent_of_row_order(small_df):
    shuffled = small_df.sample(frac=1.0, random_state=0)
    a = build_features(small_df)
    b = build_features(shuffled).loc[small_df.index]
    pd.testing.assert_frame_equal(a, b)
