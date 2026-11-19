import pandas as pd
import pytest

from fraud_detection.data import CATEGORIES, RAW_COLUMNS, generate_transactions


def test_generator_is_deterministic_for_a_seed():
    a = generate_transactions(2_000, 80, 20, seed=1)
    b = generate_transactions(2_000, 80, 20, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_transactions(2_000, 80, 20, seed=1)
    b = generate_transactions(2_000, 80, 20, seed=2)
    assert not a["amount"].equals(b["amount"])


def test_fraud_rate_is_about_one_percent(small_df):
    assert len(small_df) == 6_000
    assert small_df["is_fraud"].mean() == pytest.approx(0.01, abs=0.001)


def test_schema_sorted_and_valid(small_df):
    assert tuple(small_df.columns) == RAW_COLUMNS
    assert small_df["timestamp"].is_monotonic_increasing
    assert (small_df["txn_id"] == range(len(small_df))).all()
    assert (small_df["amount"] > 0).all()
    assert set(small_df["merchant_category"]) <= set(CATEGORIES)
    assert not small_df.isna().any().any()


@pytest.mark.parametrize("kwargs", [{"fraud_rate": 0.0}, {"fraud_rate": 0.6}, {"n_transactions": 10}])
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(**kwargs)


def test_fraud_differs_from_legit_in_synthetic_design(small_df):
    f, l = small_df[small_df.is_fraud], small_df[~small_df.is_fraud]
    assert f["country_mismatch"].mean() > 5 * l["country_mismatch"].mean()
