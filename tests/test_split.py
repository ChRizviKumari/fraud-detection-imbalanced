import pandas as pd
import pytest

from fraud_detection.split import TimeSplit, assert_no_temporal_leakage, time_based_split


def test_split_is_strictly_chronological(small_df):
    s = time_based_split(small_df)
    assert s.train["timestamp"].max() < s.val["timestamp"].min()
    assert s.val["timestamp"].max() < s.test["timestamp"].min()


def test_split_partitions_all_rows_without_gap(small_df):
    s = time_based_split(small_df)
    assert len(s.train) + len(s.val) + len(s.test) == len(small_df)


def test_gap_drops_boundary_rows(small_df):
    a = time_based_split(small_df)
    b = time_based_split(small_df, gap_seconds=2 * 86_400)
    assert len(b.val) + len(b.test) < len(a.val) + len(a.test)
    assert b.val["timestamp"].min() - b.train["timestamp"].max() >= 2 * 86_400 - 1


def test_leakage_check_catches_a_random_split(small_df):
    """A shuffled split must be rejected by the leakage assertion."""
    shuffled = small_df.sample(frac=1.0, random_state=1)
    n = len(shuffled)
    bad = TimeSplit(shuffled.iloc[: n // 2], shuffled.iloc[n // 2 : 3 * n // 4], shuffled.iloc[3 * n // 4 :])
    with pytest.raises(AssertionError):
        assert_no_temporal_leakage(bad)


def test_leakage_check_catches_shared_rows(small_df):
    s = time_based_split(small_df)
    bad = TimeSplit(s.train, pd.concat([s.val, s.train.tail(1)]), s.test)
    with pytest.raises(AssertionError):
        assert_no_temporal_leakage(bad)


@pytest.mark.parametrize("kw", [{"train_frac": 0.9, "val_frac": 0.2}, {"train_frac": 0}, {"gap_seconds": -1}])
def test_invalid_split_arguments(small_df, kw):
    with pytest.raises(ValueError):
        time_based_split(small_df, **kw)
