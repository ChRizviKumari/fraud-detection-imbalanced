import pytest

from fraud_detection.data import generate_transactions
from fraud_detection.pipeline import run_pipeline


@pytest.fixture(scope="session")
def small_df():
    return generate_transactions(n_transactions=6_000, n_users=150, n_days=30, seed=7)


@pytest.fixture(scope="session")
def small_result():
    return run_pipeline(n_transactions=20_000, n_users=500, seed=3)
