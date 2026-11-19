import pandas as pd
import pytest

from fraud_detection.api import create_app
from fraud_detection.features import build_features
from fraud_detection.models import positive_scores

GOOD = {
    "user_id": 1,
    "timestamp": "2024-02-01T12:00:00Z",
    "amount": 25.0,
    "merchant_category": "grocery",
    "is_online": False,
}


@pytest.fixture()
def client(small_result):
    app = create_app(small_result.artifact, history=small_result.split.train)
    return app.test_client()


def test_health_declares_synthetic(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["synthetic_data"] is True


def test_policy_endpoint(client):
    j = client.get("/policy").get_json()
    assert j["thresholds"]["review_threshold"] <= j["thresholds"]["block_threshold"]


def test_score_returns_valid_decision(client):
    r = client.post("/score", json=GOOD)
    j = r.get_json()
    assert r.status_code == 200
    assert 0 <= j["score"] <= 1 and j["decision"] in {"approve", "review", "block"}


def test_extreme_risky_txn_scores_higher_than_normal(client):
    risky = dict(GOOD, user_id=2, timestamp="2024-02-01T03:00:00Z", amount=5_000, merchant_category="gift_cards",
                 is_online=True, country_mismatch=True, record=False)
    normal = dict(GOOD, user_id=2, record=False)
    assert client.post("/score", json=risky).get_json()["score"] > client.post("/score", json=normal).get_json()["score"]


@pytest.mark.parametrize(
    "patch",
    [{"amount": -5}, {"amount": 0}, {"merchant_category": "casino"}, {"user_id": -1}, {"timestamp": "not-a-date"}],
)
def test_invalid_payloads_return_400(client, patch):
    r = client.post("/score", json=dict(GOOD, **patch))
    assert r.status_code == 400 and r.get_json()["error"] == "invalid request"


def test_non_json_body_returns_400(client):
    assert client.post("/score", data="nope", content_type="text/plain").status_code == 400


def test_velocity_increases_with_recorded_bursts(small_result):
    app = create_app(small_result.artifact)  # empty history
    c = app.test_client()
    counts = []
    for i in range(4):
        body = dict(GOOD, user_id=9, timestamp=f"2024-02-01T12:0{i}:00Z")
        counts.append(c.post("/score", json=body).get_json()["features"]["txn_count_10m"])
    assert counts == [0.0, 1.0, 2.0, 3.0]


def test_record_false_does_not_mutate_history(small_result):
    c = create_app(small_result.artifact).test_client()
    for _ in range(3):
        c.post("/score", json=dict(GOOD, user_id=5, record=False))
    assert c.post("/score", json=dict(GOOD, user_id=5, record=False)).get_json()["features"]["txn_count_10m"] == 0.0


def test_online_features_match_offline_features(small_result):
    """Train/serve parity: API-computed features equal batch-computed ones."""
    r = small_result
    test = r.split.test
    row = test.iloc[200]
    c = create_app(r.artifact, history=r.split.train).test_client()
    # replay this user's earlier test transactions, then score the target
    prior = test[(test.user_id == row.user_id) & (test.timestamp < row.timestamp)]
    for t in prior.itertuples():
        c.post("/score", json={"user_id": int(t.user_id), "timestamp": int(t.timestamp), "amount": float(t.amount),
                               "merchant_category": t.merchant_category, "is_online": bool(t.is_online),
                               "country_mismatch": bool(t.country_mismatch)})
    resp = c.post("/score", json={"user_id": int(row.user_id), "timestamp": int(row.timestamp), "amount": float(row.amount),
                                  "merchant_category": row.merchant_category, "is_online": bool(row.is_online),
                                  "country_mismatch": bool(row.country_mismatch)}).get_json()
    offline = build_features(pd.concat([r.split.train, test[test.timestamp <= row.timestamp]]))
    expected = positive_scores(r.artifact.model, offline.loc[[row.name]])[0]
    assert resp["score"] == pytest.approx(expected, rel=1e-6, abs=1e-9)
