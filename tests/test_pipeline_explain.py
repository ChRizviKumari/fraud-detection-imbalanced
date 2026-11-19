import numpy as np
import pytest

from fraud_detection.explain import permutation_importance_table
from fraud_detection.models import make_logreg, positive_scores
from fraud_detection.pipeline import ModelArtifact


def test_pipeline_has_no_temporal_leakage(small_result):
    s = small_result.split
    assert s.train["timestamp"].max() < s.val["timestamp"].min()
    assert s.val["timestamp"].max() < s.test["timestamp"].min()


def test_models_beat_rule_baseline_and_prevalence(small_result):
    m = small_result.test_metrics
    prev = m["logreg_balanced"]["prevalence"]
    assert m["logreg_balanced"]["average_precision"] > 5 * prev
    assert m["hgb_balanced"]["average_precision"] > m["rule_amount_z"]["average_precision"]


def test_selected_model_has_best_validation_ap(small_result):
    v = small_result.val_metrics
    best = max(v, key=lambda k: v[k]["average_precision"])
    assert small_result.artifact.model_name == best


def test_cost_strategies_beat_approving_everything(small_result):
    c = small_result.test_costs
    assert c["cost_optimal_threshold"] < c["approve_everything"]
    assert c["three_way_policy"] < c["approve_everything"]


def test_pipeline_is_reproducible():
    from fraud_detection.pipeline import run_pipeline

    a = run_pipeline(n_transactions=8_000, n_users=200, seed=5)
    b = run_pipeline(n_transactions=8_000, n_users=200, seed=5)
    assert a.test_metrics == b.test_metrics
    assert a.artifact.binary_threshold == b.artifact.binary_threshold


def test_permutation_importance_finds_planted_signal(small_result):
    r = small_result
    tbl = permutation_importance_table(
        r.artifact.model, r.features["test"], r.split.test["is_fraud"], n_repeats=2, seed=0
    )
    assert set(tbl.columns) == {"feature", "importance_mean", "importance_std"}
    assert tbl["importance_mean"].is_monotonic_decreasing
    top = set(tbl["feature"].head(5))
    assert top & {"country_mismatch", "is_online", "log_secs_since_prev", "amount_z_user", "is_night"}


def test_logreg_uses_balanced_class_weight():
    assert make_logreg().named_steps["clf"].class_weight == "balanced"


def test_artifact_roundtrip(small_result, tmp_path):
    p = tmp_path / "m.pkl"
    small_result.artifact.save(p)
    loaded = ModelArtifact.load(p)
    X = small_result.features["test"].head(50)
    np.testing.assert_allclose(
        positive_scores(loaded.model, X), positive_scores(small_result.artifact.model, X)
    )
    assert loaded.metadata["synthetic_data"] is True
