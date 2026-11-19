import numpy as np
import pytest

from fraud_detection.evaluation import confusion_at_threshold, evaluate_scores, precision_at_recall
from fraud_detection.policy import DecisionPolicy, fit_policy, policy_cost
from fraud_detection.threshold import CostModel, cost_optimal_threshold, total_cost


def _toy(seed=0, n=4_000):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.05
    s = np.clip(rng.normal(np.where(y, 0.7, 0.25), 0.2), 0, 1)
    return y, s


def test_optimal_threshold_matches_brute_force():
    y, s = _toy()
    costs = CostModel(false_positive=5, false_negative=60)
    best = cost_optimal_threshold(y, s, costs)
    grid = np.unique(s)
    brute = min(min(total_cost(y, s, t, costs) for t in grid), total_cost(y, s, np.inf, costs))
    assert best.total_cost == pytest.approx(brute)
    assert total_cost(y, s, best.threshold, costs) == pytest.approx(best.total_cost)
    assert best.tp + best.fn == y.sum() and best.fp + best.tn == (~y).sum()


def test_higher_fn_cost_lowers_threshold():
    y, s = _toy()
    cheap = cost_optimal_threshold(y, s, CostModel(10, 20))
    dear = cost_optimal_threshold(y, s, CostModel(10, 500))
    assert dear.threshold < cheap.threshold


def test_never_flags_legit_when_false_positives_are_prohibitive():
    y, s = _toy()
    r = cost_optimal_threshold(y, s, CostModel(false_positive=1_000, false_negative=1))
    assert r.fp == 0 and r.total_cost <= y.sum() * 1


def test_flag_nothing_when_every_score_is_uninformative():
    y = np.array([True] + [False] * 99)
    r = cost_optimal_threshold(y, np.full(100, 0.5), CostModel(false_positive=10, false_negative=50))
    assert r.threshold == float("inf") and r.tp == 0 and r.total_cost == 50


def test_negative_costs_rejected():
    with pytest.raises(ValueError):
        CostModel(false_positive=-1)


def test_policy_decisions_and_validation():
    p = DecisionPolicy(0.2, 0.8)
    assert [p.decide(x) for x in (0.1, 0.2, 0.5, 0.8, 0.99)] == ["approve", "review", "review", "block", "block"]
    assert p.decide_many(np.array([0.1, 0.5, 0.9])).tolist() == ["approve", "review", "block"]
    with pytest.raises(ValueError):
        DecisionPolicy(0.9, 0.1)


def test_fitted_policy_not_worse_than_trivial_strategies():
    y, s = _toy()
    costs = CostModel()
    pol = fit_policy(y, s, costs)
    assert pol.review_threshold <= pol.block_threshold
    fitted = policy_cost(y, s, pol, costs)
    assert fitted <= y.sum() * costs.false_negative  # approve-everything
    assert fitted <= cost_optimal_threshold(y, s, costs).total_cost + 1e-9  # review=0-cost case is superset


def test_pr_metrics_on_perfect_and_random_scores():
    rng = np.random.default_rng(0)
    y = rng.random(5_000) < 0.01
    perfect = evaluate_scores(y, y.astype(float))
    assert perfect["average_precision"] == pytest.approx(1.0)
    rand = evaluate_scores(y, rng.random(5_000))
    assert rand["average_precision"] < 0.05  # ~ prevalence, while a majority-class
    assert precision_at_recall(y, y.astype(float), 0.8) == pytest.approx(1.0)


def test_accuracy_is_misleading_but_pr_auc_is_not():
    y = np.zeros(1_000, dtype=int)
    y[:10] = 1
    scores = np.zeros(1_000)  # predicts "no fraud" for everyone
    acc = 1 - y.mean()
    assert acc == 0.99
    assert evaluate_scores(y, scores)["average_precision"] == pytest.approx(0.01)


def test_confusion_counts_sum_to_n():
    y, s = _toy()
    c = confusion_at_threshold(y, s, 0.5)
    assert sum(c.values()) == len(y)
