"""End-to-end demo: generate synthetic data, train, evaluate, write reports/.

Usage:  python scripts/run_demo.py [--n 150000] [--seed 42] [--out reports]

All data is SYNTHETIC; the numbers show the pipeline works, not real-world
fraud performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_detection.evaluation import confusion_at_threshold  # noqa: E402
from fraud_detection.explain import permutation_importance_table  # noqa: E402
from fraud_detection.models import positive_scores  # noqa: E402
from fraud_detection.pipeline import run_pipeline  # noqa: E402
from fraud_detection.threshold import CostModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150_000, help="number of transactions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    costs = CostModel()

    res = run_pipeline(n_transactions=args.n, seed=args.seed, costs=costs)
    art = res.artifact
    print("NOTE: all data is SYNTHETIC; results demonstrate the pipeline, not real-world performance.\n")
    print("Split summary:", json.dumps(res.split.summary(), indent=2))

    cols = ["average_precision", "roc_auc", "precision_at_80_recall", "recall_at_50_precision",
            "precision_top1pct", "recall_top1pct"]
    for title, m in (("VALIDATION", res.val_metrics), ("TEST (hold-out)", res.test_metrics)):
        tbl = pd.DataFrame(m).T[["prevalence"] + cols]
        print(f"\n{title} metrics:\n{tbl.round(4).to_string()}")
    print(f"\nSelected model (best validation PR-AUC): {art.model_name}")

    Xt, yt = res.features["test"], res.split.test["is_fraud"].to_numpy()
    scores = positive_scores(art.model, Xt)
    thr = res.binary_threshold
    print(f"Cost model: FP={costs.false_positive}, FN={costs.false_negative}, review={costs.review}")
    print(f"Cost-optimal binary threshold (fit on validation): {thr.threshold:.4f}")
    print(f"Three-way policy: {art.policy.to_dict()}")
    print("Test cost by strategy:", json.dumps({k: round(v, 1) for k, v in res.test_costs.items()}, indent=2))
    conf = confusion_at_threshold(yt, scores, thr.threshold)
    print("Test confusion at cost-optimal threshold:", conf)
    dec = pd.Series(art.policy.decide_many(scores)).value_counts().to_dict()
    print("Test decisions under policy:", dec)

    imp = permutation_importance_table(art.model, Xt, yt, n_repeats=3, seed=args.seed)
    print("\nTop permutation importances (drop in test PR-AUC):\n" + imp.head(8).round(4).to_string(index=False))

    # ---- write reports -------------------------------------------------
    pd.DataFrame(res.val_metrics).T.to_csv(out / "metrics_validation.csv")
    pd.DataFrame(res.test_metrics).T.to_csv(out / "metrics_test.csv")
    imp.to_csv(out / "permutation_importance.csv", index=False)
    summary = {
        "synthetic_data": True,
        "seed": args.seed,
        "n_transactions": args.n,
        "split": res.split.summary(),
        "selected_model": art.model_name,
        "costs": vars(costs),
        "binary_threshold": thr.threshold,
        "policy": art.policy.to_dict(),
        "test_cost_by_strategy": res.test_costs,
        "test_confusion_at_cost_optimal_threshold": conf,
        "test_policy_decisions": {k: int(v) for k, v in dec.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    precision_recall_curve_csv(yt, scores).to_csv(out / "pr_curve_test.csv", index=False)

    art.save(out.parent / "artifacts" / "model.pkl")
    print(f"\nReports written to {out}")


def precision_recall_curve_csv(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y, s)
    step = max(1, len(p) // 500)
    return pd.DataFrame({"precision": p[::step], "recall": r[::step]})


if __name__ == "__main__":
    main()
