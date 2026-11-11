"""Within-question ranking (``results/<model>/within_question.csv``): mean AUROC over the questions that have both
correct and wrong responses, i.e. how well a method ranks the answers of one question (CAPE, SC and CISC rank
whole clusters).

Usage::

    python -m uqbench.studies.within_question --model qwen3-8b
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.reporting.results import COMPETITORS


def within_question_auroc(uid: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    vals = []
    for u in np.unique(uid):
        m = uid == u
        if 0 < y[m].sum() < m.sum():
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(model: str) -> pd.DataFrame:
    rows = []
    for name in load_yaml("benchmark")["test_sets"]:
        z = np.load(scores_dir(model) / f"{name}.npz")
        y, uid = z["labels"], z["uid"]
        for m in COMPETITORS + ["CAPE"]:
            rows.append({"dataset": name, "method": m, "within_auroc": within_question_auroc(uid, y, z[m])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    evaluate(args.model).to_csv(RESULTS_DIR / args.model / "within_question.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
