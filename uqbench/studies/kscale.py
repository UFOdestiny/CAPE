"""K-scaling study of the agreement floor: Qwen3-8B with 16 responses per question.

Merges the main run (``qwen3-8b``, samples 1-8) with the extra run (``qwen3-8b-x8``, samples
9-16, independent seeds), re-clusters all 16 answers, fits CAPE / CISC / SC on the usual pool
(K = 4), and evaluates on the first K responses for K in KS, together with the agreement floor
(the best Brier score of any function of the vote share at that K, oracle-calibrated) and the
error rate of unanimous answers.

Usage::

    python -m uqbench.studies.kscale --model qwen3-8b --extra qwen3-8b-x8
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from baselines.base import labels_of
from baselines.cisc import CISC
from baselines.reference import SC
from cape.model import CAPE
from uqbench import metrics
from uqbench.calibration import select_calibrator
from uqbench.config import RESULTS_DIR
from uqbench.records import load_pool, merge_runs
from uqbench.studies.decision import agreement_floor, vote_shares
from uqbench.verify import cluster_answers

KS = (1, 2, 4, 8, 12, 16)
SETS = ("math500", "olympiadbench", "minerva", "aime", "gpqa")


def merged(model: str, extra: str, dataset: str) -> list:
    return merge_runs(model, extra, dataset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--extra", default="qwen3-8b-x8")
    args = ap.parse_args()
    pool = load_pool(args.model)
    train = [q for q in pool if q.split == "train"]
    val = [q for q in pool if q.split == "val"]
    yv, gv = labels_of(val), np.repeat([q.uid for q in val], [q.k for q in val])
    methods = {"CAPE": CAPE(), "CISC": CISC(), "SC": SC()}
    calib = {}
    for name, m in methods.items():
        m.fit(train, val)
        calib[name] = select_calibrator(m.score_all(val), yv, gv, m.probabilistic)[0]
    rows = []
    for dataset in SETS:
        full = merged(args.model, args.extra, dataset)
        for k in KS:
            qs = [q.subset(list(range(k))) for q in full]
            qs = [replace(q, clusters=cluster_answers([s["pred"] for s in q.samples], q.task, q.uid)) for q in qs]
            y = labels_of(qs)
            unanimous = [q for q in qs if all(s["pred"] is not None for s in q.samples) and len(set(q.clusters)) == 1]
            row = {"dataset": dataset, "K": k, "questions": len(qs),
                   "agreement_floor": agreement_floor(vote_shares(qs), y),
                   "unanimous_share": len(unanimous) / len(qs),
                   "unanimous_error": 1.0 - float(np.mean([q.labels[0] for q in unanimous])) if unanimous else np.nan}
            for name, m in methods.items():
                s = m.score_all(qs)
                row[f"{name}_auroc"] = metrics.auroc(s, y)
                row[f"{name}_brier"] = metrics.brier(calib[name](s), y)
            rows.append(row)
        print(f"[kscale] {dataset} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / args.model / "kscale.csv", index=False, float_format="%.4f")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
