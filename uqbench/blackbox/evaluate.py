"""Score the black-box Haiku run with every method the API permits and report the benchmark metrics.

Without log-probabilities or hidden states, CAPE uses API-visible evidence only: the vote prior and ∅ alone, or with
the verbal confidence as the response evidence g. When ``baselines/`` is present, the computable baselines are
scored too: the agreement-based ones (SC, SE, ConsistCal, SNNE), the verbal-confidence follow-up (LRMVerbal) and
the outcome reward model on the visible text (ORM). Fitting, calibration, projection and metrics are those of
``uqbench.pipeline.evaluate``; the training pools and test sets are the benchmark's.

Usage::

    python -m uqbench.blackbox.evaluate [--model claude-haiku-5-5]      # writes OUTPUT_DIR/<model>/results/

The paper's black-box table reads ``blackbox_metrics.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cape import CoherentProjection
from cape.model import AnswerModel
from uqbench.calibration import Identity, select_calibrator
from uqbench.config import load_yaml, run_dir
from uqbench.estimator import Estimator
from uqbench.pipeline.evaluate import baselines_package, evaluate_split
from uqbench.records import labels_of, load_questions

VERBAL_FALLBACK = 0.5
SINGLE = ("LRMVerbal", "ORM")


def verbal_logit(q) -> np.ndarray:
    p = np.array([f["verbal_conf"] if f and f["verbal_conf"] is not None else VERBAL_FALLBACK
                  for f in q.followup] if q.followup else [VERBAL_FALLBACK] * q.k)
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


class APIAnswerModel(AnswerModel):
    """CAPE's answer model with evidence observable through the API: none, or the verbal-confidence logit."""

    def __init__(self, verbal: bool, **kw):
        super().__init__(**kw)
        self.verbal = verbal

    def _features(self, q) -> np.ndarray:
        return verbal_logit(q)[:, None] if self.verbal else np.zeros((q.k, 0))

    def _difficulty(self, q) -> np.ndarray:   # replay-based difficulty is unavailable (and off by default)
        return np.zeros(1)

    def describe(self) -> dict:
        return {"l2": self.l2, "theta0": float(self.params["theta"][0]), "w": self.params["w"].tolist(),
                "b": float(self.params["b"]), "val_nll": self.val_nll}


class CAPEAPI(Estimator):
    paper = "ours"
    probabilistic = True

    def __init__(self, verbal: bool, name: str):
        self.verbal, self.name = verbal, name

    def fit(self, train, val) -> None:
        self.model = APIAnswerModel(self.verbal).fit(train, val)

    def score(self, q):
        import torch

        logp, M = self.model.log_outcomes(q)
        p = torch.nan_to_num(logp[: M.shape[0]].exp(), nan=0.0)
        return (M * p[:, None]).sum(0)[: q.k].numpy()

    def describe(self):
        return self.model.describe()


def blackbox_baselines(pool) -> list:
    """The baselines computable from the API's text output (none when ``baselines/`` is absent)."""
    b = baselines_package()
    if b is None:
        return []
    # the outcome reward model reads the visible responses (its GPU stage is part of baselines/); included once run
    single = [b.LRMVerbal()] + ([b.ORM()] if all(len(q.reward) == q.k for q in pool) else [])
    return [*single, b.SNNE(), b.ConsistCal(), b.SC(), b.SE()]


def load(model: str, name: str):
    return load_questions(model, name, with_replay=False, with_followup=True, partial=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--pilot", action="store_true", help="pipeline test: fit and validate on the whole pool")
    args = ap.parse_args()
    bench = load_yaml("benchmark")
    if not (run_dir(args.model) / "gen").exists():
        raise SystemExit(f"no generations under {run_dir(args.model)}; run -m uqbench.blackbox.batch merge first")

    pool = [q for name in bench["training_pools"] for q in load(args.model, name)]
    train, val = [q for q in pool if q.split == "train"], [q for q in pool if q.split == "val"]
    if args.pilot:
        train = val = pool
    print(f"[blackbox] pool: {len(train)} train / {len(val)} val questions")
    methods = blackbox_baselines(pool) + [CAPEAPI(verbal=False, name="CAPE (votes + ∅)"),
                                          CAPEAPI(verbal=True, name="CAPE (+ verbal)")]
    y_val = labels_of(val)
    groups = np.repeat([q.uid for q in val], [q.k for q in val])
    calibrators, describe = {}, {}
    for m in methods:
        m.fit(train, val)
        calibrators[m.name] = select_calibrator(m.score_all(val), y_val, groups, m.probabilistic)[0]
        describe[m.name] = m.describe()
        print(f"[blackbox] fitted {m.name}: {describe[m.name]}", flush=True)
    projected = [CoherentProjection(m, calibrators[m.name]) for m in methods
                 if m.name in (*SINGLE, "ConsistCal", "SNNE")]
    for m in projected:
        m.main = True
        calibrators[m.name] = Identity()
    methods += projected

    rows = []
    for name in bench["test_sets"]:
        questions = load(args.model, name)
        if not questions or (args.pilot and len(set(labels_of(questions))) < 2):
            continue
        rows += evaluate_split(name, questions, methods, calibrators, bench, {})
        print(f"[blackbox] {name}: {len(questions)} questions, acc={labels_of(questions).mean():.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = run_dir(args.model) / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blackbox_metrics.csv", index=False)
    table = df.pivot_table(index="method", columns="dataset", values=["auroc", "brier", "ece"], aggfunc="first")
    mean = df.groupby("method")[["auroc", "brier", "ece"]].mean()
    order = [m.name for m in methods]
    mean = mean.loc[[m for m in order if m in mean.index]]
    lines = [f"# {args.model}: black-box methods (mean over {df['dataset'].nunique()} test sets)", "",
             "| method | AUROC ↑ | Brier ↓ | ECE ↓ |", "|---|---:|---:|---:|"]
    lines += [f"| {m} | {r.auroc:.3f} | {r.brier:.3f} | {r.ece:.3f} |" for m, r in mean.iterrows()]
    (out / "blackbox_summary.md").write_text("\n".join(lines) + "\n")
    table.to_csv(out / "blackbox_per_dataset.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
