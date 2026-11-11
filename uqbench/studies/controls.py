"""Feature-matched controls for CAPE (results go to ``results/<model>/controls.csv``).

* Stacker: per-response logistic regression on CAPE's 22 trace statistics plus agreement features
  (own cluster share, top share, whether the response is in the top cluster, clusters per sample,
  answered), with the L2 strength chosen by validation log loss.
* Stacker + probe: the same with CAPE's cross-fitted hidden-state probe logit.
* Gated stacker: the probe variant on answer formats in the pool, the plain one elsewhere (CAPE's gate).
* Each stacker is reported raw and projected onto the coherent set (``+Coh``).
* CAPE agreement only: CAPE's answer model with no single-trace evidence (g constant).
* Evidence ICC: within-cluster correlation of CAPE's evidence (``evidence_icc.csv``).

Usage::

    python -m uqbench.studies.controls --model qwen3-8b
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from baselines.base import labels_of
from baselines.stacker import FeatureStacker
from cape.coherence import project_coherent
from cape.model import CAPE
from uqbench import metrics
from uqbench.calibration import select_calibrator
from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.records import load_pool, load_questions

MATH = tuple(load_yaml("benchmark")["math_sets"])   # pooled into math_all

def evaluate(model: str) -> pd.DataFrame:
    """Fit the controls on the pool, then score one test set at a time (the hidden states of all test sets do not
    fit in memory together); ``math_all`` pools the mathematics sets' responses."""
    bench = load_yaml("benchmark")
    pool = load_pool(model)
    train, val = [q for q in pool if q.split == "train"], [q for q in pool if q.split == "val"]
    methods = [FeatureStacker("none", "Stacker"), FeatureStacker("always", "Stacker + probe"),
               FeatureStacker("gated", "Stacker + probe, gated"),
               CAPE(evidence="scalar", features=[], name="CAPE agreement only")]
    yv, gv = labels_of(val), np.repeat([q.uid for q in val], [q.k for q in val])
    cals = {}
    for m in methods:
        m.fit(train, val)
        cals[m.name] = select_calibrator(m.score_all(val), yv, gv, True)[0]
        print(f"[controls] {model} {m.name} fitted", flush=True)
    del pool, train
    rows, out = {}, scores_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    pooled: dict[str, list] = {}   # name -> [(score, probability, labels)] of the mathematics sets

    def add(name, d, s, p, y):
        rows[(name, d)] = {"method": name, "dataset": d, "auroc": metrics.auroc(s, y), "brier": metrics.brier(p, y),
                           "ece": metrics.ece(p, y, bench["ece_bins"])}
        if d in MATH:
            pooled.setdefault(name, []).append((s, p, y))

    for d in bench["test_sets"]:
        qs = load_questions(model, d)
        y, saved = labels_of(qs), {}
        for m in methods:
            s = m.score_all(qs)
            p = cals[m.name](s)
            saved[m.name] = p
            add(m.name, d, s, p, y)
            if isinstance(m, FeatureStacker):
                start, proj = 0, []
                for q in qs:
                    has = np.array([x["pred"] is not None for x in q.samples])
                    proj.append(project_coherent(p[start:start + q.k], q.clusters, has))
                    start += q.k
                pp = np.concatenate(proj)
                saved[m.name + " +Coh"] = pp
                add(m.name + " +Coh", d, pp, pp, y)
        np.savez_compressed(out / f"controls_{d}.npz", **saved)   # calibrated (and projected), for paired tests
        print(f"[controls] {model} {d} done", flush=True)
        del qs
    sets = list(bench["test_sets"])
    if all(d in sets for d in MATH):
        for name, parts in pooled.items():
            s, p, y = (np.concatenate(x) for x in zip(*parts, strict=True))
            rows[(name, "math_all")] = {"method": name, "dataset": "math_all", "auroc": metrics.auroc(s, y),
                                        "brier": metrics.brier(p, y), "ece": metrics.ece(p, y, bench["ece_bins"])}
        sets.append("math_all")
    names = [n for m in methods for n in ([m.name, m.name + " +Coh"] if isinstance(m, FeatureStacker) else [m.name])]
    return pd.DataFrame([rows[(n, d)] for n in names for d in sets])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--icc-only", action="store_true", help="only the within-cluster evidence correlation")
    args = ap.parse_args()
    evidence_icc(args.model).to_csv(RESULTS_DIR / args.model / "evidence_icc.csv", index=False, float_format="%.4f")
    if not args.icc_only:
        evaluate(args.model).to_csv(RESULTS_DIR / args.model / "controls.csv", index=False, float_format="%.4f")



def icc(groups: list[np.ndarray]) -> float:
    """One-way random-effects intraclass correlation ICC(1) for unbalanced groups."""
    groups = [g for g in groups if len(g) >= 2]
    n = np.array([len(g) for g in groups], dtype=float)
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    a, total = len(groups), n.sum()
    msb, msw = ssb / (a - 1), ssw / (total - a)
    n0 = (total - (n ** 2).sum() / total) / (a - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def evidence_icc(model: str) -> pd.DataFrame:
    """Within-cluster correlation of CAPE's evidence g on responses that share an answer (equicorrelated
    evidence model); the effective number of independent responses in a cluster of n is n / (1 + (n-1) rho)."""
    import torch

    bench = load_yaml("benchmark")
    pool = load_pool(model)
    train, val = [q for q in pool if q.split == "train"], [q for q in pool if q.split == "val"]
    cape = CAPE()
    cape.fit(train, val)
    am = cape.full   # CAPE's answer model: g is the probe evidence
    rows = []
    for name in ["pool (val)"] + bench["test_sets"]:
        qs = val if name == "pool (val)" else load_questions(model, name)
        groups, by_label = [], {0: [], 1: []}
        for q in qs:
            X, R, _, _, _ = am.batch([q])
            with torch.no_grad():
                g = ((X @ am.params["w"] + am.params["b"]) * R)[0, : q.k].numpy()
            for c in {c for c, s in zip(q.clusters, q.samples, strict=True) if s["pred"] is not None}:
                idx = [i for i, (cc, s) in enumerate(zip(q.clusters, q.samples, strict=True))
                       if cc == c and s["pred"] is not None]
                if len(idx) >= 2:
                    groups.append(g[idx])
                    by_label[int(q.labels[idx[0]])].append(g[idx])
        rho = icc(groups)
        rows.append({"dataset": name, "clusters": len(groups), "icc": rho,
                     "n_eff_8": 8 / (1 + 7 * rho), "n_eff_4": 4 / (1 + 3 * rho),
                     # conditional on correctness, as in the correlated-evidence model
                     "icc_correct": icc(by_label[1]) if len(by_label[1]) > 2 else float("nan"),
                     "icc_wrong": icc(by_label[0]) if len(by_label[0]) > 2 else float("nan")})
        print(f"[controls] {model} ICC {name}: {rho:.3f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
