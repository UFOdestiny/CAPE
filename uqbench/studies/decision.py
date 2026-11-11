"""Decision-level analyses from saved scores (``$OUTPUT_DIR/<model>/scores/*.npz``).

* Consistent errors: on questions whose K answers are unanimous, how well each method separates
  unanimous-correct from unanimous-wrong questions (question-level AUROC; ``unanimous.csv``).
* Coherence effect: AUROC and Brier score of each estimator's calibrated probabilities before and after
  projection onto the coherent set (``coherence_effect.csv``); comparing calibrated with projected
  probabilities credits the projection only, not a sign flip made by the calibration map.
* Robust certificate (``robust_certificate.csv``): the robust bound of every projected estimator, checked on
  every (question, estimator) case, including questions whose labels violate (A1)-(A2).

Usage::

    python -m uqbench.studies.decision --model qwen3-8b
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.records import load_questions
from uqbench.reporting.results import COMPETITORS, low_parse

MAIN = COMPETITORS + ["CAPE"]   # every estimator, then CAPE


def vote_shares(questions) -> np.ndarray:
    out = []
    for q in questions:
        has = [s["pred"] is not None for s in q.samples]
        counts = Counter(c for c, h in zip(q.clusters, has, strict=True) if h)
        out += [counts[c] / q.k if h else 0.0 for c, h in zip(q.clusters, has, strict=True)]
    return np.array(out)


def agreement_floor(v: np.ndarray, y: np.ndarray) -> float:
    """min over functions h of mean (h(v) - y)^2 = sum over values of v of P(v) Var(y | v)."""
    total = 0.0
    for val in np.unique(v):
        m = v == val
        p = y[m].mean()
        total += m.mean() * p * (1 - p)
    return float(total)


def robust_check(questions, z, methods) -> dict:
    """Robust certificate: gain >= ||delta|| (||delta|| - 2 d(z, C)) for every question and estimator,
    including questions whose labels violate (A1)-(A2)."""
    from cape.coherence import project_coherent

    checks = held = incoherent = 0
    for m in methods:
        if f"{m}__prob" not in z.files or f"{m}+Coh" not in z.files:
            continue
        p_all, q_all, start = z[f"{m}__prob"], z[f"{m}+Coh"], 0
        for qq in questions:
            p, q, y = p_all[start : start + qq.k], q_all[start : start + qq.k], qq.labels
            start += qq.k
            has = np.array([s["pred"] is not None for s in qq.samples])
            d = float(np.linalg.norm(y - project_coherent(y, qq.clusters, has)))
            delta = float(np.linalg.norm(p - q))
            gain = float(((p - y) ** 2).sum() - ((q - y) ** 2).sum())
            checks += 1
            incoherent += d > 1e-9
            held += gain >= delta * (delta - 2 * d) - 1e-9
    return {"checks": checks, "held": held, "incoherent_label_checks": incoherent}


def unanimous_separation(model: str, test_sets: list[str]) -> pd.DataFrame:
    """Question-level AUROC between unanimous-correct and unanimous-wrong questions, one row per (test set,
    method) and a row pooled over the high-parse test sets; vote shares give every unanimous question the same
    confidence (AUROC 0.5)."""

    rows, pooled = [], []
    for name in test_sets:
        z = np.load(scores_dir(model) / f"{name}.npz")
        df = pd.DataFrame({"uid": z["uid"], "y": z["labels"], "share": z["SC"]})
        for m in MAIN + ["CAPE w/o null"]:
            df[m] = z[f"{m}__prob"] if f"{m}__prob" in z.files else z[m]
        ctrl = scores_dir(model) / f"controls_{name}.npz"
        if ctrl.exists():   # the gated stacker projected onto C (uqbench.studies.controls)
            df["Coherent stacker"] = np.load(ctrl)["Stacker + probe, gated +Coh"]
        q = df.groupby("uid").mean()
        q = q[q.share >= 1 - 1e-9]
        q["y"] = (q.y > 0.5).astype(int)
        q["dataset"] = name
        pooled.append(q)
    every = pd.concat(pooled)
    clean = every[~every.dataset.isin(low_parse(model))]
    for name, q in [*every.groupby("dataset"), ("all", clean)]:
        for m in [x for x in MAIN + ["CAPE w/o null", "Coherent stacker"] if x in q]:
            row = {"dataset": name, "method": m, "questions": len(q), "wrong": int((q.y == 0).sum()),
                   "auroc": roc_auc_score(q.y, q[m]) if q.y.nunique() > 1 else float("nan"),
                   "brier": float(np.mean((q[m] - q.y) ** 2)),
                   "conf_wrong": float(q.loc[q.y == 0, m].mean()) if (q.y == 0).any() else float("nan"),
                   "conf_right": float(q.loc[q.y == 1, m].mean()) if (q.y == 1).any() else float("nan")}
            if name == "all":   # AUROC within each test set, averaged (no between-set confidence levels)
                per = [roc_auc_score(g.y, g[m]) for _, g in q.groupby("dataset") if g.y.nunique() > 1]
                row["auroc_within_set"] = float(np.mean(per)) if per else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def coherence_effect(model: str, test_sets: list[str]) -> pd.DataFrame:
    """Calibrated vs. projected probabilities of every baseline and reference, per test set."""
    rows = []
    for name in test_sets:
        z = np.load(scores_dir(model) / f"{name}.npz")
        y = z["labels"]
        for m in MAIN[:-1]:
            if f"{m}__prob" not in z.files or f"{m}+Coh" not in z.files:
                continue
            p, q = z[f"{m}__prob"], z[f"{m}+Coh"]
            rows.append({"dataset": name, "method": m, "auroc_score": roc_auc_score(y, z[m]),
                         "auroc_prob": roc_auc_score(y, p), "auroc_coh": roc_auc_score(y, q),
                         "brier_prob": float(np.mean((p - y) ** 2)), "brier_coh": float(np.mean((q - y) ** 2))})
    return pd.DataFrame(rows)


def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    out = RESULTS_DIR / args.model
    unanimous_separation(args.model, bench["test_sets"]).to_csv(out / "unanimous.csv", index=False,
                                                                float_format="%.4f")
    coherence_effect(args.model, bench["test_sets"]).to_csv(out / "coherence_effect.csv", index=False,
                                                            float_format="%.4f")
    robust = {"checks": 0, "held": 0, "incoherent_label_checks": 0}
    for name in bench["test_sets"]:
        z = np.load(scores_dir(args.model) / f"{name}.npz")
        questions = load_questions(args.model, name, with_replay=False, with_followup=False)
        for key, val in robust_check(questions, z, COMPETITORS).items():
            robust[key] += val
    pd.DataFrame([robust]).to_csv(out / "robust_certificate.csv", index=False)
    print("robust certificate:", robust)


if __name__ == "__main__":
    main()
