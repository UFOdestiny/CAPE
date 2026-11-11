"""Ablations of CAPE, computed on the CPU from cached signals; results go to
``results/<model>/ablation_<study>.csv``.

judge        Answer-set judge (``uqbench.pipeline.judge``) as extra evidence, and the judge alone (on the questions the
             judge was run on; every backbone).
forms        Alternative forms of the answer model: a learned within-cluster correlation r (cluster evidence scaled
             by the effective number of responses) and a question-dependent "none correct" logit (every backbone).
equivalence  Grader-independent answer clusters: math answers by exact string (raw or after LaTeX surface
             normalisation), short answers by bidirectional
             NLI entailment (``uqbench.pipeline.nli``); labels unchanged. CAPE and SC refitted on the re-clustered pool,
             the published estimators' calibrated scores re-projected, and the robust certificate checked
             (every backbone; ``ablation_equivalence.csv``, ``ablation_equivalence_projection.csv``).

Usage::

    python -m uqbench.studies.ablation --model qwen3-8b [--studies judge]
"""
from __future__ import annotations

import argparse
import re
from dataclasses import replace

import numpy as np
import pandas as pd

from baselines.base import labels_of
from baselines.reference import SC
from cape.coherence import project_coherent
from cape.model import CAPE
from uqbench import metrics
from uqbench.calibration import select_calibrator
from uqbench.config import RESULTS_DIR, load_yaml, run_dir, scores_dir
from uqbench.records import load_pool, load_questions
from uqbench.reporting.results import COMPETITORS

MATH = tuple(load_yaml("benchmark")["math_sets"])   # pooled into math_all
STUDIES = ("judge", "forms", "equivalence")
FORMS = {"CAPE": {}, "learned r": {"learn_r": True},
         "question-dependent null": {"use_difficulty": True, "null_features": True}}


def _metrics(prefix: str, s, p, y) -> dict:
    return {f"{prefix}_auroc": metrics.auroc(s, y), f"{prefix}_brier": metrics.brier(p, y),
            f"{prefix}_ece": metrics.ece(p, y)}


def judge_study(model, pool_name, train, val, test_sets) -> pd.DataFrame:
    """Every variant is fitted on the pool first; the test sets are then loaded one at a time (their hidden states
    do not fit in memory together) and restricted to the questions the judge was run on."""
    from cape.model import judge_evidence
    from uqbench.calibration import Platt
    from uqbench.pipeline import judge

    judged = set.intersection(*({u for d in test_sets for u in judge.load(model, d, think=t)} for t in (False, True)))
    pools = tuple(load_yaml("benchmark").get("training_pools", [pool_name]))

    def attach(qs, names, think):
        found = {uid: v for d in names for uid, v in judge.load(model, d, think=think).items()}
        for q in qs:
            q.judge = found.get(q.uid)
            q.__dict__.pop("_cape_features", None)

    def alone(qs):
        return np.concatenate([np.exp(judge_evidence(q)[0]) for q in qs])

    yv, groups = labels_of(val), np.repeat([q.uid for q in val], [q.k for q in val])
    variants = []   # (name, think, scorer, calibrator) in the order of the rows
    for think in (False, True):
        attach(train + val, pools, think)
        tag = "reasoning judge" if think else "judge"
        cape = CAPE(use_judge=True)
        cape.fit(train, val)
        variants.append((f"CAPE + {tag}", think, cape.score_all,
                         select_calibrator(cape.score_all(val), yv, groups, cape.probabilistic)[0]))
        variants.append((f"{tag} alone (clustered self-assessment style)", think, alone,
                         Platt().fit(alone(val), yv)))
        print(f"[ablation] {tag} fitted", flush=True)
    base = CAPE()
    base.fit(train, val)
    variants.insert(0, ("CAPE", None, base.score_all,
                        select_calibrator(base.score_all(val), yv, groups, base.probabilistic)[0]))

    rows = {name: {"variant": name} for name, *_ in variants}
    pooled = {name: [] for name, *_ in variants}   # mathematics sets, for math_all
    for d in test_sets:
        qs = [q for q in load_questions(model, d) if q.uid in judged]
        if not qs:
            continue
        y = labels_of(qs)
        for name, think, score, cal in variants:
            if think is not None:
                attach(qs, (d,), think)
            s = score(qs)
            rows[name].update(_metrics(d, s, cal(s), y))
            if d in MATH:
                pooled[name].append((s, cal(s), y))
        print(f"[ablation] {d} done", flush=True)
        del qs
    for name, parts in pooled.items():
        if parts:
            rows[name].update(_metrics("math_all", *(np.concatenate(x) for x in zip(*parts, strict=True))))
    return pd.DataFrame(list(rows.values()))


def _calibrated(method, val):
    groups = np.repeat([q.uid for q in val], [q.k for q in val])
    return select_calibrator(method.score_all(val), labels_of(val), groups, method.probabilistic)[0]


def _null_metrics(method, qs) -> dict:
    """Detection (AUROC) and Brier score of the event that no sampled answer is correct, from the ∅ mass."""
    null = np.array([float(method.log_outcomes(q)[0][-1].exp()) for q in qs])
    none = np.array([float(q.labels.max() == 0) for q in qs])
    return {"null_auroc": metrics.auroc(null, none), "null_brier": metrics.brier(null, none)}


def _response_rows(variants, test_sets, load, extra=None) -> list[dict]:
    """Metrics of fitted (name, method, calibrator) variants on each test set, loaded one at a time, and on
    math_all; ``extra(name, qs)`` adds per-test-set columns."""
    rows, pooled = [], {name: [] for name, *_ in variants}
    for d in test_sets:
        qs = load(d)
        y = labels_of(qs)
        more = extra(d, qs) if extra else {}
        for name, m, cal in variants:
            s = m.score_all(qs)
            row = {"variant": name, "dataset": d, **_metrics("x", s, cal(s), y), **more}
            if isinstance(m, CAPE):
                row.update(_null_metrics(m, qs))
            rows.append(row)
            if d in MATH:
                pooled[name].append((s, cal(s), y))
        print(f"[ablation] {d} done", flush=True)
        del qs
    for name, parts in pooled.items():
        if parts:
            rows.append({"variant": name, "dataset": "math_all",
                         **_metrics("x", *(np.concatenate(x) for x in zip(*parts, strict=True)))})
    return [{k.removeprefix("x_"): v for k, v in r.items()} for r in rows]


def forms_study(model, train, val, test_sets) -> pd.DataFrame:
    variants, learned = [], {}
    for name, kw in FORMS.items():
        m = CAPE(name=name, main=name == "CAPE", **kw)
        m.fit(train, val)
        learned[name] = m.full.describe()
        variants.append((name, m, _calibrated(m, val)))
        print(f"[ablation] {name}: {learned[name]}", flush=True)
    df = pd.DataFrame(_response_rows(variants, test_sets, lambda d: load_questions(model, d)))
    df["r"] = df.variant.map(lambda v: learned[v]["r"])
    return df


LATEX_NOISE = re.compile(r"\\left|\\right|\\[,;:! ]|\^\{?\\circ\}?|\$")


def latex_key(answer: str) -> str:
    """Surface normalisation of a LaTeX answer: spacing commands, \\left/\\right, degree marks and \\text-style
    wrappers dropped, \\dfrac/\\tfrac written \\frac (no symbolic equivalence)."""
    a = "".join(LATEX_NOISE.sub("", answer).split())
    a = re.sub(r"\\[dt]frac", r"\\frac", a)
    return re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", a)


def exact_clusters(q, key=lambda a: "".join(a.split())):
    """Cluster by the extracted answer string, whitespace removed (or another surface ``key``)."""
    keys = [key(s["pred"]) if s["pred"] is not None else None for s in q.samples]
    ids, out = {}, []
    for i, key in enumerate(keys):
        out.append(ids.setdefault(key, len(ids)) if key is not None else len(keys) + i)
    return replace(q, clusters=out)


def independent_clusters(model, key):
    """Math: string ``key``; short answers: NLI clusters (``uqbench.pipeline.nli``); option letters unchanged."""
    from uqbench.pipeline import nli

    cache = {}   # uid -> clusters, over every generated dataset with short answers
    for path in sorted((run_dir(model) / "clusters_nli").glob("*.jsonl")):
        cache.update(nli.load(model, path.stem))

    def fn(q):
        if q.task == "math":
            return exact_clusters(q, key)
        if q.task == "qa":
            return replace(q, clusters=cache[q.uid])
        return q
    return fn


def coherent_labels(q) -> bool:
    """(A1)-(A2) hold for the labels under the question's clusters: the labels lie in the coherent set."""
    has = np.array([s["pred"] is not None for s in q.samples])
    return bool(np.allclose(q.labels, project_coherent(q.labels, q.clusters, has)))


def projection_rows(rule, d, qs, z) -> list[dict]:
    """Every published estimator's calibrated probabilities projected under the question's clusters: Brier score
    and AUROC before and after, and the robust certificate gain >= ||delta|| (||delta|| - 2 d(z, C)) per question."""
    from sklearn.metrics import roc_auc_score

    y, rows = z["labels"], []
    for m in COMPETITORS:
        if f"{m}__prob" not in z.files:
            continue
        p_all, out, start = z[f"{m}__prob"], [], 0
        held = incoherent = 0
        gains, bounds = [], []
        for q in qs:
            p, lab = p_all[start : start + q.k], q.labels
            start += q.k
            has = np.array([s["pred"] is not None for s in q.samples])
            proj = project_coherent(p, q.clusters, has)
            dist = float(np.linalg.norm(lab - project_coherent(lab, q.clusters, has)))
            delta = float(np.linalg.norm(p - proj))
            gain = float(((p - lab) ** 2).sum() - ((proj - lab) ** 2).sum())
            bound = delta * (delta - 2 * dist)
            held += gain >= bound - 1e-9
            incoherent += dist > 1e-9
            gains.append(gain)
            bounds.append(bound)
            out.append(proj)
        coh = np.concatenate(out)
        rows.append({"rule": rule, "dataset": d, "method": m, "auroc_prob": roc_auc_score(y, p_all),
                     "auroc_coh": roc_auc_score(y, coh), "brier_prob": float(np.mean((p_all - y) ** 2)),
                     "brier_coh": float(np.mean((coh - y) ** 2)), "questions": len(qs),
                     "brier_raised": int(sum(g < -1e-12 for g in gains)), "incoherent": incoherent,
                     "held": held, "gain": float(np.sum(gains) / len(y)), "bound": float(np.sum(bounds) / len(y))})
    return rows


def equivalence_study(model, train, val, test_sets) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = [d for d in test_sets if d in MATH or d == "triviaqa"]   # option letters cluster the same either way
    rules = {"grader": lambda q: q, "independent": independent_clusters(model, lambda a: "".join(a.split())),
             "independent, LaTeX-normalised": independent_clusters(model, latex_key)}
    rows, proj = [], []
    for rule, fn in rules.items():
        tr, va = [fn(q) for q in train], [fn(q) for q in val]
        variants = []
        for name, m in (("CAPE", CAPE()), ("SC", SC())):
            m.fit(tr, va)
            variants.append((name, m, _calibrated(m, va)))

        def extra(d, qs, rule=rule):
            proj.extend(projection_rows(rule, d, qs, np.load(scores_dir(model) / f"{d}.npz")))
            return {"clusters_per_question": float(np.mean([len(set(q.clusters)) for q in qs])),
                    "coherent_label_share": float(np.mean([coherent_labels(q) for q in qs]))}

        for r in _response_rows(variants, sets, lambda d, fn=fn: [fn(q) for q in load_questions(model, d)], extra):
            rows.append({"rule": rule, **r})
        print(f"[ablation] equivalence {rule} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(proj)


def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=bench["ablation_backbone"])
    ap.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    args = ap.parse_args()
    pool = load_pool(args.model)
    train = [q for q in pool if q.split == "train"]
    val = [q for q in pool if q.split == "val"]
    run = {
        "judge": lambda: judge_study(args.model, bench["pool"], train, val, bench["test_sets"]),
        "forms": lambda: forms_study(args.model, train, val, bench["test_sets"]),
        "equivalence": lambda: equivalence_study(args.model, train, val, bench["test_sets"]),
    }
    for study in args.studies:
        out = run[study]()
        frames = out if isinstance(out, tuple) else (out,)
        for suffix, df in zip(("", "_projection"), frames, strict=False):
            df.to_csv(RESULTS_DIR / args.model / f"ablation_{study}{suffix}.csv", index=False,
                      float_format="%.5f" if suffix else "%.4f")
        print(f"[ablation] {study} written", flush=True)


if __name__ == "__main__":
    main()
