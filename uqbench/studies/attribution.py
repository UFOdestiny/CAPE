"""Attribute the gains to the two rules, from stored scores (no refitting).

* ``attribution_safety.csv`` (rule 1): per benchmark and published estimator, on questions whose labels satisfy
  (A1)-(A2), the share of questions whose Brier score the projection raises against the share the ad hoc fusion
  "calibrated probability x vote share" raises, the latter's worst per-question increase, how often the cap
  binds, and the share of the realised gain certified without labels.
* ``attribution_null.csv`` (rule 2): per benchmark, predictor of "none correct" (``NULL_PREDICTORS``) and stratum
  (all questions, unanimous, split; one, two or three or more answers), the event rate, mean mass, share of
  questions with zero mass, AUROC and Brier score of the event.
* ``attribution_null_paired.csv``: CAPE's learned mass against each rival in ``RIVALS`` (the mass the coherent
  stacker's projection leaves; CAPE fitted with a per-response loss), per benchmark and stratum: question-level
  paired bootstrap of the event's Brier score and AUROC.
Run after ``uqbench.studies.refinement`` (which writes ``questions.csv``).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.diagnostics import labels_coherent
from uqbench.records import load_questions
from uqbench.studies.refinement import null_masses

N_BOOT = 2000
NULL_PREDICTORS = {   # questions.csv column -> label; "1 - x" columns are read as one minus the top answer
    "cape_null": "CAPE", "evidence_null": "separate classifier", "stacker_null": "coherent stacker, leftover",
    "deepconf_null": "DeepConf +C, leftover", "1 - stacker_top": "coherent stacker, 1 - top",
    "1 - sc_top": "SC, 1 - top"}
COMPUTED: dict[str, str] = {}   # columns computed here from a method's stored scores (column -> method)


def answered_clusters(q) -> int:
    return len({c for c, s in zip(q.clusters, q.samples, strict=True) if s["pred"] is not None})


def vote_share(q) -> np.ndarray:
    has = np.array([s["pred"] is not None for s in q.samples])
    counts = pd.Series(np.asarray(q.clusters)[has]).value_counts()
    return np.array([counts[c] / q.k if h else 0.0 for c, h in zip(q.clusters, has, strict=True)])


def safety_rows(name: str, questions, z) -> list[dict]:
    """Rule 1 per estimator: projection versus the product with the vote share, question by question."""
    rows = []
    shares = [vote_share(q) for q in questions]
    coherent = np.array([labels_coherent(q) for q in questions])
    for key in sorted(k for k in z.files if k.endswith("__prob")):
        m = key[:-len("__prob")]
        if f"{m}+Coh" not in z.files or m.endswith("+Coh") or m == "CAPE":
            continue
        p, proj = z[key], z[f"{m}+Coh"]
        d_proj, d_prod, cap, cert, real, start = [], [], [], 0.0, 0.0, 0
        for q, v in zip(questions, shares, strict=True):
            a, b = p[start:start + q.k], proj[start:start + q.k]
            start += q.k
            base = float(np.mean((a - q.labels) ** 2))
            d_proj.append(float(np.mean((b - q.labels) ** 2)) - base)
            d_prod.append(float(np.mean((a * v - q.labels) ** 2)) - base)
            has = np.array([s["pred"] is not None for s in q.samples])
            means = pd.Series(a[has]).groupby(np.asarray(q.clusters)[has]).mean()
            cap.append(bool(means.sum() > 1 + 1e-12))
            if labels_coherent(q):
                cert += float(((a - b) ** 2).sum())
                real += float(((a - q.labels) ** 2).sum() - ((b - q.labels) ** 2).sum())
        d_proj, d_prod = np.array(d_proj)[coherent], np.array(d_prod)[coherent]
        rows.append({"dataset": name, "method": m, "questions": int(coherent.sum()),
                     "proj_harmed": float((d_proj > 1e-12).mean()), "prod_harmed": float((d_prod > 1e-12).mean()),
                     "prod_worst": float(d_prod.max()), "prod_p99": float(np.percentile(d_prod, 99)),
                     "proj_mean": float(d_proj.mean()), "prod_mean": float(d_prod.mean()),
                     "cap_active": float(np.mean(cap)), "certified_share": cert / real if real > 0 else np.nan})
    return rows


def strata(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    m = frame.clusters.to_numpy()
    return {"all": np.ones(len(frame), bool), "unanimous": frame.unanimous.to_numpy(bool),
            "split": ~frame.unanimous.to_numpy(bool), "m1": m == 1, "m2": m == 2, "m3+": m >= 3}


def mass(frame: pd.DataFrame, col: str) -> np.ndarray:
    return 1.0 - frame[col[4:]].to_numpy() if col.startswith("1 - ") else frame[col].to_numpy()


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def null_rows(name: str, frame: pd.DataFrame) -> list[dict]:
    y = (~frame.correct.to_numpy(bool)).astype(float)
    rows = []
    for stratum, keep in strata(frame).items():
        if not keep.any():
            continue
        for col, label in NULL_PREDICTORS.items():
            if col.removeprefix("1 - ") not in frame:
                continue
            s = mass(frame, col)[keep]
            rows.append({"dataset": name, "stratum": stratum, "predictor": label, "questions": int(keep.sum()),
                         "none_rate": float(y[keep].mean()), "mean_mass": float(s.mean()),
                         "zero_mass": float((s < 1e-9).mean()), "auroc": auroc(y[keep], s),
                         "brier": float(np.mean((s - y[keep]) ** 2))})
    return rows


RIVALS = {"stacker_null": "leftover"}   # column -> rival label


def paired_rows(name: str, frame: pd.DataFrame, seed: int) -> list[dict]:
    """CAPE's learned mass against each rival's (the coherent stacker's leftover): positive deltas favour CAPE."""
    rows = []
    for col, rival in RIVALS.items():
        if col in frame:
            rows += [dict(r, rival=rival) for r in _paired(name, frame, frame[col].to_numpy(), seed)]
    return rows


def _paired(name: str, frame: pd.DataFrame, b: np.ndarray, seed: int) -> list[dict]:
    y = (~frame.correct.to_numpy(bool)).astype(float)
    a = frame.cape_null.to_numpy()
    rows = []
    rng = np.random.default_rng(seed)
    for stratum, keep in strata(frame).items():
        idx0 = np.flatnonzero(keep)
        if len(idx0) < 10:
            continue

        def stats(idx):
            yy = y[idx]
            return (float(np.mean((b[idx] - yy) ** 2) - np.mean((a[idx] - yy) ** 2)),
                    auroc(yy, a[idx]) - auroc(yy, b[idx]))

        point = stats(idx0)
        boot = np.array([stats(rng.choice(idx0, len(idx0))) for _ in range(N_BOOT)])
        row = {"dataset": name, "stratum": stratum, "questions": len(idx0)}
        for j, metric in enumerate(("brier", "auroc")):
            d = boot[:, j][np.isfinite(boot[:, j])]
            row.update({f"{metric}_delta": point[j], f"{metric}_lo": float(np.percentile(d, 2.5)) if len(d) else np.nan,
                        f"{metric}_hi": float(np.percentile(d, 97.5)) if len(d) else np.nan,
                        f"p_{metric}": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else 1.0})
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = RESULTS_DIR / args.model
    qcsv = pd.read_csv(out / "questions.csv")
    safety, nulls, paired = [], [], []
    for name in load_yaml("benchmark")["test_sets"]:
        path = scores_dir(args.model) / f"{name}.npz"
        if not path.exists():
            continue
        questions = load_questions(args.model, name)
        z = np.load(path)
        safety += safety_rows(name, questions, z)
        frame = qcsv[qcsv.dataset == name].set_index("uid").loc[[q.uid for q in questions]].reset_index()
        frame["clusters"] = [answered_clusters(q) for q in questions]
        for col, method in COMPUTED.items():
            if f"{method}__prob" in z.files:
                frame[col] = null_masses(questions, z[f"{method}__prob"])
        nulls += null_rows(name, frame)
        paired += paired_rows(name, frame, args.seed)
        print(f"[attribution] {args.model} {name} done", flush=True)
    pd.DataFrame(safety).to_csv(out / "attribution_safety.csv", index=False, float_format="%.6f")
    pd.DataFrame(nulls).to_csv(out / "attribution_null.csv", index=False, float_format="%.6f")
    pd.DataFrame(paired).to_csv(out / "attribution_null_paired.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
