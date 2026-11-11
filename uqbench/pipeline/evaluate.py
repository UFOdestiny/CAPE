"""CPU stage: fit CAPE (and the baselines, when ``baselines/`` is present) on the training pool, score the frozen
test sets, report metrics.

Protocol: learned parts and hyper-parameters use the pool's train/val
split only. Brier/ECE use a calibration map chosen on the validation pool: Platt for score-valued
methods; for probability-valued methods the native probabilities unless 2-fold cross-fitted Platt
has lower validation NLL (raw Brier/ECE are reported as well); test sets
are scored once. Confidence intervals come from a question-level cluster bootstrap (all K
responses of a question are resampled together).

Usage::

    python -m uqbench.pipeline.evaluate --model qwen3-8b
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from cape import CAPE, CoherentProjection
from uqbench import metrics
from uqbench.calibration import Identity, select_calibrator
from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.diagnostics import certificate, null_mass_rows, structure_stats
from uqbench.records import labels_of, load_pool, load_questions

MATH_SETS = tuple(load_yaml("benchmark")["math_sets"])   # pooled into math_all
CI_METRICS = ("auroc", "brier", "ece")
CALIBRATION: dict[str, str] = {}   # method -> calibration map chosen on the validation pool


VAL_QUESTIONS: list = []   # validation pool, kept for post-hoc decision analyses


def baselines_package():
    """The optional ``baselines`` package (imported lazily), or None with a one-line notice when it is absent."""
    try:
        return importlib.import_module("baselines")
    except ModuleNotFoundError as err:
        if err.name != "baselines":
            raise
        print("[evaluate] baselines/ not found: evaluating CAPE only", flush=True)
        return None


def optional_baselines() -> list:
    """The compared estimators (``baselines.build_all``), or none when ``baselines/`` is absent."""
    registry = baselines_package()
    return registry.build_all() if registry is not None else []


def fit_methods(model: str) -> tuple[list, dict]:
    """Fit the baselines (if present) and CAPE on the training pool; return methods and their calibration maps."""
    pool = load_pool(model)
    train = [q for q in pool if q.split == "train"]
    val = [q for q in pool if q.split == "val"]
    VAL_QUESTIONS[:] = val
    methods = optional_baselines()
    for b in methods:
        b.fit(train, val)
        print(f"[evaluate] fitted {b.name}: {b.describe()}", flush=True)
    y_val = labels_of(val)
    val_groups = np.repeat([q.uid for q in val], [q.k for q in val])

    def calibrate(m):
        cal, kind = select_calibrator(m.score_all(val), y_val, val_groups, m.probabilistic)
        CALIBRATION[m.name] = kind
        return cal

    calibrators = {b.name: calibrate(b) for b in methods}
    ours = [CAPE(), CAPE(use_null=False, name="CAPE w/o null", main=False),
            CAPE(use_votes=False, name="CAPE w/o votes", main=False),
            CAPE(evidence="scalar", name="CAPE w/o probe", main=False)]
    for m in ours:
        m.fit(train, val)
        print(f"[evaluate] fitted {m.name}: {m.describe()}", flush=True)
    projected = [CoherentProjection(b, calibrators[b.name]) for b in methods if b.main]
    calibrators.update({m.name: calibrate(m) for m in ours})
    for m in projected:   # already calibrated before projection; a second map would leave the coherent set
        calibrators[m.name], CALIBRATION[m.name] = Identity(), "identity"
    ours += projected
    methods += ours
    return methods, calibrators


def bootstrap_ci(groups: list[np.ndarray], conf: np.ndarray, prob: np.ndarray, labels: np.ndarray,
                 n_boot: int, n_bins: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    stats = {m: [] for m in CI_METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y = labels[idx]
        if y.min() == y.max():
            continue
        stats["auroc"].append(metrics.auroc(conf[idx], y))
        stats["brier"].append(metrics.brier(prob[idx], y))
        stats["ece"].append(metrics.ece(prob[idx], y, n_bins))
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in stats.items()}


def evaluate_split(name: str, questions, methods, calibrators, bench, scores_out: dict) -> list[dict]:
    labels = labels_of(questions)
    sizes = np.array([q.k for q in questions])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    groups = [np.arange(s, s + k) for s, k in zip(starts, sizes, strict=True)]
    rows = []
    for b in methods:
        if b.name not in scores_out:
            scores_out[b.name] = b.score_all(questions)
        conf = scores_out[b.name]
        prob = calibrators[b.name](conf)
        row = {"dataset": name, "method": b.name, "main": b.main, "ours": b.paper == "ours",
               "reference": getattr(b, "reference", False),
               **metrics.evaluate(conf, labels, prob=prob)}
        row["ece"] = metrics.ece(prob, labels, bench["ece_bins"])
        if b.probabilistic:
            raw = np.clip(conf, 0.0, 1.0)
            row.update(brier_raw=metrics.brier(raw, labels), ece_raw=metrics.ece(raw, labels, bench["ece_bins"]))
        if b.main:
            for m, (lo, hi) in bootstrap_ci(groups, conf, prob, labels, bench["bootstrap"],
                                            bench["ece_bins"], bench["seed"]).items():
                row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        rows.append(row)
    return rows


def markdown_table(df: pd.DataFrame, datasets: list[str]) -> str:
    """Main rows; per column the best AUROC (max) / Brier, ECE (min) among main rows is bold."""
    main = df[df["main"]]
    header = "| Method | " + " | ".join(f"{d} AUROC↑ / Brier↓ / ECE↓" for d in datasets) + " |"
    lines = [header, "|---|" + "---|" * len(datasets)]
    acc = main.groupby("dataset")["acc"].first()
    lines.append("| *Acc (backbone)* | " + " | ".join(f"{acc[d]:.3f}" for d in datasets) + " |")
    best = {(d, m): (main[main["dataset"] == d][m].max() if m == "auroc" else main[main["dataset"] == d][m].min())
            for d in datasets for m in ("auroc", "brier", "ece")}
    for method in main["method"].unique():
        cells = []
        for d in datasets:
            r = main[(main["method"] == method) & (main["dataset"] == d)].iloc[0]
            vals = [f"**{r[m]:.3f}**" if np.isclose(r[m], best[(d, m)]) else f"{r[m]:.3f}"
                    for m in ("auroc", "brier", "ece")]
            cells.append(" / ".join(vals))
        label = method + (" †" if main[main["method"] == method]["reference"].iloc[0] else "")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n† matched-budget classic reference (same K responses), not a 2025+ baseline."


def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=bench["test_sets"])
    args = ap.parse_args()

    methods, calibrators = fit_methods(args.model)
    out_dir = scores_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_scores = {m.name: m.score_all(VAL_QUESTIONS) for m in methods if m.main}
    np.savez(out_dir / "val.npz", labels=labels_of(VAL_QUESTIONS),
             uid=np.repeat([q.uid for q in VAL_QUESTIONS], [q.k for q in VAL_QUESTIONS]),
             **val_scores, **{f"{k}__prob": calibrators[k](v) for k, v in val_scores.items()})
    rows, pooled, struct_rows, cert_rows, null_rows = [], [], [], [], []
    for name in args.datasets:
        questions = load_questions(args.model, name)
        scores = {}
        rows += evaluate_split(name, questions, methods, calibrators, bench, scores)
        struct_rows.append({"dataset": name, **structure_stats(questions)})
        for m in methods:
            if isinstance(m, CoherentProjection):
                before = calibrators[m.inner.name](scores[m.inner.name])
                cert_rows.append({"dataset": name, "method": m.inner.name,
                                  **certificate(questions, before, scores[m.name])})
        for m in methods:
            if m.name in ("CAPE", "SC", "CISC", "CAPE w/o null"):
                null_rows += null_mass_rows(name, questions, calibrators[m.name](scores[m.name]), m.name)
        np.savez(out_dir / f"{name}.npz", labels=labels_of(questions),
                 uid=np.repeat([q.uid for q in questions], [q.k for q in questions]),
                 **{k: v for k, v in scores.items()},
                 **{f"{k}__prob": calibrators[k](v) for k, v in scores.items()})
        if name in MATH_SETS:   # math_all needs labels and the stored scores only: drop the hidden states
            pooled.append(([replace(q, replay=[], followup=[]) for q in questions], scores))
        print(f"[evaluate] {args.model}/{name} done", flush=True)
    if pooled:
        questions = [q for qs, _ in pooled for q in qs]
        scores = {m.name: np.concatenate([s[m.name] for _, s in pooled]) for m in methods}
        rows += evaluate_split("math_all", questions, methods, calibrators, bench, scores)

    df = pd.DataFrame(rows)
    res_dir = RESULTS_DIR / args.model
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "metrics.csv", index=False, float_format="%.4f")
    (res_dir / "hyperparameters.json").write_text(
        json.dumps({b.name: {**b.describe(), "calibration": CALIBRATION.get(b.name)} for b in methods},
                   indent=2, default=float))
    columns = [d for d in args.datasets if d in MATH_SETS] + (["math_all"] if pooled else []) + \
              [d for d in args.datasets if d not in MATH_SETS]
    pd.DataFrame(struct_rows).to_csv(res_dir / "answer_structure.csv", index=False, float_format="%.4f")
    pd.DataFrame(cert_rows).to_csv(res_dir / "certificate.csv", index=False, float_format="%.5f")
    pd.DataFrame(null_rows).to_csv(res_dir / "null_mass.csv", index=False, float_format="%.4f")
    table = markdown_table(df, columns)
    (res_dir / "main_table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
