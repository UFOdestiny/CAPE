"""Combine the results of the reported backbones (``uqbench/configs/benchmark.yaml: backbones``) into
``results/summary.md``.

Per backbone: main table (``main_table.md``), CAPE win counts against the best baseline per cell,
paired bootstrap of the AUROC gain over the strongest baseline, the label-free certificate and coherence
projection (``+Coh``) of every baseline, CAPE ablation rows, and answer-structure statistics. The paper's tables and
figures are written by ``visualization/build.py``.

Every (backbone, test set, method) result is also exported for later analysis: one file per backbone and test set,
``results/<model>/per_dataset/<dataset>.csv`` (one row per method), and all of them in ``results/all_metrics.csv``.
The per-response verdicts behind them (answer, cluster, label and every method's probability) go to
``results/<model>/responses/<dataset>.parquet``, and CAPE's standing against the competitors per backbone and test set
(ranks, strongest competitor per metric, gaps to the main rivals) to ``results/cape_vs_competitors.csv``.

Usage::

    python -m uqbench.reporting.report
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.reporting.results import (  # noqa: F401  (CONSTRUCTED: re-export)
    BASELINES,
    CONSTRUCTED,
    MATH_SETS,
    REFERENCES,
)
from uqbench.stats import paired_bootstrap

DATASETS = [*MATH_SETS, "math_all", *[d for d in load_yaml("benchmark")["test_sets"] if d not in MATH_SETS]]
METRICS = {"auroc": max, "brier": min, "ece": min}
SUMMARY_SETS = ["math_all", *[d for d in DATASETS if d not in (*MATH_SETS, "math_all")]]   # pooled math + the rest


def win_table(df: pd.DataFrame, method: str = "CAPE", pool: list[str] = BASELINES) -> tuple[str, int, int]:
    lines = ["| Dataset | " + " | ".join(f"{m} (CAPE / best baseline)" for m in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    wins = total = 0
    for d in DATASETS:
        sub = df[df.dataset == d].set_index("method")
        cells = []
        if sub.empty:
            continue
        for m, best in METRICS.items():
            pool_here = [x for x in pool if x in sub.index]
            b, c = best(sub.loc[pool_here, m]), sub.loc[method, m]
            ok = c >= b if m == "auroc" else c <= b
            wins, total = wins + ok, total + 1
            who = sub.loc[pool_here, m].idxmax() if m == "auroc" else sub.loc[pool_here, m].idxmin()
            cells.append(f"{'**' if ok else ''}{c:.3f}{'**' if ok else ''} / {b:.3f} ({who})")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines), wins, total


def significance_table(model: str, df: pd.DataFrame) -> str:
    bench = load_yaml("benchmark")
    lines = ["| Dataset | strongest baseline (AUROC) | ΔAUROC CAPE − baseline [95% CI] | P(Δ ≤ 0) |",
             "|---|---|---|---|"]
    for d in DATASETS:
        sub = df[df.dataset == d].set_index("method")
        if sub.empty:
            continue
        rival = sub.loc[BASELINES, "auroc"].idxmax()
        r = paired_bootstrap(model, d, "CAPE", rival, bench["bootstrap"], bench["seed"])
        lines.append(f"| {d} | {rival} | {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p_le0']:.3f} |")
    return "\n".join(lines)


def certificate_table(model_dir) -> str:
    path = model_dir / "certificate.csv"
    if not path.exists():
        return "(no certificate)"
    cols = ["certified_gain", "realised_gain", "bound_held", "bound_held_coherent"]
    raw = pd.read_csv(path)
    cert = raw.groupby("method")[[c for c in cols if c in raw]].mean()
    lines = ["| Baseline | certified Brier gain (label-free) | realised Brier gain | bound held | held on (A1)-(A2) |",
             "|---|---|---|---|---|"]
    for m, r in cert.iterrows():
        coh = f"{r.bound_held_coherent:.0%}" if "bound_held_coherent" in r else "–"
        lines.append(f"| {m} | {r.certified_gain:.4f} | {r.realised_gain:.4f} | {r.bound_held:.0%} | {coh} |")
    return "\n".join(lines)


def coherence_table(df: pd.DataFrame) -> str:
    sets = [d for d in SUMMARY_SETS if d in set(df["dataset"])]
    lines = ["| Baseline | " + " | ".join(f"{d}: AUROC / Brier, before → after" for d in sets) + " |",
             "|---|" + "---|" * len(sets)]
    for b in BASELINES + REFERENCES:
        cells = []
        for d in sets:
            sub = df[df.dataset == d].set_index("method")
            if b not in sub.index or f"{b}+Coh" not in sub.index:
                cells.append("–")
                continue
            a, c = sub.loc[b], sub.loc[f"{b}+Coh"]
            cells.append(f"{a['auroc']:.3f} → {c['auroc']:.3f} / {a['brier']:.3f} → {c['brier_raw']:.3f}")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_table(df: pd.DataFrame) -> str:
    rows = [r for r in ["CAPE", "CAPE w/o null", "CAPE w/o votes", "CAPE w/o probe"]
            if r in set(df["method"])]
    sets = [d for d in SUMMARY_SETS if d in set(df["dataset"])]
    lines = ["| Variant | " + " | ".join(f"{d} AUROC / Brier / ECE" for d in sets) + " |",
             "|---|" + "---|" * len(sets)]
    for r in rows:
        cells = []
        for d in sets:
            x = df[(df.dataset == d) & (df.method == r)].iloc[0]
            cells.append(f"{x['auroc']:.3f} / {x['brier']:.3f} / {x['ece']:.3f}")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# per-method results outside metrics.csv: (file, methods to take; None = all)
EXTRA_SOURCES = (("controls.csv", None), ("stacker_null.csv", ["Coherent stacker + learned null"]))


def export_cells() -> pd.DataFrame:
    """Write one row per (backbone, test set, method): ``results/<model>/per_dataset/<dataset>.csv`` and
    ``results/all_metrics.csv``. Rows come from ``metrics.csv`` (every method and variant scored by
    ``uqbench.pipeline.evaluate``) and the feature-matched controls (``source`` names the file)."""
    rows = []
    for model in load_yaml("benchmark")["backbones"]:
        model_dir = RESULTS_DIR / model
        frames = [pd.read_csv(model_dir / "metrics.csv").assign(source="metrics.csv")]
        for name, keep in EXTRA_SOURCES:
            if (model_dir / name).exists():
                extra = pd.read_csv(model_dir / name)
                frames.append((extra if keep is None else extra[extra.method.isin(keep)]).assign(source=name))
        df = pd.concat(frames, ignore_index=True)
        df = df[df.dataset.isin(DATASETS)]
        out = model_dir / "per_dataset"
        out.mkdir(exist_ok=True)
        for old in out.glob("*.csv"):
            old.unlink()
        for d, g in df.groupby("dataset", sort=False):
            g.drop(columns="dataset").to_csv(out / f"{d}.csv", index=False, float_format="%.4f")
        rows.append(df.assign(backbone=model))
    cells = pd.concat(rows, ignore_index=True)
    cells = cells[["backbone", "dataset", "method", *[c for c in cells.columns if c not in ("backbone", "dataset",
                                                                                              "method")]]]
    cells.to_csv(RESULTS_DIR / "all_metrics.csv", index=False, float_format="%.4f")
    return cells


def export_responses(model: str) -> None:
    """``results/<model>/responses/<dataset>.parquet``: one row per sampled response with its generated dataset,
    question id, sample index, extracted answer (``pred``; empty = no answer), answer cluster, label, every method's
    calibrated probability (``p:<method>``) and raw score (``raw:<method>``), the controls' and the modular
    implementation's probabilities, and per-question "none correct" quantities (``q:`` columns, repeated over the
    question's responses)."""
    from uqbench.records import load_questions
    from uqbench.studies.refinement import null_masses

    sdir, out = scores_dir(model), RESULTS_DIR / model / "responses"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()
    for name in load_yaml("benchmark")["test_sets"]:
        if not (sdir / f"{name}.npz").exists():
            continue
        z = np.load(sdir / f"{name}.npz")
        questions = load_questions(model, name, with_replay=False, with_followup=False)
        ks = [q.k for q in questions]
        uid = np.repeat([q.uid for q in questions], ks)
        if not np.array_equal(uid.astype(str), z["uid"].astype(str)):
            raise ValueError(f"{model}/{name}: response order of the scores differs from the generations")
        cols = {"part": np.repeat([q.dataset for q in questions], ks), "uid": uid,
                "k": np.concatenate([np.arange(k) for k in ks]),
                "pred": [s.get("pred") for q in questions for s in q.samples],
                "cluster": np.concatenate([q.clusters for q in questions]),
                "closed": [bool(s.get("closed", True)) for q in questions for s in q.samples],
                "correct": z["labels"].astype(int)}
        cols["no_answer"] = np.array([p is None for p in cols["pred"]])
        for key in z.files:
            if key.endswith("__prob"):
                cols[f"p:{key[:-6]}"] = z[key]
            elif key not in ("labels", "uid"):
                cols[f"raw:{key}"] = z[key]
        for extra in (f"controls_{name}.npz", f"modular_{name}.npz"):
            if (sdir / extra).exists():
                e = np.load(sdir / extra)
                cols.update({f"p:{k}": e[k] for k in e.files if k != "null_classifier"})
                if "null_classifier" in e.files:
                    cols["q:null separate classifier"] = np.repeat(e["null_classifier"], ks)
        none = np.array([float(not np.any(q.labels == 1)) for q in questions])
        cols["q:none_correct"] = np.repeat(none, ks)
        cols["q:null CAPE"] = np.repeat(null_masses(questions, z["CAPE__prob"]), ks)
        df = pd.DataFrame(cols)
        df["pred"] = df["pred"].astype("string")
        df.to_parquet(out / f"{name}.parquet", index=False)


RIVALS = ("SEP+Coh", "ReasonProbe+Coh", "DeepConf+Coh", "SC")   # the strongest published rows of Table 1


def export_standing(cells: pd.DataFrame) -> None:
    """``results/cape_vs_competitors.csv``: per (backbone, test set), CAPE's rank among itself and every competitor
    with and without projection (1 = best) on AUROC, Brier and ECE, the strongest competitor on each metric with its
    value, and CAPE's gap to the main rivals (positive = CAPE better)."""
    pool = BASELINES + REFERENCES
    pool = pool + [f"{m}+Coh" for m in pool]
    rows = []
    for (model, d), g in cells[cells.source == "metrics.csv"].groupby(["backbone", "dataset"], sort=False):
        x = g.set_index("method")
        if "CAPE" not in x.index:
            continue
        comp, cape = x.loc[[m for m in pool if m in x.index]], x.loc["CAPE"]
        both = pd.concat([comp, x.loc[["CAPE"]]])
        row = {"backbone": model, "dataset": d, "competitors": len(comp),
               "rank_auroc": int(both.auroc.rank(ascending=False, method="min")["CAPE"]),
               "rank_brier": int(both.brier.rank(method="min")["CAPE"]),
               "rank_ece": int(both.ece.rank(method="min")["CAPE"]),
               "cape_auroc": cape.auroc, "cape_brier": cape.brier, "cape_ece": cape.ece,
               "best_auroc_method": comp.auroc.idxmax(), "best_auroc": comp.auroc.max(),
               "best_brier_method": comp.brier.idxmin(), "best_brier": comp.brier.min(),
               "best_ece_method": comp.ece.idxmin(), "best_ece": comp.ece.min()}
        for r in RIVALS:
            if r in x.index:
                row[f"gain_auroc_vs_{r}"] = cape.auroc - x.loc[r, "auroc"]
                row[f"gain_brier_vs_{r}"] = x.loc[r, "brier"] - cape.brier
                row[f"gain_ece_vs_{r}"] = x.loc[r, "ece"] - cape.ece
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cape_vs_competitors.csv", index=False, float_format="%.4f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    missing = [model for model in load_yaml("benchmark")["backbones"]
               if not (RESULTS_DIR / model / "metrics.csv").is_file()]
    if missing:
        parser.error(f"missing metrics.csv for: {', '.join(missing)}; run evaluate first")
    export_standing(export_cells())
    for model in load_yaml("benchmark")["backbones"]:
        export_responses(model)
    parts = ["# Results summary\n",
             "Generated by `python -m uqbench.reporting.report` from `results/<model>/metrics.csv`. "
             f"Bold in win tables = CAPE ties or beats the best of the {len(BASELINES)} baselines on that cell.\n"]
    for model in load_yaml("benchmark")["backbones"]:
        model_dir = RESULTS_DIR / model
        df = pd.read_csv(model_dir / "metrics.csv")
        table, wins, total = win_table(df)
        table_ref, wins_ref, total_ref = win_table(df, pool=BASELINES + REFERENCES)
        parts += [f"## {model_dir.name}\n", "### Main table\n", (model_dir / "main_table.md").read_text(),
                  f"\n### CAPE vs. best baseline per cell — {wins}/{total} cells\n", table,
                  f"\n### CAPE vs. best of baselines + matched-budget references (SC, SE) — "
                  f"{wins_ref}/{total_ref} cells\n", table_ref,
                  "\n### Paired question-level bootstrap (AUROC)\n", significance_table(model_dir.name, df),
                  "\n### Label-free Brier certificate of the coherence projection\n",
                  certificate_table(model_dir),
                  "\n### Label-free coherence projection on every baseline\n",
                  coherence_table(df), "\n### CAPE ablations\n", ablation_table(df),
                  "\n### Answer structure\n", pd.read_csv(model_dir / "answer_structure.csv").to_markdown(
                      index=False, floatfmt=".3f"), ""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
