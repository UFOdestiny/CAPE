"""Appendix: every method on every (backbone, test set) pair, with paired significance marks."""
from __future__ import annotations

import numpy as np

from uqbench.reporting.results import BACKBONES, COMPETITORS, DATASETS, low_parse, metric_frames
from visualization.data import HIGHER, significance
from visualization.tables.fmt import mark, name, ranked
from visualization.tables.results import summary_long

ROWS = COMPETITORS + ["CAPE"]
SUMMARY_GROUPS = (
    ("Single response", ["DeepConf", "CDG", "SelfCertainty", "LRMVerbal", "ReasonProbe", "SEP", "ORM"]),
    ("Multiple responses", ["SC", "SE", "CISC", "DeepConf-Vote", "ConsistCal", "SNNE", "CoCoA", "SemEnergy"]),
    ("Published, projected", ["DeepConf+C", "SEP+Coh", "ReasonProbe+Coh"]),
    ("Controls", ["DeepConf×SC", "Stacker"]),
    ("Ours", ["CAPE"]),
)


def summary_full() -> str:
    """Per backbone, mean AUROC and Brier over the high-parse test sets, and the mean over all high-parse pairs
    with ECE; rows grouped as in Table 1. Bold: best per column; underline: second."""
    long = summary_long(metric_frames())
    n_pairs = len(long[["model", "dataset"]].drop_duplicates())
    models = [m for m in BACKBONES if m in set(long.model)]
    rows = [r for _, group in SUMMARY_GROUPS for r in group if r in set(long.method)]
    table = {}
    for r in rows:
        d = long[long.method == r]
        vals = []
        for m in models:
            g = d[d.model == m]
            vals += [g.auroc.mean(), g.brier.mean()]
        table[r] = vals + [d.auroc.mean(), d.brier.mean(), d.ece.mean()]
    arr = np.array([table[r] for r in rows], dtype=float)
    higher = [True, False] * (len(models) + 1) + [False]
    rank = [ranked(arr[:, j], hi) for j, hi in enumerate(higher)]
    n = 2 * len(models) + 3
    out = [r"\begin{tabular}{@{}l" + "cc" * len(models) + "ccc@{}}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{BACKBONES[m]}}}" for m in models)
           + rf" & \multicolumn{{3}}{{c}}{{All {n_pairs} settings}} \\",
           "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models)))
           + rf"\cmidrule(lr){{{2 + 2 * len(models)}-{n + 1}}}",
           "Method & " + " & ".join(r"AUROC$\uparrow$ & Brier$\downarrow$" for _ in models)
           + r" & AUROC$\uparrow$ & Brier$\downarrow$ & ECE$\downarrow$ \\"]
    for title, group in SUMMARY_GROUPS:
        present = [r for r in group if r in table]
        if not present:
            continue
        out += [r"\midrule", rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for r in present:
            k = rows.index(r)
            shade = r"\rowcolor{capebg}" if r == "CAPE" else ""
            out.append(shade + name(r) + " & " + " & ".join(mark(v, rank[j]) for j, v in enumerate(arr[k])) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def _metric_table(metric: str) -> str:
    """One metric for every method, backbone and test set; best per cell in bold (ties at the printed precision
    included). Low-parse pairs are in parentheses and not bolded."""
    frames = metric_frames()
    entries, counts = significance()
    tests = {(model, dataset): (r if metric == "auroc" else rb)
             for model, dataset, _, r, _, rb in entries}
    threshold = counts["thr"]["p" if metric == "auroc" else "p_brier"]
    sets = [d for d in DATASETS if any(d in set(f["dataset"]) for f in frames.values())]
    head = " & ".join(DATASETS[d] for d in sets)
    out = [r"\begin{tabular}{ll" + "c" * len(sets) + "}", r"\toprule",
           rf"Backbone & Method & {head} \\", r"\midrule"]
    for model, df in frames.items():
        piv = df.pivot_table(index="method", columns="dataset", values=metric)
        rows = [r for r in ROWS if r in piv.index]
        n_rows = len(rows)
        for i, method in enumerate(rows):
            cells = []
            for d in sets:
                if d not in piv.columns or np.isnan(piv.loc[method, d]):
                    cells.append("--")
                    continue
                col = piv.loc[rows, d]
                best = col.max() if HIGHER[metric] else col.min()
                v = piv.loc[method, d]
                if d in low_parse(model):
                    cells.append(f"({v:.3f})")
                else:
                    cell = rf"\textbf{{{v:.3f}}}" if f"{v:.3f}" == f"{best:.3f}" else f"{v:.3f}"
                    if method == "CAPE" and d != "math_all":
                        test = tests[model, d]
                        p = 2 * min(test["p_le0"], 1 - test["p_le0"])
                        if p <= threshold:
                            cell += r"$^\ast$"
                    cells.append(cell)
            label = r"\textbf{\m{} (ours)}" if method == "CAPE" else name(method)
            first = rf"\multirow{{{n_rows}}}{{*}}{{{BACKBONES[model]}}}" if i == 0 else ""
            if method == "CAPE":
                out.append(r"\cmidrule{2-" + str(len(sets) + 2) + "}")
            out.append(f"{first} & {label} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out)


def auroc() -> str:
    """AUROC of every method on every (backbone, test set) pair."""
    return _metric_table("auroc")


def brier() -> str:
    """Brier score of every method on every (backbone, test set) pair."""
    return _metric_table("brier")


TABLES = (summary_full, auroc, brier)
