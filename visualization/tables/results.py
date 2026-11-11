"""Tables of the Results section: the main comparison and the black-box transfer to Claude Haiku 5.5."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR, run_dir
from uqbench.reporting.results import CLEAN_SETS, POOL_SETS, low_parse, metric_frames
from visualization.tables.fmt import mark, name, ranked

# Projection rows mirror the single-response rows; multi-response methods are already coherent (projection leaves
# them unchanged). The feature-matched fusions (DeepConf×SC, Combiner) are in the ablation table.
COMPACT_ROWS = [("Single response", ["DeepConf", "SEP", "ReasonProbe", "ORM"]),
                ("Multiple responses", ["SC", "DeepConf-Vote", "SemEnergy"]),
                ("Published + our projection", ["DeepConf+C", "SEP+Coh", "ReasonProbe+Coh", "ORM+Coh"]),
                ("Ours", ["CAPE"])]
NAMES = {"CAPE": r"\m{}", "SC": "SC", "SE": r"SE\smash{$^\dagger$}"}   # Table 1 overrides of fmt.name


def summary_long(frames: dict) -> pd.DataFrame:
    """Per (backbone, benchmark, method) AUROC / Brier / ECE on high-parse pairs, including projected DeepConf."""
    rows = []
    for m, df in frames.items():
        sets = [d for d in df.dataset.unique() if d != "math_all" and d not in low_parse(m)]
        rows.append(df[df.dataset.isin(sets)][["method", "dataset", "auroc", "brier", "ece"]].assign(model=m))
        path = RESULTS_DIR / m / "projection_steps.csv"
        if path.exists():
            e = pd.read_csv(path)
            e = e[(e.method == "DeepConf") & e.dataset.isin(sets)]
            rows.append(pd.DataFrame({"method": "DeepConf+C", "dataset": e.dataset, "auroc": e.auroc_coh,
                                      "brier": e.brier_coh, "ece": e.ece_coh, "model": m}))
    return pd.concat(rows)


def summary() -> str:
    """Representative methods: mean over the in-distribution pairs (sources in the training pool), the
    out-of-distribution pairs (CLEAN_SETS: GPQA and SuperGPQA) and all pairs. Bold: best; underline: second best."""
    long = summary_long(metric_frames())
    metrics = ("auroc", "brier", "ece")
    blocks = [("ID", POOL_SETS), ("OOD", CLEAN_SETS), ("All", None)]
    rows = [r for _, g in COMPACT_ROWS for r in g]
    vals = {}
    for r in rows:
        d = long[long.method == r]
        v = []
        for _, sets in blocks:
            dd = d if sets is None else d[d.dataset.isin(sets)]
            v += [dd[k].mean() for k in metrics]
        vals[r] = v
    arr = np.array([vals[r] for r in rows])
    higher = [True, False, False] * len(blocks)
    rank = [ranked(arr[:, j], hi) for j, hi in enumerate(higher)]
    n = len(metrics) * len(blocks)
    out = [r"\begin{tabular}{@{}l" + "c" * n + "@{}}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{title}}}" for title, _ in blocks) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(blocks))),
           "Method & " + " & ".join(r"AUROC$\uparrow$ & Brier$\downarrow$ & ECE$\downarrow$" for _ in blocks)
           + r" \\"]
    for title, group in COMPACT_ROWS:
        out += [r"\midrule", rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for r in group:
            shade = r"\rowcolor{capebg}" if r == "CAPE" else ""
            out.append(shade + NAMES.get(r, name(r)) + " & "
                       + " & ".join(mark(v, rank[j]) for j, v in enumerate(vals[r])) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# Black-box rows: the methods computable from an API's text output (-m uqbench.blackbox.evaluate)
BLACKBOX_MODEL = "claude-haiku-5-5"
BLACKBOX_ROWS = [("Single response", ["LRMVerbal", "ORM"]),
                 ("Multiple responses", ["SC", "SE", "ConsistCal", "SNNE"]),
                 ("Published + our projection", ["LRMVerbal+Coh", "ORM+Coh", "SNNE+Coh"]),
                 ("Ours", ["CAPE (votes + ∅)", "CAPE (+ verbal)"])]
BLACKBOX_NAMES = {"CAPE (+ verbal)": r"\m{} (verbal evidence)", "CAPE (votes + ∅)": r"\m{} (no evidence)",
                  "SE": r"SE\smash{$^\dagger$}"}


def _blackbox(blocks) -> str:
    """Rows of BLACKBOX_ROWS, mean AUROC / Brier / ECE of the Haiku run over each block of test sets."""
    df = pd.read_csv(run_dir(BLACKBOX_MODEL) / "results" / "blackbox_metrics.csv")
    metrics = ("auroc", "brier", "ece")
    groups = [(title, [r for r in g if r in set(df.method)]) for title, g in BLACKBOX_ROWS]
    rows = [r for _, g in groups for r in g]
    vals = {}
    for r in rows:
        d = df[df.method == r]
        vals[r] = [(d if sets is None else d[d.dataset.isin(sets)])[k].mean() for _, sets in blocks for k in metrics]
    arr = np.array([vals[r] for r in rows])
    rank = [ranked(arr[:, j], hi) for j, hi in enumerate([True, False, False] * len(blocks))]
    n = len(metrics) * len(blocks)
    head = r"AUROC$\uparrow$ & Brier$\downarrow$ & ECE$\downarrow$"
    out = [r"\begin{tabular}{@{}l" + "c" * n + "@{}}", r"\toprule"]
    if len(blocks) > 1:
        out += [" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{title}}}" for title, _ in blocks) + r" \\",
                "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(blocks)))]
    out.append("Method & " + " & ".join(head for _ in blocks) + r" \\")
    for title, group in groups:
        out += [r"\midrule", rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for r in group:
            shade = r"\rowcolor{capebg}" if r == "CAPE (+ verbal)" else ""
            out.append(shade + BLACKBOX_NAMES.get(r, name(r)) + " & "
                       + " & ".join(mark(v, rank[j]) for j, v in enumerate(vals[r])) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def blackbox_full() -> str:
    r"""Claude Haiku 5.5 through the API (no log-probabilities or hidden states): the methods computable from the
    responses' text, by block (ID, OOD, all test sets); \m's evidence is the verbal confidence."""
    return _blackbox([("ID", POOL_SETS), ("OOD", CLEAN_SETS), ("All", None)])


TABLES = (summary, blackbox_full)
