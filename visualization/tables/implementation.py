"""Appendix: implementation details (test-set statistics, CAPE's fitted weights, inference cost)."""
from __future__ import annotations

import pandas as pd

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import BACKBONES, DATASETS, low_parse, metric_frames, token_stats
from visualization.tables.fmt import paren


def data() -> str:
    """Test-set statistics per backbone: accuracy, answer clusters per question, unanimous-wrong rate (low-parse
    pairs in parentheses)."""
    sets = [d for d in DATASETS if d != "math_all"]
    out = [r"\begin{tabular}{ll" + "c" * len(sets) + "}", r"\toprule",
           "Backbone & Statistic & " + " & ".join(DATASETS[d] for d in sets) + r" \\", r"\midrule"]
    frames = metric_frames()
    n = next(iter(frames.values())).groupby("dataset")["n"].first() / 8   # K = 8 responses per question
    out += [" & Questions & " + " & ".join(f"{n[d]:,.0f}".replace(",", "{,}") for d in sets) + r" \\", r"\midrule"]
    for model, df in frames.items():
        st = pd.read_csv(RESULTS_DIR / model / "answer_structure.csv").set_index("dataset")
        acc = df.groupby("dataset")["acc"].first()

        def cell(d, col, fmt, st=st, model=model):
            return paren(format(st.loc[d, col], fmt), d in low_parse(model)) if d in st.index else "--"

        stats = [("Accuracy", lambda d, acc=acc, model=model: paren(f"{acc[d]:.3f}", d in low_parse(model))
                  if d in acc else "--"),
                 ("Clusters per question", lambda d: cell(d, "clusters_per_q", ".2f")),
                 ("Unanimous wrong", lambda d: cell(d, "unanimous_wrong_rate", ".3f")),
                 ("Unanswered responses", lambda d: cell(d, "no_answer_rate", ".3f")),
                 ("Two correct clusters", lambda d: cell(d, "gold_split_rate", ".3f"))]
        for i, (label, fn) in enumerate(stats):
            first = rf"\multirow{{{len(stats)}}}{{*}}{{{BACKBONES[model]}}}" if i == 0 else ""
            out.append(f"{first} & {label} & " + " & ".join(fn(d) for d in sets) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out)


def cost(k: int = 8) -> str:
    """Per-question inference cost in thousands of tokens for the same K shared responses: the decoded tokens, which
    every method shares, and each method's extra teacher-forced (prefill) or prefix-cached tokens, from measured
    mean lengths."""
    frames = metric_frames()
    per_response = {  # (replay passes, short continuations) per scored response
        "DeepConf, CDG, Self-Cert.": (1, 0), "Verbal confidence, P(True)": (0, 1), "Correctness probe": (1, 0),
        "SC, SE, SNNE": (0, 0), "CISC": (0, 1), "\\m (ours)": (1, 0)}
    stats = {m: token_stats(m) for m in frames}
    out = [r"\begin{tabular}{@{}l" + "r" * len(frames) + "@{}}", r"\toprule",
           "Tokens & " + " & ".join(BACKBONES[m] for m in frames) + r" \\", r"\midrule",
           r"Decoding, all methods & "
           + " & ".join(f"{k * stats[m]['response'] / 1e3:.1f}k" for m in frames) + r" \\", r"\midrule",
           rf"\multicolumn{{{1 + len(frames)}}}{{@{{}}l}}"
           r"{\emph{Extra prefill or prefix-cached tokens}} \\"]
    for name, (replay, cont) in per_response.items():
        cells = [f"{k * (replay * (stats[m]['prompt'] + stats[m]['response']) + cont * 30) / 1e3:.1f}k" for m in frames]
        out.append(name + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


TABLES = (data, cost)
