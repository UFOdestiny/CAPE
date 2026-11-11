"""Ablations of CAPE's components and the clustered self-assessment adaptation."""
from __future__ import annotations

import pandas as pd

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import BACKBONES, low_parse, metric_frames
from visualization.tables.fmt import paren

ABLATION_ROWS = [("CAPE", r"\m"), ("CAPE w/o null", r"w/o $\nul$ outcome"),
                 ("CAPE w/o votes", r"w/o vote prior"), ("CAPE w/o probe", r"w/o hidden states")]
# Controls that combine baseline trace and vote features.
CONTROL_ROWS = [("DeepConf×SC", r"DeepConf$\times$SC"), ("Stacker", "Combiner")]
# Model extensions (``uqbench.studies.ablation --studies forms``)
FORM_ROWS = [("learned r", r"learned correlation $r$"), ("question-dependent null", r"question-dependent $\nul$ logit")]
JUDGE_SETS = [("math_all", "Math"), ("gpqa", "GPQA"), ("mmlupro", "MMLU-Pro"), ("supergpqa", "SuperGPQA"),
              ("triviaqa", "TriviaQA")]


def ablations() -> str:
    """CAPE without each component, model extensions, and controls that combine baseline features:
    mean AUROC, Brier score and ECE over all high-parse pairs. Bold: best per column."""
    rows = []
    frames = metric_frames()
    for model, df in frames.items():
        sets = [d for d in df.dataset.unique() if d != "math_all" and d not in low_parse(model)]
        rows.append(df[df.method.isin([m for m, _ in ABLATION_ROWS + CONTROL_ROWS]) & df.dataset.isin(sets)]
                    [["method", "dataset", "auroc", "brier", "ece"]])
        forms = RESULTS_DIR / model / "ablation_forms.csv"
        if forms.exists():
            f = pd.read_csv(forms).rename(columns={"variant": "method"})
            rows.append(f[f.method.isin([m for m, _ in FORM_ROWS]) & f.dataset.isin(sets)]
                        [["method", "dataset", "auroc", "brier", "ece"]])
    mean = pd.concat(rows).groupby("method")[["auroc", "brier", "ece"]].mean()
    best = {"auroc": mean.auroc.max(), "brier": mean.brier.min(), "ece": mean.ece.min()}
    out = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"Variant & AUROC$\uparrow$ & Brier$\downarrow$ & ECE$\downarrow$ \\", r"\midrule"]
    blocks = [("", ABLATION_ROWS), ("Model extensions", [r for r in FORM_ROWS if r[0] in mean.index]),
              ("Combinations of baseline signals", CONTROL_ROWS)]
    for title, members in blocks:
        if not members:
            continue
        if title:
            out += [r"\midrule", rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for meth, label in members:
            cells = [rf"\textbf{{{mean.loc[meth, k]:.3f}}}" if f"{mean.loc[meth, k]:.3f}" == f"{best[k]:.3f}"
                     else f"{mean.loc[meth, k]:.3f}" for k in ("auroc", "brier", "ece")]
            shade = r"\rowcolor{capebg}" if meth == "CAPE" else ""
            out.append(shade + label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def backbone_ablation(name: str) -> str:
    """An ablation run on every reported backbone (``ablation_<study>.csv``): one row per variant."""
    frames = {m: pd.read_csv(RESULTS_DIR / m / name) for m in BACKBONES if (RESULTS_DIR / m / name).exists()}
    if not frames:
        return f"% {name} pending"
    out = [r"\begin{tabular}{ll" + "cc" * len(JUDGE_SETS) + "}", r"\toprule",
           "Backbone & Variant & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in JUDGE_SETS) + r" \\",
           " & & " + " & ".join("AUROC & Brier" for _ in JUDGE_SETS) + r" \\", r"\midrule"]
    for i, (model, df) in enumerate(frames.items()):
        if i:
            out.append(r"\midrule")
        for j, (_, r) in enumerate(df.iterrows()):
            cells = [paren(f"{r[f'{d}_auroc']:.3f}", d in low_parse(model)) + " & "
                     + paren(f"{r[f'{d}_brier']:.3f}", d in low_parse(model)) for d, _ in JUDGE_SETS]
            label = (r["variant"].replace("CAPE (format x residual gate)", r"\m{} (shift gate)")
                     .replace("CAPE", r"\m{}").replace("robust model", "trace model")
                     .replace(" (clustered self-assessment style)", "")
                     .replace("reasoning judge alone", "Self-assessment with reasoning")
                     .replace("judge alone", "Self-assessment")
                     .replace("reasoning judge", "self-assessment with reasoning")
                     .replace("judge", "self-assessment")
                     .replace("format x residual", r"format $\times$ residual").replace("alpha = ", r"$\alpha=$"))
            out.append((BACKBONES[model] if j == 0 else "") + f" & {label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def judge() -> str:
    """Clustered self-assessment as an additional CAPE input, and alone (every backbone)."""
    return backbone_ablation("ablation_judge.csv")


TABLES = (ablations, judge)
