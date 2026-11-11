"""Appendix: robustness of the main comparison, question-level decisions (``uqbench.studies.robustness``,
``uqbench.studies.abstention``) and grader-independent clusters (``uqbench.studies.ablation``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import BACKBONES
from visualization.tables.fmt import mark, ranked

SHOWN = [("CAPE", r"\m"), ("SEP+C", r"SEP $+\C$"), ("ReasonProbe+C", r"Correctness probe $+\C$"),
         ("Combiner", "Combiner")]
SCHEMES = [("pairs (unweighted)", "Settings (main)"), ("questions", "Questions"), ("domains (balanced)", "Domains"),
           ("without the lowest-accuracy pair", "w/o lowest-accuracy setting")]
SOURCES = ["MATH", "OlympiadBench", "Minerva", "AIME", "GPQA", "MMLU-Pro", "SuperGPQA", "TriviaQA"]


def robustness() -> str:
    """Mean AUROC / Brier / ECE under other aggregations: weighted by questions, balanced over the three domains,
    and with each source, each backbone or the lowest-accuracy pair left out. Bold: best of the shown methods."""
    path = RESULTS_DIR / "robustness_aggregate.csv"
    if not path.exists():
        return "% robustness pending"
    agg = pd.read_csv(path).set_index(["scheme", "method"])
    schemes = SCHEMES + [(f"without {s}", f"w/o {s}") for s in SOURCES] + \
        [(f"without {m}", f"w/o {BACKBONES[m]}") for m in BACKBONES]
    metrics = (("auroc", True), ("brier", False), ("ece", False))
    out = [r"\begin{tabular}{l" + "c" * 3 * len(SHOWN) + "}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{{len(SHOWN)}}}{{c}}{{{t}}}"
                              for t in (r"AUROC$\uparrow$", r"Brier$\downarrow$", r"ECE$\downarrow$")) + r" \\",
           "Aggregation & " + " & ".join(n for _ in metrics for _, n in SHOWN) + r" \\", r"\midrule"]
    for i, (key, label) in enumerate(schemes):
        if i in (len(SCHEMES), len(SCHEMES) + len(SOURCES)):
            out.append(r"\midrule")
        cells = []
        for k, hi in metrics:
            vals = [agg.loc[(key, m), k] for m, _ in SHOWN]
            top = ranked(vals, hi)[:1]
            cells += [mark(v, top) for v in vals]
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


PAIRED_RIVALS = [("SEP+C", r"SEP $+\C$"), ("ReasonProbe+C", r"Correctness probe $+\C$"), ("Combiner", "Combiner"),
                 ("Sem. Energy", r"Sem.\ Energy"), ("SC", "SC"), ("CAPE w/o null", r"\m w/o $\nul$"),
                 ("CAPE w/o votes", r"\m w/o vote prior"), ("CAPE w/o probe", r"\m w/o hidden states")]


def paired() -> str:
    """\\m's gain in the mean over pairs against each rival (positive favors \\m), with 95% intervals from a
    question bootstrap that resamples each test set jointly across backbones."""
    path = RESULTS_DIR / "robustness_paired.csv"
    if not path.exists():
        return "% paired pending"
    p = pd.read_csv(path).set_index(["group", "rival", "metric"])
    cols = (("All", "auroc"), ("All", "brier"), ("ID", "brier"), ("OOD", "brier"))

    def cell(g, r, k):
        x = p.loc[(g, r, k)]
        star = "$^\\ast$" if x.lo > 0 else ""
        return f"{x.gain:+.4f}{star}", f"[{x.lo:+.4f}, {x.hi:+.4f}]"

    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Rival & $\Delta$AUROC, all & $-\Delta$Brier, all & $-\Delta$Brier, ID & $-\Delta$Brier, OOD \\",
           r"\midrule"]
    for r, label in PAIRED_RIVALS:
        if r == "CAPE w/o null":
            out.append(r"\midrule")
        cells = [cell(g, r, k) for g, k in cols]
        out.append(f"{label} & " + " & ".join(c[0] for c in cells) + r" \\")
        out.append(" & " + " & ".join(r"{\footnotesize " + c[1] + "}" for c in cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


DECIDERS = [("CAPE", r"\m"), ("SEP+C", r"SEP $+\C$"), ("DeepConf+C", r"DeepConf $+\C$"), ("SC", "SC")]


def decisions() -> str:
    """Selective prediction, means over the 30 pairs: accuracy of the most probable answer, AURC and accuracy of the
    most confident 50% / 80% of questions."""
    sel_path = RESULTS_DIR / "robustness_selective.csv"
    if not sel_path.exists():
        return "% decisions pending"
    sel = pd.read_csv(sel_path).groupby("method")[["accuracy", "aurc", "acc_at_50", "acc_at_80"]].mean()
    rows = [(label, [100 * sel.loc[m, k] for k in ("accuracy", "aurc", "acc_at_50", "acc_at_80")])
            for m, label in DECIDERS]
    arr = np.array([v for _, v in rows])
    higher = (True, False, True, True)
    tops = [ranked(arr[:, j], h, 1) for j, h in enumerate(higher)]
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Method & Acc.$\uparrow$ & AURC$\downarrow$ & Acc.@50\%$\uparrow$ & Acc.@80\%$\uparrow$ \\", r"\midrule"]
    for label, vals in rows:
        out.append(f"{label} & " + " & ".join(mark(v, tops[j], 1) for j, v in enumerate(vals)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


ALPHAS = (0.01, 0.02, 0.05)
ANSWERERS = [("CAPE", r"\m"), ("Comb.+C", r"Comb.+probe$+\C$"), ("Modular", r"\quad with $\nul$ classifier"),
             ("ReasonProbe+C", r"Corr.\ probe $+\C$"), ("SEP+C", r"SEP $+\C$"), ("DeepConf+C", r"DeepConf $+\C$"),
             ("SC", "SC"), ("CAPE w/o null", r"\m w/o $\nul$")]


def _short(x: float) -> str:
    """Three decimals without the leading zero, keeping the sign of negative values."""
    return f"{x:.3f}".replace("0.", ".", 1).replace("-", "$-$")


def abstention() -> str:
    r"""Cost-sensitive answering (``uqbench.studies.abstention``): each method answers with its most probable
    answer iff its probability is at least 1 - alpha; mean utility per question over the 30 pairs (correct +1, wrong
    -(1 - alpha)/alpha, abstain 0). Last rows: \m minus the strongest rival shown, with the 95% interval of the joint
    question bootstrap. Bold: best; underlined: second."""
    path, ppath = RESULTS_DIR / "abstention.csv", RESULTS_DIR / "abstention_paired.csv"
    if not path.exists() or not ppath.exists():
        return "% abstention pending"
    df = pd.read_csv(path)
    u = df[df["mode"] == "own"].groupby(["method", "alpha"]).utility.mean()
    pp = pd.read_csv(ppath)
    pp = pp[(pp["mode"] == "own") & (pp.group == "All")].set_index(["rival", "alpha"])
    vals = np.array([[u[(m, a)] for a in ALPHAS] for m, _ in ANSWERERS])
    tops = [ranked(vals[:, j], True) for j in range(len(ALPHAS))]
    out = [r"\begin{tabular}{@{}l" + "c" * len(ALPHAS) + "@{}}", r"\toprule",
           r"Cost of an error & " + " & ".join(f"{(1 - a) / a:.0f}" for a in ALPHAS) + r" \\", r"\midrule"]
    for i, (_, label) in enumerate(ANSWERERS):
        shade = r"\rowcolor{capebg}" if i == 0 else ""
        out.append(f"{shade}{label} & " + " & ".join(mark(v, tops[j]) for j, v in enumerate(vals[i])) + r" \\")
    out.append(r"\midrule")
    gains, intervals = [], []
    for a in ALPHAS:
        rival = max((m for m, _ in ANSWERERS[1:]), key=lambda m: u[(m, a)])
        x = pp.loc[(rival, a)]
        gains.append(f"{x.gain:+.3f}")
        intervals.append(rf"{{\scriptsize [{_short(x.lo)}, {_short(x.hi)}]}}")
    out.append(r"\m $-$ best rival & " + " & ".join(gains) + r" \\")
    out.append(" & " + " & ".join(intervals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


RULES = [("grader", "Main clustering"), ("independent, LaTeX-normalised", "String (math) / NLI (TriviaQA)")]
CLUSTERED = [("CAPE", r"\m"), ("ReasonProbe", r"Correctness probe $+\C$"), ("DeepConf", r"DeepConf $+\C$"),
             ("SC", "SC")]
PROJECTED = ("ReasonProbe", "DeepConf")   # calibrated probabilities projected onto each set of clusters


def equivalence() -> str:
    r"""Clusters from a grader-independent equivalence (``uqbench.studies.ablation --studies equivalence``): math
    answers by
    exact string after LaTeX surface normalization, TriviaQA answers by bidirectional NLI entailment; labels
    unchanged. \m and SC are refitted on the re-clustered pool; DeepConf's calibrated probabilities are projected
    onto the new clusters. Means over the math pairs (15) and the TriviaQA pairs (3). Bold: best per block."""
    frames = [(m, RESULTS_DIR / m / "ablation_equivalence.csv", RESULTS_DIR / m / "ablation_equivalence_projection.csv")
              for m in BACKBONES]
    if not all(a.exists() and b.exists() for _, a, b in frames):
        return "% equivalence pending"
    resp = pd.concat([pd.read_csv(a).assign(model=m) for m, a, _ in frames])
    proj = pd.concat([pd.read_csv(b).assign(model=m) for m, _, b in frames])
    resp, proj = resp[resp.dataset != "math_all"], proj[proj.dataset != "math_all"]
    group = lambda d: np.where(d == "triviaqa", "qa", "math")  # noqa: E731
    resp, proj = resp.assign(g=group(resp.dataset)), proj.assign(g=group(proj.dataset))
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{Math} & \multicolumn{2}{c}{TriviaQA} \\",
           r"Clusters & Method & AUROC$\uparrow$ & Brier$\downarrow$ & AUROC$\uparrow$ & Brier$\downarrow$ \\",
           r"\midrule"]
    for i, (rule, title) in enumerate(RULES):
        if i:
            out.append(r"\midrule")
        vals = []
        for m, _ in CLUSTERED:
            if m in PROJECTED:
                x = proj[(proj.rule == rule) & (proj.method == m)].groupby("g")[["auroc_coh", "brier_coh"]].mean()
                x.columns = ["auroc", "brier"]
            else:
                x = resp[(resp.rule == rule) & (resp.variant == m)].groupby("g")[["auroc", "brier"]].mean()
            vals.append([x.loc["math", "auroc"], x.loc["math", "brier"], x.loc["qa", "auroc"], x.loc["qa", "brier"]])
        arr = np.array(vals)
        tops = [ranked(arr[:, j], j % 2 == 0)[:1] for j in range(4)]
        for j, (_, label) in enumerate(CLUSTERED):
            lead = title if j == 0 else ""
            out.append(f"{lead} & {label} & " + " & ".join(mark(v, tops[k]) for k, v in enumerate(arr[j])) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


DEPENDENCE_METHODS = [("DeepConf", "DeepConf"), ("ReasonProbe", "Correctness probe"), ("SEP", "SEP"), ("SC", "SC"),
                      ("SemEnergy", r"Sem.\ Energy"), ("DeepConf×SC", r"DeepConf$\times$SC"),
                      ("CAPE w/o null", r"\m w/o $\nul$"), ("CAPE", r"\m")]


def dependence() -> str:
    """Shared-prefix intervention on Qwen3-8B (``uqbench.studies.dependence``): structure of the K responses, mean
    Brier score of each method, the extra change without the outside option (95% question-bootstrap interval) and
    the refitted vote coefficient and "none correct" score, per shared fraction f. Means over the six test sets."""
    root = RESULTS_DIR / "qwen3-8b"
    if not (root / "dependence_bootstrap.csv").exists():
        return "% dependence pending"
    st = pd.read_csv(root / "dependence_structure.csv").groupby("fraction").mean(numeric_only=True)
    me = pd.read_csv(root / "dependence_methods.csv").groupby(["method", "fraction"])[["auroc", "brier"]].mean()
    bs = pd.read_csv(root / "dependence_bootstrap.csv")
    pa = pd.read_csv(root / "dependence_refit_params.csv")
    pa["theta"] = pa.theta_refit.str.strip("[]").str.split(",").str[0].astype(float)
    pa = pa.groupby("fraction")[["alpha_only", "theta"]].agg(["mean", "std"])
    fracs = list(st.index)
    out = [r"\begin{tabular}{l" + "c" * len(fracs) + "}", r"\toprule",
           r"Shared fraction $f$ & " + " & ".join(f"{f:g}" for f in fracs) + r" \\", r"\midrule",
           r"\multicolumn{" + str(len(fracs) + 1) + r"}{l}{\emph{Responses}} \\"]
    for key, name in (("accuracy", "Accuracy"), ("icc", "Correctness ICC"), ("none_rate", r"``None correct'' rate"),
                      ("unanimous_error", "Consistent-error rate"), ("distinct_answers", "Distinct answers")):
        out.append(f"{name} & " + " & ".join(f"{st.loc[f, key]:.3f}" for f in fracs) + r" \\")
    out += [r"\midrule", r"\multicolumn{" + str(len(fracs) + 1) + r"}{l}{\emph{Brier score}} \\"]
    tops = {f: ranked([me.loc[(k, f), "brier"] for k, _ in DEPENDENCE_METHODS], False) for f in fracs}
    for key, name in DEPENDENCE_METHODS:
        out.append(f"{name} & " + " & ".join(mark(me.loc[(key, f), "brier"], tops[f]) for f in fracs) + r" \\")
    out += [r"\midrule", r"\multicolumn{" + str(len(fracs) + 1) + r"}{l}{\emph{Extra change of \m w/o $\nul$ "
            r"over \m}} \\"]
    extra = bs[bs.kind == "change: w/o null minus CAPE"].set_index(["metric", "fraction"])
    for metric, name in (("auroc", "AUROC"), ("brier", "Brier score")):
        cells = ["--"] + [f"{extra.loc[(metric, f), 'estimate']:+.3f}" for f in fracs[1:]]
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
        out.append(" & & " + " & ".join(rf"\scriptsize[{extra.loc[(metric, f), 'lo']:+.3f}, "
                                        rf"{extra.loc[(metric, f), 'hi']:+.3f}]" for f in fracs[1:]) + r" \\")
    out += [r"\midrule", r"\multicolumn{" + str(len(fracs) + 1) + r"}{l}{\emph{Refitted \m}} \\"]
    for key, name in (("alpha_only", "Coefficient of $\\log(n_c/K)$"), ("theta", r"``None correct'' score $\theta_0$")):
        out.append(f"{name} & " + " & ".join(f"{pa.loc[f, (key, 'mean')]:.2f}$\\pm${pa.loc[f, (key, 'std')]:.2f}"
                                            for f in fracs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


TABLES = (robustness, paired, decisions, abstention, equivalence, dependence)
