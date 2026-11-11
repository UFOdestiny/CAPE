"""Appendix: forecasting "none correct" (by question stratum), the decisions it enables, information beyond
calibration, and consistent errors with the verifier audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import (
    BACKBONES,
    DATASETS,
    SINGLE_TRACE,
    escalation_recall,
    frames_of,
    low_parse,
    metric_frames,
    unanimous_confidence,
    with_null_signals,
)
from visualization.data import attribution_frames
from visualization.tables.fmt import mark, name, paren, ranked


def information() -> str:
    """Refinement (cross-fitted isotonic Brier on the test set) and the "none correct" detector
    (``uqbench.studies.refinement``), per backbone and mean over high-parse test sets."""
    ref, nul = frames_of("refinement"), frames_of("null_signal")
    if ref.empty or nul.empty:
        return ""
    show = ["SC", "CISC", "DeepConf-Vote", "CoCoA", "SEP", "DeepConf", "DeepConf×SC", "Stacker", "CAPE"]
    names = {"SC": "SC (calibrated)", "Stacker": "Combiner"}
    out = [r"\begin{tabular}{@{}l" + "c" * (2 * len(BACKBONES) + 2) + "@{}}", r"\toprule",
           r" & \multicolumn{" + str(len(BACKBONES) + 1) + r"}{c}{Brier after isotonic recalibration $\downarrow$}"
           r" & \multicolumn{"
           + str(len(BACKBONES) + 1) + r"}{c}{``None correct'' AUROC $\uparrow$} \\",
           r"\cmidrule(lr){2-" + str(len(BACKBONES) + 2) + r"}\cmidrule(lr){" + str(len(BACKBONES) + 3) + "-"
           + str(2 * len(BACKBONES) + 3) + "}",
           "Method & " + " & ".join(list(BACKBONES.values()) + ["mean"]) + " & "
           + " & ".join(list(BACKBONES.values()) + ["mean"]) + r" \\", r"\midrule"]
    rmean = {m: ref.groupby("model")[m].mean() for m in show}
    nmean = {m: nul.groupby("model")[m if m == "CAPE" else f"{m}+Coh"].mean() for m in show}
    nmean_top = nul.groupby("model")["1 - top share"].mean()
    best_r = {mod: min(rmean[m][mod] for m in show) for mod in BACKBONES}
    best_n = {mod: max(nmean[m][mod] for m in show) for mod in BACKBONES}

    def cell(v, best, fmt):
        txt = fmt.format(v)
        return rf"\textbf{{{txt}}}" if abs(v - best) < 5e-4 else txt

    for m in show:
        if m == "CAPE":
            out.append(r"\midrule")
        label = r"\textbf{\m{} (ours)}" if m == "CAPE" else names.get(m, name(m))
        best_rm = min(ref[x].mean() for x in show)
        best_nm = max(nul[x if x == "CAPE" else f"{x}+Coh"].mean() for x in show)
        rc = [cell(rmean[m][mod], best_r[mod], "{:.3f}") for mod in BACKBONES]
        rc.append(cell(ref[m].mean(), best_rm, "{:.3f}"))
        nc = [cell(nmean[m][mod], best_n[mod], "{:.2f}") for mod in BACKBONES]
        nc.append(cell(nul[m if m == "CAPE" else f"{m}+Coh"].mean(), best_nm, "{:.2f}"))
        out.append(f"{label} & " + " & ".join(rc + nc) + r" \\")
        if m == "SC":
            out.append(r"$1-$top vote share & " + " & ".join(["--"] * (len(BACKBONES) + 1)) + " & "
                       + " & ".join([f"{nmean_top[mod]:.2f}" for mod in BACKBONES]
                                    + [f"{nul['1 - top share'].mean():.2f}"])
                       + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def consistent_errors() -> str:
    """Per-backbone numbers behind the analyses of the Results section (high-parse test sets)."""
    models = list(metric_frames())
    rows = {k: [] for k in ("Unanimous questions (wrong)", "Unanimous AUROC: votes / best trace score",
                            r"\quad within each test set: best trace score / \m",
                            r"Brier on unanimous questions: SC / \m", "Within-question AUROC: SC / \\m",
                            "Evidence correlation $r$ (correct; wrong)", r"$\kappa(8)$ at the smallest $r$",
                            r"\m confidence: unanimous correct / wrong")}
    for m in models:
        u = pd.read_csv(RESULTS_DIR / m / "unanimous.csv")
        a = u[u.dataset == "all"].set_index("method")
        rows["Unanimous questions (wrong)"].append(f"{int(a.questions.iloc[0]):,} ({int(a.wrong.iloc[0])})")
        best = max(a.loc[x, "auroc"] for x in SINGLE_TRACE if x in a.index)
        rows["Unanimous AUROC: votes / best trace score"].append(f"{a.loc['SC', 'auroc']:.2f} / {best:.2f}")
        if "auroc_within_set" in a:
            bw = max(a.loc[x, "auroc_within_set"] for x in SINGLE_TRACE if x in a.index)
            rows[r"\quad within each test set: best trace score / \m"].append(
                f"{bw:.2f} / {a.loc['CAPE', 'auroc_within_set']:.2f}")
            rows[r"Brier on unanimous questions: SC / \m"].append(
                f"{a.loc['SC', 'brier']:.3f} / {a.loc['CAPE', 'brier']:.3f}")
        else:
            rows[r"\quad within each test set: best trace score / \m"].append("--")
            rows[r"Brier on unanimous questions: SC / \m"].append("--")
        f = pd.read_csv(RESULTS_DIR / m / "within_question.csv")
        f = f[~f.dataset.isin(low_parse(m))].groupby("method").within_auroc.mean()
        rows["Within-question AUROC: SC / \\m"].append(f"{f['SC']:.3f} / {f['CAPE']:.3f}")
        icc = pd.read_csv(RESULTS_DIR / m / "evidence_icc.csv")
        icc = icc[icc.dataset != "pool (val)"]
        rc, rw = icc.icc_correct, icc.icc_wrong
        rows["Evidence correlation $r$ (correct; wrong)"].append(
            f"{rc.min():.2f}--{rc.max():.2f}; {rw.min():.2f}--{rw.max():.2f}")
        rmin = pd.concat([icc.icc_correct, icc.icc_wrong]).min()
        rows[r"$\kappa(8)$ at the smallest $r$"].append(f"{8 / (1 + 7 * rmin):.1f}")
        g = unanimous_confidence(m)
        rows[r"\m confidence: unanimous correct / wrong"].append(
            f"{g['unanimous_correct']:.2f} / {g['unanimous_wrong']:.2f}")
    out = [r"\begin{tabular}{@{}l" + "c" * len(models) + "@{}}", r"\toprule",
           " & " + " & ".join(BACKBONES[m] for m in models) + r" \\", r"\midrule"]
    out += [f"{k} & " + " & ".join(v) + r" \\" for k, v in rows.items()]
    out += _audit_rows(models)
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def _audit_rows(models: list[str]) -> list[str]:
    """Consistent errors that a looser checker accepts (``uqbench.studies.audit``), per backbone and audited test
    set, and
    the unanimous AUROC with those questions removed."""
    audits = {m: pd.read_csv(RESULTS_DIR / m / "verifier_audit.csv").set_index("dataset") for m in models
              if (RESULTS_DIR / m / "verifier_audit.csv").exists()}
    if len(audits) < len(models):
        return []
    out = [r"\midrule", rf"\multicolumn{{{1 + len(models)}}}{{@{{}}l}}{{\emph{{Verifier audit: unanimous-wrong "
           r"questions a looser verifier accepts}} \\"]
    audited = [d for d in DATASETS if any(d in a.index for a in audits.values())]   # math and short-answer sets
    for d in audited:
        cells = [paren(f"{int(a.loc[d, 'flag_tol5'])}/{int(a.loc[d, 'wrong'])}", d in low_parse(m))
                 if d in a.index else "--" for m, a in audits.items()]
        out.append(rf"\quad {DATASETS[d]} & " + " & ".join(cells) + r" \\")
    for t, label in (("all", r"Unanimous AUROC here: \m{} / best single"),
                     ("all_clean", r"\quad accepted questions removed")):
        out.append(f"{label} & " + " & ".join(f"{a.loc[t, 'auroc_cape']:.2f} / {a.loc[t, 'auroc_best_single']:.2f}"
                                               for a in audits.values()) + r" \\")
    return out


NONE_CORRECT_ROWS = (
    ("CAPE", r"\m, learned $\pi_\nul$"),
    ("coherent stacker, leftover", r"Comb.+probe$+\C$, residual"),
    ("coherent stacker, 1 - top", r"Comb.+probe$+\C$, $1-$top"),
    ("DeepConf +C, leftover", r"DeepConf $+\C$, residual"), ("SC, 1 - top", r"SC, $1-$top"))


ESC_COLUMN = {"CAPE": "cape_null", "coherent stacker, leftover": "stacker_null",
              "coherent stacker, 1 - top": "stack_top", "DeepConf +C, leftover": "deepconf_null",
              "SC, 1 - top": "vote"}   # predictor -> its score in questions.csv (escalation at 20%)
def none_correct_full() -> str:
    """Forecasting "none correct" by question stratum (means over the high-parse pairs): the event detected
    (AUROC) and forecast (Brier score) by each predictor on all, unanimous and split questions, the share of split
    questions on which it is zero, and its mean mass against the event's rate by the number of answers."""
    nul = attribution_frames()["null"]
    if nul.empty:
        return "% attribution pending"
    g = nul.groupby(["predictor", "stratum"])[["auroc", "brier", "zero_mass", "mean_mass", "none_rate"]].mean()
    q = with_null_signals(frames_of("questions"))
    esc = {k: 100 * np.mean(escalation_recall(q, c, 0.2)) for k, c in ESC_COLUMN.items() if c in q}
    rate = {m: g.loc[("CAPE", m), "none_rate"] for m in ("m1", "m2", "m3+")}
    keys = [k for k, _ in NONE_CORRECT_ROWS if (k, "split") in g.index]
    cols = [("all", "auroc", True), ("all", "brier", False), ("unanimous", "auroc", True),
            ("unanimous", "brier", False), ("split", "auroc", True), ("split", "brier", False)]
    rank = [ranked([g.loc[(k, st), c] for k in keys], hi) for st, c, hi in cols]
    out = [r"\begin{tabular}{@{}l" + "c" * 11 + "@{}}", r"\toprule",
           r" & \multicolumn{3}{c}{All questions} & \multicolumn{2}{c}{Unanimous} & \multicolumn{3}{c}{Non-unanimous}"
           r" & \multicolumn{3}{c}{Mean probability by answer count} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
           r"Predictor & AUROC$\uparrow$ & Brier$\downarrow$"
           r" & \shortstack{Recall$\uparrow$\\(20\% deferred)} & AUROC$\uparrow$ & "
           r"Brier$\downarrow$ & AUROC$\uparrow$ & Brier$\downarrow$ & zero (\%) & 1 & 2 & 3+ \\", r"\midrule",
           r"\emph{Rate of $\nul$} & & & & & & & & & " + " & ".join(f"{rate[m]:.2f}" for m in rate) + r" \\",
           r"\midrule"]
    for key, label in NONE_CORRECT_ROWS:
        if key not in keys:
            continue
        cells = [mark(g.loc[(key, st), c], rank[j]) for j, (st, c, _) in enumerate(cols)]
        cells[2:2] = [f"{esc[key]:.0f}" if key in esc else "--"]
        cells.append(f"{100 * g.loc[(key, 'split'), 'zero_mass']:.0f}")
        cells += [f"{g.loc[(key, m), 'mean_mass']:.2f}" for m in rate]
        shade = r"\rowcolor{capebg}" if key == "CAPE" else ""
        out.append(f"{shade}{label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


NONE_ROWS = (("SC, 1 - top", "SC", r"1$-$top"), ("DeepConf +C, leftover", r"DC $+\C$", "residual"),
             ("coherent stacker, leftover", r"Comb.+probe$+\C$", "residual"),
             ("coherent stacker, 1 - top", "", r"1$-$top"), ("CAPE", r"\textbf{\m}", r"$\pi_\nul$"))


def none_correct() -> str:
    """Forecasting "none correct" (no sampled answer is correct) on all, unanimous and split questions: AUROC and
    Brier score of the event per estimator and scoring rule, means over pairs. Bold: best per column."""
    nul = attribution_frames()["null"]
    if nul.empty:
        return "% none-correct summary pending"
    cols = (("all", "auroc"), ("all", "brier"), ("unanimous", "brier"), ("split", "brier"))
    vals = {p: [nul[(nul.stratum == s) & (nul.predictor == p)][m].mean() for s, m in cols] for p, _, _ in NONE_ROWS}
    best = [max(v[j] for v in vals.values()) if m == "auroc" else min(v[j] for v in vals.values())
            for j, (_, m) in enumerate(cols)]
    out = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
           r" & & \multicolumn{2}{c}{All} & Unan. & Other \\", r"\cmidrule(lr){3-4}",
           r"Estimator & $\nul$ score & AUROC$\uparrow$ & Brier$\downarrow$ & Brier$\downarrow$ & Brier$\downarrow$ \\",
           r"\midrule"]
    for p, label, rule in NONE_ROWS:
        cells = []
        for j, v in enumerate(vals[p]):
            txt = f"{v:.3f}"
            cells.append(rf"\textbf{{{txt}}}" if txt == f"{best[j]:.3f}" else txt)
        shade = r"\rowcolor{capebg}" if p == "CAPE" else ""
        if p == "CAPE":
            out.append(r"\midrule")
        out.append(shade + label + " & " + rule + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


TABLES = (none_correct_full, information, consistent_errors, none_correct)
