"""Appendix: projection onto the coherent set (step by step, and per estimator with its question-level safety)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import BACKBONES, COMPETITORS, frames_of, low_parse, metric_frames
from visualization.data import PUBLISHED, attribution_frames
from visualization.tables.fmt import name


def projsteps() -> str:
    """The projection step by step, applied cumulatively to the calibrated probabilities of the estimators whose
    outputs are not coherent, against the product with the vote share and a constant score (means over estimators
    and high-parse pairs)."""
    st = frames_of("projection_steps")
    if st.empty:
        return "% projection steps pending"
    base = st[~st.method.str.contains(" x share") & (st.method != "constant")]
    mean = base[[c for c in base.columns if c.startswith(("auroc_", "brier_"))]].mean()
    prod = st[st.method.str.contains(" x share")][["auroc_coh", "brier_coh"]].mean()
    const = st[st.method == "constant"][["auroc_coh", "brier_coh"]].mean()
    rows = [("Calibrated probabilities", "prob"), ("$+$ zero unanswered responses", "zero"),
            ("$+$ average within clusters (A1)", "avg"), ("$+$ sum constraint $\\sum_c\\pi_c\\le1$ (A2)", "coh")]
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Projection step & AUROC$\uparrow$ & Brier$\downarrow$ \\",
             r"\midrule"]
    lines += [f"{label} & {mean['auroc_' + k]:.3f} & {mean['brier_' + k]:.3f} \\\\" for label, k in rows]
    lines += [r"\midrule",
              f"Calibrated $\\times$ vote share & {prod.auroc_coh:.3f} & {prod.brier_coh:.3f} \\\\",
              f"Constant score, projected & {const.auroc_coh:.3f} & {const.brier_coh:.3f} \\\\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _effect(model: str) -> pd.DataFrame:
    """Calibrated vs. projected probabilities of every estimator on the high-parse pairs of a backbone."""
    path = RESULTS_DIR / model / "coherence_effect.csv"
    if not path.exists():
        return pd.DataFrame()
    e = pd.read_csv(path)
    return e[(e.dataset != "math_all") & ~e.dataset.isin(low_parse(model))]


def projection() -> str:
    """Per estimator on the eight test sets: calibrated vs. projected mean AUROC and Brier score over the
    high-parse pairs, the pairs on which projection raises AUROC / lowers Brier, CAPE's wins against the projected
    estimator, and question by question the share of the realized Brier gain certified without labels, the share of
    questions on which the sum constraint binds, and the share of questions with (A1)-(A2) labels whose Brier score
    the projection and the product with the vote share raise, with the product's largest increase. Last row: mean
    over the incoherent published estimators."""
    rows, n_pairs = {}, 0
    for m, df in metric_frames().items():
        e = _effect(m)
        cape = df[df.method == "CAPE"].set_index("dataset")
        n_pairs += e.dataset.nunique()
        for r in e.itertuples():
            rows.setdefault(r.method, []).append((r.auroc_prob, r.auroc_coh, r.brier_prob, r.brier_coh,
                                                  cape.loc[r.dataset, "auroc"], cape.loc[r.dataset, "brier"]))
    safety = attribution_frames()["safety"]
    if not rows or safety.empty:
        return "% projection pending"
    cols = ("certified_share", "cap_active", "proj_harmed", "prod_harmed")

    def safe(d):
        if d.empty:
            return ["--"] * 5
        return [f"{100 * d[c].mean():.0f}" for c in cols] + [f"{d.prod_worst.max():.2f}"]

    out = [r"\begin{tabular}{@{}lcccccccccccc@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{Brier} & \multicolumn{2}{c}{Settings improved}"
           r" & \multicolumn{2}{c}{\m{} better} & Bound covers & Sum binds"
           r" & \multicolumn{2}{c}{Brier increased (\%)} & Worst \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){12-13}",
           r"Estimator & cal. & proj. & cal. & proj. & AUROC & Brier & AUROC & Brier & gain (\%) & (\%) & proj."
           r" & $\times$share & increase \\", r"\midrule"]
    out[0] = r"\begin{tabular}{@{}l" + "c" * 13 + "@{}}"
    for meth in [x for x in COMPETITORS if x in rows]:
        v = np.array(rows[meth])
        cells = [f"{v[:, 0].mean():.3f}", f"{v[:, 1].mean():.3f}", f"{v[:, 2].mean():.3f}", f"{v[:, 3].mean():.3f}",
                 f"{int((v[:, 1] > v[:, 0]).sum())}", f"{int((v[:, 3] < v[:, 2]).sum())}",
                 f"{int((v[:, 4] >= v[:, 1]).sum())}", f"{int((v[:, 5] <= v[:, 3]).sum())}"]
        cells += safe(safety[safety.method == meth])
        out.append(name(meth) + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    eff = pd.concat([_effect(m) for m in BACKBONES])
    eff = eff[eff.method.isin(PUBLISHED)] if not eff.empty else eff
    if not eff.empty:
        cells = [f"{eff[c].mean():.3f}" for c in ("auroc_prob", "auroc_coh", "brier_prob", "brier_coh")]
        cells += ["--"] * 4 + safe(safety[safety.method.isin(PUBLISHED)])
        out.append("Mean, published & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


TABLES = (projsteps, projection)
