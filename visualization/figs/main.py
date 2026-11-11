"""Main-text figures, one function per figure. A multi-panel figure writes one PDF per panel, named after the
panel's label (``teaser-null.pdf`` <-> ``fig:teaser-null``), and a legend strip when panels share one."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import (
    BACKBONES,
    escalation_recall,
    frames_of,
    low_parse,
    metric_frames,
    with_null_signals,
)
from visualization.data import PUBLISHED, UNANIMOUS_DOT, attribution_frames
from visualization.style import (
    AQUA,
    BLUE,
    COLUMN,
    FONT_SIZE,
    INK,
    ORANGE,
    PURPLE,
    SURFACE,
    legend_strip,
    panel,
    ring,
    save,
)

SUB3 = 0.32                          # width of each of three subfigures (fraction of \textwidth)
PANELS = {"teaser": ["teaser-unanimous", "teaser-geometry", "teaser-null"],
          "analysis": ["analysis-projection", "analysis-reliability", "analysis-none"]}
# one line style per method across figures: CAPE solid, each baseline its own dash
DASH = {"CAPE": "-", "SC": (0, (3, 2)), "DC": (0, (5, 1.5, 1, 1.5)), "Comb": (0, (1, 1.3)), "rate": (0, (2, 1.5))}


def coherence_means() -> pd.DataFrame:
    """Calibrated vs. projected probabilities (results/<model>/coherence_effect.csv), high-parse test sets."""
    rows = []
    for model, df in metric_frames().items():
        eff = pd.read_csv(RESULTS_DIR / model / "coherence_effect.csv")
        eff = eff[~eff.dataset.isin(low_parse(model))]
        for r in eff.itertuples():
            rows.append({"method": r.method, "kind": "raw", "auroc": r.auroc_prob, "brier": r.brier_prob})
            rows.append({"method": r.method, "kind": "coh", "auroc": r.auroc_coh, "brier": r.brier_coh})
        cape = df[(df.method == "CAPE") & (df.dataset != "math_all") & ~df.dataset.isin(low_parse(model))]
        rows += [{"method": "CAPE", "kind": "raw", "auroc": a, "brier": br}
                 for a, br in zip(cape.auroc, cape.brier, strict=True)]
    return pd.DataFrame(rows).groupby(["method", "kind"])[["auroc", "brier"]].mean()


def teaser() -> None:
    """Problem, mechanism and remedy: (a) on unanimous questions votes are certain, yet some answers are wrong, and
    CAPE's confidence separates them; (b) the coherent set: votes and projected single-response confidences sit on the
    edge where "none correct" has no mass, CAPE's joint prediction inside; (c) the mass each leaves for "none
    correct" against how often it happens, by the number of distinct answers."""
    # (a) CAPE's confidence on unanimous questions, right against wrong; votes give every one confidence 1
    fig, ax = panel(SUB3, 1.9)
    q = pd.concat([pd.read_csv(RESULTS_DIR / m / "questions.csv")
                   .pipe(lambda d, m=m: d[~d.dataset.isin(low_parse(m) | {"math_all"})]) for m in BACKBONES])
    u = q[q.unanimous]
    bins = np.linspace(0.2, 1.0, 17)
    for ok, color, sign in ((True, AQUA, 1), (False, BLUE, -1)):   # right answers above the axis, wrong below
        v = u[u.correct == ok].cape_top.to_numpy()
        h, _ = np.histogram(np.clip(v, 0.2, 1.0), bins=bins)
        ax.stairs(sign * h / len(v), bins, baseline=0, fill=True, color=color, alpha=0.9, zorder=2)
    ax.axhline(0, color=INK, linewidth=0.6, zorder=3)
    ax.axvline(1.0, color=ORANGE, linewidth=1.6, linestyle=DASH["SC"], zorder=4)
    wrong = (~u.correct).mean()
    ax.text(0.97, -0.42, "votes: 1\nfor all", ha="right", va="center", fontsize=FONT_SIZE, color=INK)
    ax.text(0.25, 0.30, "right", ha="left", va="center", fontsize=FONT_SIZE, color=INK)
    ax.text(0.25, -0.20, rf"wrong ({wrong:.0%})", ha="left", va="center", fontsize=FONT_SIZE,
            color=BLUE)
    ax.set_xlim(0.2, 1.03)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([0.2, 0.6, 1.0], ["0.2", "0.6", "1"])
    ax.set_yticks([-0.4, 0, 0.4], ["0.4", "0", "0.4"])
    ax.set_xlabel("CAPE confidence in the\nunanimous answer")
    ax.set_ylabel("share within each group")
    save(fig, "teaser-unanimous", height=2.12)

    # (b) correctness probabilities for two mutually exclusive answers. The horizontal gap from
    # (pi_42, pi_41) to pi_42 + pi_41 = 1 is exactly the probability of "none correct".
    fig, ax = panel(SUB3 + 0.015, 1.9)
    tri = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    ax.fill(tri[:, 0], tri[:, 1], color="#EEF3F9", zorder=0)
    ax.plot(tri[:, 0], tri[:, 1], color=INK, linewidth=0.8, zorder=1)
    ax.plot([1, 0], [0, 1], color=ORANGE, linewidth=2.2, zorder=1, solid_capstyle="butt")
    ax.text(0.04, 1.01, r"$P(\varnothing)=0$", rotation=-45, rotation_mode="anchor",
            ha="left", va="bottom", fontsize=FONT_SIZE, color=INK)
    ax.text(-0.04, -0.04, "0", ha="right", va="top", fontsize=FONT_SIZE, color=INK)
    ax.text(1.0, -0.04, "1", ha="center", va="top", fontsize=FONT_SIZE, color=INK)
    ax.text(-0.04, 1.0, "1", ha="right", va="center", fontsize=FONT_SIZE, color=INK)
    ax.text(0.52, -20 / 97.2, r"$P(42\ \mathrm{correct})$", ha="center", va="top", color=INK)
    ax.text(-0.17, 0.50, r"$P(41\ \mathrm{correct})$", rotation=90, ha="center", va="center", color=INK)
    p0, q0 = np.array([0.8, 0.8]), np.array([0.56, 0.44])   # each answer's mean confidence 0.8; its projection
    cape = np.array([0.62, 0.23])                           # the running example's CAPE output (Figure 2)
    ax.annotate("", xy=q0, xytext=p0, zorder=3,
                arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.8, "mutation_scale": 7,
                            "shrinkA": 3, "shrinkB": 3})
    ring(ax, [p0[0]], [p0[1]], PURPLE, size=22, z=4)
    ring(ax, [q0[0]], [q0[1]], AQUA, size=22, z=4)
    # This horizontal length is 1 - 0.62 - 0.23 = 0.15. Its boundary endpoint
    # is distinct from q0; a vertical segment beside q0 suggests a false alignment.
    edge_x = 1 - cape[1]
    none_mass = 1 - cape.sum()
    ax.plot([cape[0], edge_x], [cape[1], cape[1]], color=BLUE, linewidth=1.4, zorder=4)
    ax.plot([edge_x, edge_x], [cape[1] - 0.025, cape[1] + 0.025], color=BLUE, linewidth=1.4, zorder=4)
    ring(ax, [cape[0]], [cape[1]], BLUE, marker="*", size=70, z=5)
    ax.text(0.80, 0.90, "raw scores" + "\n" + r"$(0.8,\,0.8)$", ha="center", va="bottom",
            fontsize=FONT_SIZE, color=INK, linespacing=1.0)
    ax.text(0.69, 0.62, "projection", ha="left", va="top", fontsize=FONT_SIZE, color=INK)
    ax.text(0.08, 0.53, r"sum $\leq 1$", ha="left", va="center", color=INK)
    ax.text(cape[0] - 0.06, cape[1], "CAPE", ha="right", va="center", color=BLUE)
    ax.text((cape[0] + edge_x) / 2, cape[1] - 0.07,
            rf"$P(\varnothing)={none_mass:.2f}$", ha="center", va="top", color=BLUE)
    ax.set_xlim(-0.23, 1.08)
    ax.set_ylim(-0.26, 1.17)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "teaser-geometry", height=2.12)

    # (c) mass on "none correct" against its rate, by the number of distinct answers
    fig, ax = panel(SUB3, 1.9)
    nul = attribution_frames()["null"]
    strata = ("m1", "m2", "m3+")
    mean = nul.groupby(["predictor", "stratum"])[["mean_mass", "none_rate"]].mean()
    x = np.arange(len(strata))
    rate = [mean.loc[("CAPE", m), "none_rate"] for m in strata]
    ax.plot(x, rate, color=INK, linewidth=1.1, linestyle=DASH["rate"], marker="o", markersize=3, zorder=4)
    ax.plot(x, [mean.loc[("CAPE", m), "mean_mass"] for m in strata], color=BLUE, linewidth=1.8, marker="o",
            markersize=3.5, zorder=5)
    dc = [mean.loc[("DeepConf +C, leftover", m), "mean_mass"] for m in strata]
    ax.plot(x, dc, color=AQUA, linewidth=1.3, linestyle=DASH["DC"], marker="o", markersize=3, zorder=3)
    ax.plot(x, [0, 0, 0], color=ORANGE, linewidth=1.4, linestyle=DASH["SC"], zorder=3, clip_on=False)
    ax.text(2.05, mean.loc[("CAPE", "m3+"), "mean_mass"], "CAPE", ha="left", va="center", color=BLUE)
    ax.text(2.05, rate[2], "actual\nrate", ha="left", va="center", color=INK, linespacing=0.9)
    ax.text(2.05, dc[2] + 0.035, r"DC $+\mathcal{C}$", ha="left", va="center", color=INK)
    ax.text(1.0, 0.022, "SC", ha="center", va="center", color=INK)
    ax.set_xticks(x, ["1", "2", "3+"])
    ax.set_xlim(-0.15, 2.0)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.2, 0.4], ["0", "0.2", "0.4"])
    ax.set_xlabel("distinct answers among\n8 responses")
    ax.set_ylabel(r"“none correct” probability")
    save(fig, "teaser-null", height=2.12)


def analysis() -> None:
    """The two principles and a decision: (a) AUROC of every incoherent published estimator before and after
    projection; (b) the "none correct" Brier score of the same evidence predicted jointly or read off a projection;
    (c) escalating the questions with the most "none correct" mass. Appendix: the unanimous-question ROC."""
    # (a) the projection applied to every incoherent estimator (vote-based ones are already coherent)
    coh = coherence_means()
    rows = [r for r in PUBLISHED if (r, "raw") in coh.index]
    rows.sort(key=lambda r: -coh.loc[(r, "coh"), "auroc"])
    short = {"SelfCertainty": "Self-Cert.", "SemEnergy": "Sem. Energy", "ReasonProbe": "Corr. probe",
             "LRMVerbal": "Verbal conf.", "ConsistCal": "Sample consist."}
    fig, ax = panel(SUB3, 1.9)
    cape, sc = coh.loc[("CAPE", "raw"), "auroc"], coh.loc[("SC", "raw"), "auroc"]
    ax.axvline(cape, color=BLUE, linewidth=1.4, zorder=0)
    ax.axvline(sc, color=ORANGE, linewidth=1.1, linestyle=DASH["SC"], zorder=0)
    for y, b in enumerate(rows):
        raw, pr = coh.loc[(b, "raw"), "auroc"], coh.loc[(b, "coh"), "auroc"]
        ax.annotate("", xy=(pr, y), xytext=(raw, y), zorder=1,
                    arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.7, "mutation_scale": 5,
                                "shrinkA": 2.4, "shrinkB": 2.4})
        ring(ax, [raw], [y], PURPLE, size=13)
        ring(ax, [pr], [y], INK, size=13, z=4)
    ax.text(cape + 0.004, -1.0, "CAPE", ha="left", va="center", color=BLUE)
    ax.text(sc - 0.004, -1.0, "SC", ha="right", va="center", color=INK)
    ax.set_xlim(0.57, 0.93)
    ax.set_xticks([0.6, 0.75, 0.9], [".60", ".75", ".90"])
    ax.set_ylim(len(rows) - 0.45, -1.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC")
    ax.set_yticks(range(len(rows)), [short.get(r, r) for r in rows], fontsize=FONT_SIZE)
    ax.tick_params(axis="y", length=0, pad=2)
    save(fig, "analysis-projection", height=2.12)

    # (b) reliability of the "none correct" mass on split questions: CAPE's learned mass against the residual
    # that projection leaves (pooled questions, marker area by count)
    q = with_null_signals(frames_of("questions"))
    q = q[[d not in low_parse(m) | {"math_all"} for m, d in zip(q.model, q.dataset, strict=True)]]
    split = q[~q.unanimous.astype(bool)]
    edges = [-1e-9, 1e-6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    fig, ax = panel(SUB3, 1.9)
    ax.plot([0, 1], [0, 1], color=INK, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    for col, color, dash, z in (("deepconf_null", AQUA, DASH["DC"], 2), ("stacker_null", PURPLE, DASH["Comb"], 3),
                                ("cape_null", BLUE, "-", 4)):
        g = split.groupby(pd.cut(split[col], edges), observed=True).agg(n=("none", "size"), rate=("none", "mean"),
                                                                        pred=(col, "mean"))
        g = g[g.n >= 50]
        ax.plot(g.pred, g.rate, color=color, linestyle=dash, linewidth=1.8 if z == 4 else 1.2, zorder=z)
        ax.scatter(g.pred, g.rate, s=4 + 60 * g.n / len(split), color=color, edgecolor=SURFACE, linewidth=0.6,
                   zorder=z)
    zero_rate = split.loc[split.deepconf_null <= 1e-6, "none"].mean()
    ax.annotate("zero residual", xy=(0, zero_rate), xytext=(0.04, 0.72),
                ha="left", va="center", color=INK,
                arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.6})
    ax.text(0.62, 0.86, "CAPE", ha="right", va="center", color=BLUE)
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel(r"predicted probability of $\varnothing$")
    ax.set_ylabel("actual “none correct” rate")
    save(fig, "analysis-reliability", height=2.12)

    # (c) share of the questions without a correct sample that are caught, per escalation budget (mean over
    # pairs); fixed read-out rule as in the tables
    fig, ax = panel(SUB3, 1.9)
    budgets = np.arange(0.05, 0.501, 0.05)
    curves = (("vote", "SC", ORANGE, DASH["SC"]), ("dc_top", r"DC $+\mathcal{C}$", AQUA, DASH["DC"]),
              ("stack_top", r"Comb.+probe$+\mathcal{C}$", PURPLE, DASH["Comb"]), ("cape_null", "CAPE", BLUE, "-"))
    ends = {}
    for col, label, color, dash in curves:
        caught = [np.mean(escalation_recall(q, col, b, dropna=False)) for b in budgets]
        ax.plot(budgets, caught, color=color, linestyle=dash, linewidth=1.8 if col == "cape_null" else 1.2,
                zorder=3 if col == "cape_null" else 2)
        ends[label] = caught
    ax.plot([0, 0.5], [0, 0.5], color=INK, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    i20 = int(np.argmin(np.abs(budgets - 0.2)))
    ax.scatter([0.2], [ends["CAPE"][i20]], s=20, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    ax.text(0.19, ends["CAPE"][i20] + 0.05, f"{ends['CAPE'][i20]:.0%}", ha="right",
            color=BLUE)
    ax.text(0.21, ends["SC"][i20] - 0.07, f"{ends['SC'][i20]:.0%}", ha="left", color=INK)
    label_y = {"CAPE": 0.04, "SC": -0.05}
    for label, vals in ends.items():
        if label in label_y:
            ax.text(0.51, vals[-1] + label_y[label], label, ha="left", va="center",
                    color=BLUE if label == "CAPE" else INK)
    ax.set_xlim(0.05, 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0.1, 0.3, 0.5], ["10", "30", "50"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.set_xlabel("questions deferred (%)")
    ax.set_ylabel(r"“none correct” recall (%)")
    save(fig, "analysis-none", height=2.12)
    legend_strip([Line2D([], [], color=BLUE, linewidth=1.8, label="CAPE"),
                  Line2D([], [], color=ORANGE, linewidth=1.2, linestyle=DASH["SC"], label="SC"),
                  Line2D([], [], color=AQUA, linewidth=1.2, linestyle=DASH["DC"], label=r"DC $+\mathcal{C}$"),
                  Line2D([], [], color=PURPLE, linewidth=1.2, linestyle=DASH["Comb"],
                         label=r"Comb.+probe$+\mathcal{C}$"),
                  Line2D([], [], marker="o", linestyle="", color=PURPLE, markersize=4, label="calibrated"),
                  Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="projected")],
                 "analysis-legend", ncol=6, fontsize=FONT_SIZE, handletextpad=0.3, columnspacing=0.9,
                 handlelength=1.8)

    # appendix: unanimous questions, share of consistent errors flagged against consistent successes flagged
    fig, ax = plt.subplots(figsize=(COLUMN, 2.5))
    qq = pd.concat([pd.read_csv(RESULTS_DIR / m / "questions.csv")
                    .pipe(lambda d, m=m: d[~d.dataset.isin(low_parse(m))]) for m in BACKBONES])
    u = qq[qq.unanimous]
    ax.plot([0, 1], [0, 1], color=ORANGE, linewidth=1.3, linestyle=DASH["SC"], zorder=2, label="SC")
    for col, color, label, dash in (("deepconf_top", AQUA, r"DC $+\mathcal{C}$", DASH["DC"]),
                                    ("stacker_top", PURPLE, r"Comb.+probe$+\mathcal{C}$", DASH["Comb"]),
                                    ("cape_top", BLUE, "CAPE", "-")):
        right, wrong = np.sort(u[u.correct][col].to_numpy()), np.sort(u[~u.correct][col].to_numpy())
        grid = np.unique(np.r_[right, wrong, 1.01])
        ax.plot(np.searchsorted(right, grid) / len(right), np.searchsorted(wrong, grid) / len(wrong), color=color,
                linewidth=1.8 if col == "cape_top" else 1.2, linestyle=dash, zorder=3, label=label)
    t = UNANIMOUS_DOT
    fw, fr = (u[~u.correct].cape_top < t).mean(), (u[u.correct].cape_top < t).mean()
    ax.scatter([fr], [fw], s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("correct unanimous answers flagged (FPR)")
    ax.set_ylabel("incorrect unanimous answers\nflagged (TPR)")
    ax.legend(loc="lower right", fontsize=FONT_SIZE, handlelength=1.8, handletextpad=0.3, borderaxespad=0.1,
              labelspacing=0.2)
    save(fig, "analysis-unanimous")


def dependence() -> None:
    """Shared-prefix intervention on Qwen3-8B (``uqbench.studies.dependence``): change of the mean Brier score from
    independent sampling as the K responses share the first fraction f of one reasoning trace. Single-response
    estimators keep their per-response distribution; vote counting degrades; CAPE's "none correct" absorbs it."""
    me = pd.read_csv(RESULTS_DIR / "qwen3-8b" / "dependence_methods.csv").groupby(["method", "fraction"]).brier.mean()
    fracs = sorted(me.index.get_level_values(1).unique())
    fig, ax = plt.subplots(figsize=(COLUMN, 1.2))
    fig.subplots_adjust(top=0.98)
    ax.axhline(0, color=INK, linewidth=0.6, zorder=1)
    lines = (("DeepConf", None, "#9A9A9A", "-", 0.9), ("ReasonProbe", None, "#9A9A9A", "-", 0.9),
             ("SEP", "single response", "#9A9A9A", "-", 0.9),
             ("SC", "SC", ORANGE, DASH["SC"], 1.2),
             ("SemEnergy", "Sem. Energy", PURPLE, (0, (4, 1, 1, 1, 1, 1)), 1.2),
             ("DeepConf×SC", r"DC$\times$SC", AQUA, DASH["DC"], 1.2),
             ("CAPE w/o null", r"CAPE w/o $\varnothing$", BLUE, (0, (4, 1.5)), 1.2),
             ("CAPE", "CAPE", BLUE, "-", 1.8))
    for key, label, color, dash, width in lines:
        delta = [me[key, f] - me[key, 0.0] for f in fracs]
        ax.plot(fracs, delta, color=color, linestyle=dash, linewidth=width, label=label,
                zorder=3 if key == "CAPE" else 2)
        ring(ax, fracs, delta, color, size=10, z=4 if key == "CAPE" else 3)
    ax.set_xticks(fracs, [f"{f:g}" for f in fracs])
    ax.set_xlim(-0.03, 0.78)
    ax.set_yticks([0, 0.02, 0.04, 0.06], ["0", ".02", ".04", ".06"])
    ax.set_xlabel(r"shared fraction $f$ of the reasoning trace")
    ax.set_ylabel("Brier change")
    handles, labels = ax.get_legend_handles_labels()
    order = [5, 4, 1, 3, 2, 0]   # CAPE, CAPE w/o null, SC | DC x SC, Sem. Energy, single response (column-major)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", ncol=2, frameon=False,
               bbox_to_anchor=(0.0, 0.99), fontsize=FONT_SIZE, handlelength=1.7, columnspacing=0.9,
               labelspacing=0.2, borderaxespad=0.0)
    save(fig, "dependence")


FIGURES = (teaser, analysis, dependence)
