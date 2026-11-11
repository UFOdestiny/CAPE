"""Appendix figures, one function per figure (see ``figs/main.py`` for the file-naming convention)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import ABLATION_MODEL, BACKBONES
from visualization.style import (
    BLUE,
    COLUMN,
    FONT_SIZE,
    INK,
    ORANGE,
    PURPLE,
    SHORT,
    legend_strip,
    panel,
    ring,
    save,
)

SUB3 = 0.32                          # width of each of three subfigures (fraction of \textwidth)
PANELS = {"kscale": ["kscale-auroc", "kscale-brier", "kscale-unanimous"]}


def coherence() -> None:
    """Label-free lower bound versus realized Brier gain of the projection, one point per backbone, estimator and
    benchmark."""
    cert = pd.concat([pd.read_csv(RESULTS_DIR / m / "certificate.csv") for m in BACKBONES])
    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    hi = float(max(cert.certified_gain.max(), cert.realised_gain.max())) * 1.06
    ax.plot([0, hi], [0, hi], color=INK, linewidth=0.8, zorder=1)
    ring(ax, cert.certified_gain, cert.realised_gain, INK, size=12)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("label-free bound per response")
    ax.set_ylabel("realized Brier gain")
    ax.text(hi * 0.04, hi * 0.96, "realized = bound", fontsize=FONT_SIZE, color=INK, ha="left", va="top")
    save(fig, "coherence")


def kscale() -> None:
    """Measured AUROC, Brier score with the empirical vote-share floor, and unanimous-error rate up to K = 16."""
    df = pd.read_csv(RESULTS_DIR / ABLATION_MODEL / "kscale.csv")
    mean = df.groupby("K").mean(numeric_only=True)
    ks = mean.index.to_numpy()
    figs = [panel(SUB3, 1.8) for _ in range(3)]
    axes = [ax for _, ax in figs]
    curves = (("CAPE", BLUE, "D"), ("SC", ORANGE, "o"), ("CISC", INK, "^"))
    for ax, metric, label in ((axes[0], "auroc", "mean AUROC"), (axes[1], "brier", "mean Brier score")):
        for name, color, marker in curves:
            ax.plot(ks, mean[f"{name}_{metric}"], color=color, linewidth=1.6, zorder=2)
            ring(ax, ks, mean[f"{name}_{metric}"], color, marker=marker, size=16)
        ax.set_ylabel(label)
    ax = axes[1]
    ax.plot(ks, mean.agreement_floor, color=PURPLE, linewidth=3.2, zorder=0, solid_capstyle="round")
    legend_strip([Line2D([], [], color=c, marker=m, markersize=4, linewidth=1.6, label=n) for n, c, m in curves]
                 + [Line2D([], [], color=PURPLE, linewidth=3.2, label="vote shares only ($F_K$)")],
                 "kscale-legend", ncol=3, fontsize=FONT_SIZE, columnspacing=1.2)
    ax = axes[2]
    ends = {}
    for d, g in df[df.K >= 2].groupby("dataset", sort=False):
        if d == "math500":
            continue
        ax.plot(g.K, g.unanimous_error, color=INK, linewidth=1.2, zorder=2)
        ring(ax, g.K, g.unanimous_error, INK, size=10)
        ends[d] = g.unanimous_error.iloc[-1]
    placed = []
    for d, y in sorted(ends.items(), key=lambda t: t[1]):
        y = max(y, placed[-1] + 0.05) if placed else y       # keep end labels at least 0.05 apart
        placed.append(y)
        ax.text(df.K.max() * 1.16, y, SHORT[d], fontsize=FONT_SIZE, color=INK, va="center")
    ax.set_ylabel("fraction wrong")
    ax.set_ylim(0, None)
    ax.set_xlim(1.8, 30)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.minorticks_off()
        ax.set_xlabel("samples $K$")
    axes[2].set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
    for (fig, _), label in zip(figs, PANELS["kscale"], strict=True):
        save(fig, label)


FIGURES = (coherence, kscale)
