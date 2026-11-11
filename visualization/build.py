"""Rebuild the paper's tables and figures from ``results/`` and the saved scores in ``$OUTPUT_DIR``, directly into
the LaTeX project (``PAPER_DIR``, default ``latex/``).

Tables (``tables/<section>.py``) go to ``latex/tables/<label>-panel-<i>.tex``; figures (``figs/{main,appendix}.py``)
to ``latex/figs/<label>.pdf`` with the panel widths in ``latex/figs/panels.tex`` and PNG previews in
``artifacts/previews/``; the framework diagram (``figs/framework/framework.tex``) to
``latex/figs/framework.pdf``. Then ``make -C latex`` compiles the paper.

Usage::

    python -m visualization.build                    # every table and figure
    python -m visualization.build --tables           # tables only
    python -m visualization.build --figures          # figures only
    python -m visualization.build --only summary teaser   # selected tables (labels) and figures (functions)
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("agg")   # layout only; PDFs are written by the pdf backend

from uqbench.config import PAPER_DIR, RESULTS_DIR  # noqa: E402
from visualization import (
    style,  # noqa: E402
    tables,  # noqa: E402
)
from visualization.figs import appendix, main  # noqa: E402
from visualization.figs.framework import framework  # noqa: E402

FIGURES = (*main.FIGURES, *appendix.FIGURES, framework.main)


def build_figures(only: list[str] | None = None) -> list[str]:
    style.use()
    done = []
    for fn in FIGURES:
        name = fn.__module__.rsplit(".", 1)[-1] if fn is framework.main else fn.__name__
        if only and name not in only:
            continue
        fn()
        done.append(name)
    if done:
        from visualization.crop import main as crop_figures

        crop_figures()
    return done


def main_() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="tables only")
    ap.add_argument("--figures", action="store_true", help="figures only")
    ap.add_argument("--only", nargs="+", help="table labels and figure function names (framework: the diagram)")
    args = ap.parse_args()
    both = not (args.tables or args.figures)
    if args.tables or both:
        written = tables.write(PAPER_DIR / "tables", args.only)
        if {"auroc", "brier"} & set(written):
            from visualization.data import significance_frame

            significance_frame().to_csv(RESULTS_DIR / "paired_significance.csv", index=False)
        print("tables:", " ".join(written), flush=True)
    if args.figures or both:
        print("figures:", " ".join(build_figures(args.only)), flush=True)


if __name__ == "__main__":
    main_()
