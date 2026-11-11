"""Paper tables, one function per table, grouped by paper section (one module per section).

A table function takes no arguments and returns the ``tabular``; ``write`` lays it out in text-width panels
(``tables.layout``) and saves them to the paper as ``latex/tables/<label>-panel-<i>.tex``, where the label is the
function name with ``_`` replaced by ``-`` (``summary_full`` -> ``summary-full-panel-1.tex`` -> ``tab:summary-full``).
To add a table, write the function in its section module and list it in that module's ``TABLES``.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

from visualization.tables.fmt import typeset_minus
from visualization.tables.layout import write_paired_metrics, write_panels

SECTIONS = ("results", "implementation", "main_metrics", "coherence", "beyond_agreement", "ablations",
            "robustness")


def label(fn) -> str:
    return fn.__name__.replace("_", "-")


def all_tables() -> list:
    return [fn for name in SECTIONS for fn in import_module(f"visualization.tables.{name}").TABLES]


PAIRED = ("auroc", "brier")   # printed together as metrics-panel-<i>.tex


def write(out: Path, only: list[str] | None = None) -> list[str]:
    """Write the panels of every table (or of the labels in ``only``) to ``out/<label>-panel-<i>.tex``, the
    files the paper inputs."""
    out.mkdir(parents=True, exist_ok=True)
    if only and set(PAIRED) & set(only):
        only = [*only, *PAIRED]
    written, paired = [], {}
    for fn in all_tables():
        if only and label(fn) not in only:
            continue
        source = typeset_minus(fn() + "\n")
        if label(fn) in PAIRED:
            paired[label(fn)] = source
        else:
            write_panels(out, label(fn), source)
        written.append(label(fn))
    if len(paired) == len(PAIRED):
        write_paired_metrics(out, *(paired[k] for k in PAIRED))
    return written
