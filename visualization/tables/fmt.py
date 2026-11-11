"""Cell formatting shared by the paper tables (booktabs tabulars, no float environment)."""
from __future__ import annotations

import re

import numpy as np

MINUS = re.compile(r"(?<=[\s&{(\[/])-(?=\d)")


def bold(text: str) -> str:
    return rf"\textbf{{{text}}}"


def underline(text: str) -> str:
    return rf"\underline{{{text}}}"


def ranked(values, higher: bool, digits: int = 3) -> list[float]:
    """The best and second-best values of a column at the printed precision (NaN ignored)."""
    return sorted({round(v, digits) for v in values if not np.isnan(v)}, reverse=higher)[:2]


def mark(value: float, top: list[float], digits: int = 3) -> str:
    """A value at the printed precision, in bold if it is the best of its column and underlined if second."""
    if np.isnan(value):
        return "--"
    txt, v = f"{value:.{digits}f}", round(value, digits)
    if top and v == top[0]:
        return bold(txt)
    if len(top) > 1 and v == top[1]:
        return underline(txt)
    return txt


def close(out: list[str]) -> str:
    """Finish a tabular whose last block ends with a ``\\midrule``."""
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out)


def typeset_minus(text: str) -> str:
    """Negative numbers with a true minus sign."""
    return MINUS.sub("$-$", text)


# Printed method names (glossary of the paper): internal result names -> table labels.
NAMES = {"ReasonProbe": "Correctness probe", "LRMVerbal": "Verbal confidence",
         "ConsistCal": "Sample consistency", "CAPE": r"\textbf{\m}",
         "SelfCertainty": "Self-Certainty", "SemEnergy": r"Sem.\ Energy",
         "DeepConf×SC": r"DeepConf$\times$SC", "Stacker": "Combiner", "Stacker +Coh": r"Combiner $+\C$",
         "Stacker + probe +Coh": r"\quad $+$ probe", "Stacker + probe, gated +Coh": "Coherent combiner",
         "Coherent stacker": "Coherent combiner", "Coherent stacker + learned null": r"\quad $+$ $\nul$ classifier",
         "Modular": r"Modular ($+\,\nul$ classifier)", "CAPE agreement only": r"\m, votes only",
         "SC": r"SC\smash{$^\dagger$}", "SE": r"SE\smash{$^\dagger$}"}


def name(method: str) -> str:
    """The printed name of a method; projected estimators (``<name>+Coh`` / ``<name>+C``) get ``$+\\C$``."""
    if method in NAMES:
        return NAMES[method]
    for suffix in ("+Coh", "+C"):
        if method.endswith(suffix):
            return NAMES.get(method[:-len(suffix)], method[:-len(suffix)]) + r" $+\C$"
    return method


def paren(text: str, low: bool) -> str:
    """A cell of a low-parse pair, in parentheses (excluded from every count and from bold)."""
    return f"({text})" if low else text
