"""Readable, unscaled table panels derived from the canonical numeric tabulars.

The source tables remain complete. Panels only select rows/columns and wrap labels;
no value, precision, ranking or significance mark is recomputed here.
"""

from __future__ import annotations

import re
from pathlib import Path

# Zero-based source columns. Repeated identifier columns make each panel self-contained.
PLANS = {
    "data": [(0, 1, *range(2, 7)), (0, 1, *range(7, 12))],
    "auroc": [(1, *range(2, 8)), (1, *range(8, 13))],
    "brier": [(1, *range(2, 8)), (1, *range(8, 13))],
    "summary-full": [tuple(range(10))],
    "projection": [(0, *range(1, 9)), (0, *range(9, 14))],
    "none-correct-full": [(0, *range(1, 6)), (0, *range(6, 12))],
    "information": [tuple(range(9))],
    "judge": [tuple(range(12))],
    "robustness": [(0, *range(1, 9)), (0, *range(9, 13))],
}
IDENTIFIERS = {"data": 2, "judge": 2}


def split_cells(row: str) -> list[str]:
    cells, start, depth = [], 0, 0
    for i, ch in enumerate(row):
        if ch == "{" and (i == 0 or row[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or row[i - 1] != "\\"):
            depth -= 1
        elif ch == "&" and depth == 0 and (i == 0 or row[i - 1] != "\\"):
            cells.append(row[start:i].strip())
            start = i + 1
    cells.append(row[start:].strip())
    return cells


def spans(line: str):
    pos = 0
    for cell in split_cells(line.removesuffix(r"\\").strip()):
        m = re.fullmatch(r"\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}", cell)
        n = int(m[1]) if m else 1
        yield pos, n, m[2] if m else cell
        pos += n


def wrap(text: str) -> str:
    replacements = {
        r"Correctness probe $+\C$": r"\shortstack{Correctness\\probe $+\C$}",
        "MATH (non-500)": r"\shortstack{MATH\\(non-500)}",
        "MATH-500": r"\shortstack{MATH-\\500}",
        "GPQA-Ext.": r"\shortstack{GPQA-\\Ext.}",
        "MMLU-Pro": r"\shortstack{MMLU-\\Pro}",
        "SuperGPQA": r"\shortstack{Super-\\GPQA}",
        "TriviaQA": r"\shortstack{Trivia-\\QA}",
        "Qwen3-8B": r"\shortstack{Qwen3-\\8B}",
        "R1-Distill-7B": r"\shortstack{R1-Distill-\\7B}",
        "gpt-oss-20b": r"\shortstack{gpt-oss-\\20b}",
        "Covered domains": r"\shortstack{Covered\\domains}",
        "GPQA, GPQA-Ext., SuperGPQA": r"\shortstack{GPQA, GPQA-Ext.,\\SuperGPQA}",
        "AUROC rival": r"\shortstack{AUROC\\rival}",
        "Brier rival": r"\shortstack{Brier\\rival}",
        r"$\Delta$AUROC [95\% CI]": r"\shortstack{$\Delta$AUROC\\{[95\% CI]}}",
        r"$-\Delta$Brier [95\% CI]": r"\shortstack{$-\Delta$Brier\\{[95\% CI]}}",
        r"None correct$\uparrow$": r"\shortstack{None\\correct$\uparrow$}",
        r"Consistent errors$\uparrow$": r"\shortstack{Consistent\\errors$\uparrow$}",
        r"\m{} AUROC": r"\shortstack{\m{}\\AUROC}",
        r"\m{} Brier": r"\shortstack{\m{}\\Brier}",
        "Sum binds": r"\shortstack{Sum\\binds}",
        "Guaranteed": r"\shortstack{Guaranteed}",
    }
    return replacements.get(text, text)


def select(line: str, cols: tuple[int, ...], wrap_labels: bool = True) -> str:
    if line.startswith(r"\cmidrule"):
        return ""  # Header groups are separated by their labels and a single booktabs rule.
    if not line.rstrip().endswith(r"\\"):
        return line
    result = []
    for start, count, content in spans(line):
        keep = [c for c in cols if start <= c < start + count]
        if not keep:
            continue
        if wrap_labels:
            content = wrap(content)
        if count > 1:
            result.append(rf"\multicolumn{{{len(keep)}}}{{l}}{{{content}}}")
        else:
            # Replace a multirow backbone by text on its first row; no font-size change.
            content = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]+)\}", r"\1", content)
            result.append(content)
    return " & ".join(result) + r" \\"


def panels(label: str, source: str) -> list[tuple[str, str]]:
    lines = source.strip().splitlines()
    body = lines[2:-2]
    boundary = next(i for i, s in enumerate(body) if s == r"\midrule")
    head, rows = body[:boundary], body[boundary + 1 :]
    if label in ("summary", "none-correct", "ablations", "blackbox-full", "abstention", "dependence"):
        # These compact summaries already fit the text width without splitting.
        return [("", source.strip() + "\n")]
    if label == "judge":
        # Backbone names are section headings, leaving space for full variant
        # names while all five test sets remain visible in one matrix.
        cols = tuple(range(1, 12))
        result = [r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X" + "c" * 10 + "}", r"\toprule"]
        result.extend(select(row, cols) for row in head)
        result.append(r"\midrule")
        for row in rows:
            if row.endswith(r"\\"):
                model = split_cells(row.removesuffix(r"\\").strip())[0]
                if model:
                    result.append(r"\multicolumn{11}{l}{\textbf{" + model + r"}} \\")
            result.append(select(row, cols, wrap_labels=False))
        result += [r"\bottomrule", r"\end{tabularx}"]
        return [("", "\n".join(result) + "\n")]
    n = max(sum(n for _, n, _ in spans(s)) for s in head if s.endswith(r"\\"))
    cols_list = PLANS.get(label, [tuple(range(n))])
    blocks = [("", rows)]
    if label in ("auroc", "brier"):
        blocks = []
        for row in rows:
            m = re.search(r"\\multirow\{\d+\}\{\*\}\{([^}]+)\}", row)
            if m:
                blocks.append((m[1], []))
            if blocks:
                blocks[-1][1].append(row)
    covered = {c for cols in cols_list for c in cols}
    if label in ("auroc", "brier"):
        covered.add(0)  # Backbone moves to the panel heading, never disappears.
    if covered != set(range(n)):
        raise ValueError(f"{label}: panel plan does not cover all {n} source columns")
    out = []
    for model, block in blocks:
        for cols in cols_list:
            nid = IDENTIFIERS.get(label, 1)
            if label in ("auroc", "brier"):
                nid = 1
            # Flexible text columns wrap naturally. Numeric cells preserve their original text.
            fmt = r">{\raggedright\arraybackslash}X" * nid + "c" * (len(cols) - nid)
            natural = label in {
                "cost",
                "ablations",
                "none-correct",
                "projsteps",
                } or (label == "summary-full" and len(cols) == 4)
            env = "tabular" if natural else "tabularx"
            if natural:
                fmt = "l" * nid + "c" * (len(cols) - nid)
            result = [r"\begin{" + env + "}" + ("" if natural else r"{\linewidth}") + "{" + fmt + "}", r"\toprule"]
            if model:
                result += [r"\multicolumn{" + str(len(cols)) + r"}{l}{\textbf{" + model + r"}} \\"]
            result.extend(select(x, cols) for x in head)
            result.append(r"\midrule")
            for row in block:
                rendered = select(row, cols, wrap_labels=False)
                if rendered.endswith(r"\\") and not any(
                        c.strip() for c in split_cells(rendered.removesuffix(r"\\").strip())[1:]):
                    continue  # a row whose values all fall in another panel
                if rendered == r"\midrule" and result[-1] == r"\midrule":
                    continue
                result.append(rendered)
            while result[-1] in ("", r"\midrule"):
                result.pop()
            result.extend([r"\bottomrule", r"\end{" + env + "}"])
            out.append((model, "\n".join(x for x in result if x) + "\n"))
    return out


def write_panels(out: Path, label: str, source: str) -> list[str]:
    paths = []
    for i, (_, text) in enumerate(panels(label, source), 1):
        name = f"{label}-panel-{i}"
        (out / f"{name}.tex").write_text(text)
        paths.append(name)
    for old in out.glob(f"{label}-panel-*.tex"):
        if old.stem not in paths:
            old.unlink()
    return paths


def write_paired_metrics(out: Path, auroc: str, brier: str) -> list[str]:
    """Place matched AUROC/Brier cells together (``metrics-panel-<i>.tex``), retaining every original mark."""
    auc, brier = panels("auroc", auroc), panels("brier", brier)
    written = []
    for i, ((model_a, text_a), (model_b, text_b)) in enumerate(zip(auc, brier, strict=True), 1):
        if model_a != model_b:
            raise ValueError("AUROC/Brier backbone order differs")
        lines, in_body = [], False
        for a, b in zip(text_a.splitlines(), text_b.splitlines(), strict=True):
            if in_body and a.endswith(r"\\"):
                ac, bc = (split_cells(row.removesuffix(r"\\").strip()) for row in (a, b))
                if ac[0] != bc[0] or len(ac) != len(bc):
                    raise ValueError(f"AUROC/Brier row mismatch: {ac[0]} / {bc[0]}")
                a = " & ".join([ac[0], *(x + "/" + y for x, y in zip(ac[1:], bc[1:], strict=True))]) + r" \\"
            elif a != b:
                raise ValueError("AUROC/Brier table structures differ")
            lines.append(a)
            if a == r"\midrule":
                in_body = True
        name = f"metrics-panel-{i}"
        (out / f"{name}.tex").write_text("\n".join(lines) + "\n")
        written.append(name)
    return written
