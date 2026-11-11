"""Shared figure style for the paper (ACL two-column, A4).

Figures are drawn at their printed width and included without scaling, so the font sizes below are the sizes on
the page. Times New Roman is embedded at the ACL body's 10.95 TeX pt (10.909 PDF pt).
``save`` checks text size and color. Mathematical indices retain their normal size ratios.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from uqbench.config import PAPER_DIR, PREVIEW_DIR

HERE = Path(__file__).resolve().parent
OUT = PAPER_DIR / "figs"               # <label>.pdf, included by the paper as figs/<label>.pdf
PREVIEW = PREVIEW_DIR    # PNG previews rendered from the PDFs

CM = 1 / 2.54
COLUMN = 7.7 * CM          # \columnwidth of the ACL template (inches)
TEXT = 16.0 * CM           # \textwidth
FONT_SIZE = 10.95 * 72 / 72.27   # Matplotlib uses PDF points; ACL normalsize is 10.95 TeX pt

# Stable jewel-tone roles across the paper: CAPE, SC, DeepConf, and the combined baseline.
BLUE, ORANGE, AQUA, PURPLE = "#2F5DA8", "#C9752D", "#238477", "#7A5A9E"
INK, SURFACE = "#000000", "#ffffff"
TEXT_COLORS = {INK.lower(), BLUE.lower()}   # text is black or CAPE's colour

SHORT = {"math500": "MATH", "olympiadbench": "Olymp.", "minerva": "Minerva", "aime": "AIME", "math_all": "Math",
         "gpqa": "GPQA", "mmlupro": "MMLU-Pro", "supergpqa": "SuperGPQA", "triviaqa": "TriviaQA"}


def use() -> None:
    """Apply the paper style to matplotlib."""
    from matplotlib import font_manager

    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.bfit": "Times New Roman:italic:bold",
        "mathtext.cal": "Times New Roman:italic",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def check_text(fig) -> None:
    """Every label uses the body size and black or CAPE's colour."""
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        assert abs(t.get_fontsize() - FONT_SIZE) < 1e-6, \
            f"text {t.get_text()!r} at {t.get_fontsize()} pt != {FONT_SIZE} pt"
        color = to_hex(t.get_color(), keep_alpha=False)
        assert color in TEXT_COLORS, f"text {t.get_text()!r} in {color}; use black or the method colour"


def _fit_width(fig, target: float, rounds: int = 6) -> None:
    """Resize the canvas until the tight bounding box is exactly ``target`` inches wide, so the PDF is
    included at 100% and its fonts print at their nominal size."""
    for _ in range(rounds):
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        width = box.width + 2 * mpl.rcParams["savefig.pad_inches"]
        if abs(width - target) < 0.004:
            return
        fig.set_figwidth(fig.get_figwidth() + target - width)


def preview(path: Path, dpi: int = 220) -> Path:
    """A PNG preview of a PDF figure (``PREVIEW_DIR/<label>.png``)."""
    import pymupdf

    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = PREVIEW / f"{path.stem}.png"
    with pymupdf.open(path) as doc:
        doc[0].get_pixmap(dpi=dpi).save(out)
    return out


def save(fig, label: str, fit: bool = True, height: float | None = None) -> Path:
    """Check the figure's text, fit it to its printed width (the width it was created with) and write
    ``latex/figs/<label>.pdf`` and its preview. Legend strips (``fit=False``) keep their natural size."""
    check_text(fig)
    grouped = label.startswith(("teaser-", "analysis-", "kscale-")) and not label.endswith("legend")
    if grouped:
        # Equal paper-space canvas and plotting baselines within each subfigure group.
        height = 2.35 if label.startswith("kscale-") else 2.60
        fig.set_figheight(height)
        bottom, top = (48, 140) if label.startswith("kscale-") else (54, 151.2)
        if label in {"analysis-projection", "analysis-reliability", "analysis-none"}:
            # Eleven method rows need at least one body-text line each. Keep all
            # three analysis axes aligned, without shrinking their typography.
            height, top = 3.20, 194.4
            fig.set_figheight(height)
        width = fig.get_figwidth() * 72
        ax = fig.axes[0]
        if label == "teaser-geometry":
            side, left = top - bottom, 31
            ax.set_position([0, 0, 1, 1])
            ax.set_xlim(-left / side, (width - left) / side)
            ax.set_ylim(-bottom / side, (height * 72 - bottom) / side)
        else:
            pos = ax.get_position()
            ax.set_position([pos.x0, bottom / (height * 72), pos.width, (top - bottom) / (height * 72)])
            for _ in range(12):
                fig.canvas.draw()
                box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                pos = ax.get_position()
                left = pos.x0 + max(0, .025 - box.x0) / fig.get_figwidth()
                right = pos.x1 - max(0, box.x1 - fig.get_figwidth() + .025) / fig.get_figwidth()
                if abs(left - pos.x0) + abs(right - pos.x1) < .001:
                    break
                ax.set_position([left, pos.y0, max(.12, right - left), pos.height])
            pos = ax.get_position()
            ax.xaxis.set_label_coords((pos.x0 + pos.x1) / 2, (bottom - 20) / (height * 72), transform=fig.transFigure)
            ax.xaxis.label.set_verticalalignment("top")
    elif fit:
        _fit_width(fig, fig.get_figwidth())
    if height is not None and not grouped:
        for _ in range(8):
            actual = fig.get_tightbbox(fig.canvas.get_renderer()).height + 2 * mpl.rcParams["savefig.pad_inches"]
            if abs(actual - height) < 0.003:
                break
            fig.set_figheight(fig.get_figheight() + (height - actual) / 0.77)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.pdf"
    with mpl.rc_context({"savefig.bbox": None if grouped else "tight"}):
        fig.savefig(path)
    plt.close(fig)
    preview(path)
    return path


def panel(frac: float, height: float, base: float = TEXT):
    """One subfigure panel, drawn at ``frac`` of ``base`` (the subfigure width used in LaTeX)."""
    return plt.subplots(figsize=(frac * base, height))


def legend_strip(handles, label: str, ncol: int, **kw) -> Path:
    """A legend shared by several subfigures, saved on its own and included at natural size."""
    fig = plt.figure(figsize=(TEXT, 0.3 if len(handles) <= ncol else 0.58))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, borderaxespad=0.0, **kw)
    return save(fig, label, fit=False)


def ring(ax, x, y, color, marker="o", size=18, z=3, **kw):
    """Data marker with a 1 px surface ring so overlapping points stay separable."""
    return ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor=SURFACE, linewidth=0.8, zorder=z, **kw)


def write_panels(groups: dict[str, list[str]]) -> Path:
    """``latex/figs/panels.tex``: the ``\\panel`` command and the printed width of every panel PDF, so a multi-panel
    figure is assembled as ``\\panel{\\panelwidth{<file>}}{<file>}{<subcaption>}{fig:<file>}`` without scaling."""
    import pymupdf

    lines = ["% Generated by visualization/build.py: printed width of each figure panel (figs/<label>.pdf).",
             r"\providecommand{\panel}[4]{\begin{subfigure}[t]{#1}\centering\includegraphics{figs/#2}"
             r"\caption{#3}\label{#4}\end{subfigure}}",
             r"\providecommand{\panelwidth}[1]{\csname panelwidth@#1\endcsname}"]
    for figure, labels in groups.items():
        lines.append(f"% {figure}")
        for name in labels:
            with pymupdf.open(OUT / f"{name}.pdf") as doc:
                width = doc[0].rect.width / 72 * 2.54
            lines.append(rf"\expandafter\def\csname panelwidth@{name}\endcsname{{{width:.3f}cm}}")
    path = OUT / "panels.tex"
    path.write_text("\n".join(lines) + "\n")
    return path
