"""Compile the CAPE framework figure ``framework.tex`` (hand-edited TikZ) into the paper as
``latex/figs/framework.pdf`` (``fig:framework``). The canvas is 16 cm wide (the ACL text width), so the paper
includes it unscaled.

Usage::

    python -m visualization.figs.framework.framework      # also run by visualization/build.py --figures
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from uqbench.config import BUILD_DIR, PAPER_DIR
from visualization.style import preview

HERE = Path(__file__).resolve().parent


def main() -> Path:
    build = BUILD_DIR / "framework"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build}",
                    "framework.tex"], cwd=HERE, check=True, capture_output=True)
    out = PAPER_DIR / "figs" / "framework.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(build / "framework.pdf", out)
    preview(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
