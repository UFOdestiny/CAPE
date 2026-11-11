"""Trim vector figure margins without scaling type or rasterizing the PDF.

Pixel inspection locates visible ink only; the saved PDF retains its native
text and paths. Related panels share vertical crop limits for aligned axes
and subcaptions. Resetting the crop box makes repeat builds idempotent.
"""
from pathlib import Path

import numpy as np
import pymupdf


def trim_group(paths, pad=1.5):
    records = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        doc = pymupdf.open(path)
        if len(doc) != 1:
            doc.close()
            raise ValueError(f"Expected a one-page figure: {path}")
        page = doc[0]
        previous = tuple(page.cropbox)
        page.set_cropbox(page.mediabox)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2),
                              colorspace=pymupdf.csGRAY, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        yy, xx = np.nonzero(pixels < 250)
        if not len(xx):
            doc.close()
            raise ValueError(f"Empty figure: {path}")
        box = pymupdf.Rect(xx.min() / 2 - pad, yy.min() / 2 - pad,
                           (xx.max() + 1) / 2 + pad, (yy.max() + 1) / 2 + pad)
        records.append((path, doc, previous, box & page.rect))
    if not records:
        return
    top = min(r[3].y0 for r in records)
    bottom = max(r[3].y1 for r in records)
    for path, doc, previous, box in records:
        box.y0, box.y1 = top, bottom
        doc[0].set_cropbox(box)
        if any(abs(a - b) > .01 for a, b in zip(previous, box, strict=True)):
            payload = doc.tobytes(garbage=3, deflate=True)
            doc.close()
            path.write_bytes(payload)
        else:
            doc.close()


def main():
    from uqbench.config import PAPER_DIR
    from visualization.figs import appendix
    from visualization.figs import main as figures
    from visualization.style import preview, write_panels
    root = PAPER_DIR / "figs"
    groups = {key: names for key, names in {**figures.PANELS, **appendix.PANELS}.items()
              if all((root / f"{name}.pdf").is_file() for name in names)}
    for names in groups.values():
        trim_group([root / f"{name}.pdf" for name in names])
    singles = ["framework", "coherence", "analysis-unanimous",
               "analysis-legend", "kscale-legend", "dependence"]
    for name in singles:
        trim_group([root / f"{name}.pdf"])
    write_panels(groups)
    for name in [*singles, *(name for names in groups.values() for name in names)]:
        if (root / f"{name}.pdf").is_file():
            preview(root / f"{name}.pdf")


if __name__ == "__main__":
    main()

