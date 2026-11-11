"""DeepConf (Fu et al., 2025, "Deep Think with Confidence") — offline trace confidence.

Token confidence C_i = -mean of the top-20 log-probs at step i. Trace-level variants (paper
Sec. 3): ``mean`` (average trace confidence), ``bottom10`` (mean of the lowest 10% sliding-window
group confidences), ``lowest`` (lowest group), ``tail`` (mean over the final 2048 tokens).
Group windows have size 2048 and stride 1; traces shorter than one window use the whole trace.
The main row uses the variant with the best validation AUROC.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline, select_by_auroc

WINDOW = 2048
VARIANTS = ("mean", "bottom10", "lowest", "tail")


def group_confidences(conf: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(conf) <= window:
        return np.array([conf.mean()])
    csum = np.concatenate([[0.0], np.cumsum(conf, dtype=np.float64)])
    return (csum[window:] - csum[:-window]) / window


def trace_confidence(conf: np.ndarray, variant: str) -> float:
    if variant == "mean":
        return float(conf.mean())
    if variant == "tail":
        return float(conf[-WINDOW:].mean())
    groups = group_confidences(conf)
    if variant == "lowest":
        return float(groups.min())
    if variant == "bottom10":
        m = max(1, int(np.ceil(0.1 * len(groups))))
        return float(np.sort(groups)[:m].mean())
    raise ValueError(variant)


class DeepConf(Baseline):
    paper = "DeepConf"

    def __init__(self, variant: str | None = None):
        self.variant = variant
        self.fixed = variant is not None
        self.name = f"DeepConf-{variant}" if self.fixed else "DeepConf"
        self.main = not self.fixed
        self.val_auroc = {}

    def _score(self, variant: str, q) -> np.ndarray:
        return np.array([trace_confidence(r["conf_topk"], variant) for r in q.replay])

    def fit(self, train, val):
        if not self.fixed:
            self.variant, self.val_auroc = select_by_auroc(VARIANTS, val, self._score)

    def score(self, q):
        return self._score(self.variant, q)

    def describe(self):
        return {"variant": self.variant, "window": WINDOW, "val_auroc": self.val_auroc}
