"""Matched-budget reference rows (pre-2025 classics; not counted as 2025+ baselines).

* ``SC``: self-consistency vote share of a response's own answer, n_c / K (Wang et al., ICLR 2023).
* ``SE``: discrete semantic entropy over answer clusters (Farquhar et al., Nature 2024); confidence
  = −H(cluster shares), a question-level score. Responses without an answer form singletons.

Both use the same K responses as CAPE, so they isolate what CAPE adds beyond answer agreement.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from baselines.base import Baseline


class SC(Baseline):
    name = "SC"
    paper = "SC"
    probabilistic = True
    reference = True

    def score(self, q):
        has = [s["pred"] is not None for s in q.samples]
        counts = Counter(c for c, h in zip(q.clusters, has, strict=True) if h)
        return np.array([counts[c] / q.k if h else 0.0 for c, h in zip(q.clusters, has, strict=True)])


class SE(Baseline):
    name = "SE"
    paper = "SE"
    reference = True

    def score(self, q):
        shares = np.array(list(Counter(q.clusters).values()), dtype=float) / q.k
        return np.full(q.k, float((shares * np.log(shares)).sum()))
