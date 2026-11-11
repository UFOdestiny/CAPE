"""CISC (Taubenfeld et al., Findings of ACL 2025, "Confidence Improves Self-Consistency in LLMs").

Per-response confidence c_i = P("1") after the CISC prompt "Now I will rate my confidence in the
proposed answer as either 0 or 1. Proposed confidence: (" (the paper's best P(True) variant).
CISC normalises confidences within a question, c̃ = softmax(c / T), and votes with c̃. As a
response-level score we use the confidence-weighted vote share of the response's own answer
(T → ∞ recovers self-consistency). T is chosen on the validation pool from the paper's grid of 80
log-spaced values in [1e-4, 1e4] by AUROC. ``CISC-PTrue`` is the single-response c_i.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline, select_by_auroc, vote_share

T_GRID = tuple(np.logspace(-4, 4, 80))


def ptrue(q) -> np.ndarray:
    return np.array([np.exp(f["ptrue_logp1"]) if f["ptrue_logp1"] is not None else 0.0 for f in q.followup])


def weighted_share(q, temperature: float) -> np.ndarray:
    c = ptrue(q) / temperature
    w = np.exp(c - c.max())
    return vote_share(w, q.clusters)


class CISC(Baseline):
    paper = "CISC"

    def __init__(self, single: bool = False):
        self.single = single
        self.name = "CISC-PTrue" if single else "CISC"
        self.main = not single
        self.probabilistic = True
        self.temperature = None

    def fit(self, train, val):
        if not self.single:
            self.temperature, _ = select_by_auroc(T_GRID, val, lambda t, q: weighted_share(q, t))

    def score(self, q):
        return ptrue(q) if self.single else weighted_share(q, self.temperature)

    def describe(self):
        return {"temperature": self.temperature}
