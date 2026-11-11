"""Projection of response-level correctness forecasts onto the coherent set (paper: "Projection with a Brier
Guarantee").

For one question with K responses, the correctness vector y ∈ {0,1}^K is *coherent*: responses with
equivalent answers share one label, responses without an extractable answer are incorrect, and at
most one answer cluster is correct (single-answer tasks). The convex hull of coherent labels is

    C = { q ∈ [0,1]^K : q constant within clusters, q_i = 0 if no answer, Σ_clusters q_c ≤ 1 }.

The Euclidean projection onto C never increases the response-level Brier score for any coherent
outcome (Pythagorean inequality), so it is a label-free improvement for any estimator.
"""
from __future__ import annotations

import numpy as np

from uqbench.estimator import Estimator


def _project_capped_simplex(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """argmin_q Σ_c w_c (q_c − m_c)^2  s.t.  q ≥ 0, Σ_c q_c ≤ 1."""
    q = np.clip(means, 0.0, 1.0)
    if q.sum() <= 1.0:
        return q
    lo, hi = 0.0, float(np.max(weights * means))
    for _ in range(100):  # bisection on the multiplier λ of Σ q = 1
        lam = 0.5 * (lo + hi)
        if np.maximum(means - lam / weights, 0.0).sum() > 1.0:
            lo = lam
        else:
            hi = lam
    return np.maximum(means - hi / weights, 0.0)


def project_coherent(p: np.ndarray, clusters: list[int], has_answer: np.ndarray) -> np.ndarray:
    """Euclidean projection of response probabilities ``p`` onto the coherent set C."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    ids = sorted({c for c, h in zip(clusters, has_answer, strict=True) if h})
    if not ids:
        return out
    members = [np.array([i for i, c in enumerate(clusters) if c == cid and has_answer[i]]) for cid in ids]
    weights = np.array([len(m) for m in members], dtype=float)
    means = np.array([p[m].mean() for m in members])
    for m, q in zip(members, _project_capped_simplex(means, weights), strict=True):
        out[m] = q
    return out


class CoherentProjection(Estimator):
    """Wrap a fitted estimator and its calibration map: calibrate, then project onto C (label-free)."""

    probabilistic = True
    main = False

    def __init__(self, inner: Estimator, calibrator):
        self.inner, self.calibrator = inner, calibrator
        self.name = f"{inner.name}+Coh"
        self.paper = inner.paper

    def score(self, q):
        p = self.calibrator(np.asarray(self.inner.score(q), dtype=float))
        has_answer = np.array([s["pred"] is not None for s in q.samples])
        return project_coherent(p, q.clusters, has_answer)
