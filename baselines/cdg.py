"""CDG (Wang et al., ICML 2026, "Inference Time Optimization with Confidence Dynamics").

Trace score s = C̄ + β·ΔC with DeepConf token confidence C, ΔC = mean(C over the last 10% of
tokens) − mean(C over the first 10%) (token windows, as in the official code). β follows the
paper's rule β = r_b = μ_C / |μ⁺_ΔC − μ⁻_ΔC| estimated on the training pool. ``CDG-Vote`` turns
the answer-level score R(a) = |T_a|^α · mean_{ℓ∈T_a} s_ℓ (α = 0.5) into a normalised share for
each response's answer; responses without an answer get 0 (they do not vote in the paper).
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline, labels_of

TAIL = 0.10
ALPHA = 0.5


def dynamics(conf: np.ndarray) -> tuple[float, float]:
    """(mean confidence, head→tail gain)."""
    n_cut = max(1, int(len(conf) * TAIL))
    return float(conf.mean()), float(conf[-n_cut:].mean() - conf[:n_cut].mean())


class CDG(Baseline):
    paper = "CDG"

    def __init__(self, mode: str = "trace"):
        self.mode = mode  # "trace" | "gain" | "vote"
        self.name = {"trace": "CDG", "gain": "CDG-ΔC", "vote": "CDG-Vote"}[mode]
        self.main = mode == "trace"
        self.beta = None

    def fit(self, train, val):
        feats = np.array([dynamics(r["conf_topk"]) for q in train for r in q.replay])
        y = labels_of(train).astype(bool)
        gap = abs(feats[y, 1].mean() - feats[~y, 1].mean())
        self.beta = float(feats[:, 0].mean() / max(gap, 1e-8))

    def _trace(self, q) -> np.ndarray:
        feats = np.array([dynamics(r["conf_topk"]) for r in q.replay])
        return feats[:, 0] + self.beta * feats[:, 1]

    def score(self, q):
        if self.mode == "gain":
            return np.array([dynamics(r["conf_topk"])[1] for r in q.replay])
        s = self._trace(q)
        if self.mode == "trace":
            return s
        has_answer = np.array([smp["pred"] is not None for smp in q.samples])
        groups = {}
        for i, c in enumerate(q.clusters):
            if has_answer[i]:
                groups.setdefault(c, []).append(s[i])
        r = {c: len(v) ** ALPHA * float(np.mean(v)) for c, v in groups.items()}
        total = sum(max(v, 0.0) for v in r.values())
        return np.array([max(r[c], 0.0) / total if has_answer[i] and total > 0 else 0.0
                         for i, c in enumerate(q.clusters)])

    def describe(self):
        return {"beta": self.beta, "tail_fraction": TAIL, "alpha": ALPHA}
