"""SNNE (Nguyen et al., Findings of ACL 2025, "Beyond Semantic Entropy ... Pairwise Semantic Similarity").

SNNE(q) = −(1/n) Σ_i log Σ_j exp(f(a_i, a_j) / τ), τ = 1, self-similarity included (official
default). Confidence = −SNNE, shared by all responses of a question (question-level score). The sum over
j runs over the K responses, so the score shifts by log K between the K=4 pool (where its calibration
map is fitted) and K=8 test questions; we subtract log K (a log-mean), which leaves AUROC at fixed K
unchanged and makes the calibration map transfer across K.
``SNNE`` uses the paper's main similarity, ROUGE-L F1, on the final responses after ``</think>``
(empty when the trace was cut off). ``SNNE-MathEq`` uses 0/1 answer equivalence (math_verify /
option letter), under which SNNE behaves like discrete semantic entropy.
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz.distance import LCSseq
from scipy.special import logsumexp

from baselines.base import Baseline

TAU = 1.0


def rouge_tokens(text: str) -> list[str]:
    """rouge_score's default tokenizer: lowercase alphanumeric runs."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    lcs = LCSseq.similarity(a, b)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)


class SNNE(Baseline):
    paper = "SNNE"

    def __init__(self, similarity: str = "rougeL"):
        self.similarity = similarity
        self.name = "SNNE" if similarity == "rougeL" else "SNNE-MathEq"
        self.main = similarity == "rougeL"

    def _matrix(self, q) -> np.ndarray:
        if self.similarity == "math_eq":
            c = np.array(q.clusters)
            has = np.array([s["pred"] is not None for s in q.samples])
            return ((c[:, None] == c[None, :]) & has[:, None] & has[None, :]).astype(float)
        toks = [rouge_tokens(t) for t in q.responses()]
        n = len(toks)
        f = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                f[i, j] = f[j, i] = rouge_l(toks[i], toks[j])
        return f

    def score(self, q):
        conf = float(np.mean(logsumexp(self._matrix(q) / TAU, axis=1))) - np.log(q.k)
        return np.full(q.k, conf)
