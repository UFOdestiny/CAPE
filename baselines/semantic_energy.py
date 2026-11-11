"""Semantic energy (Ma et al., 2025, "Semantic Energy: Detecting LLM Hallucination Beyond Entropy").

The energy of a token is its negative raw logit, the energy of a response the mean over the tokens of its final
answer (the reasoning is discarded, as in the paper's think mode), and a response is scored by the total energy
of its answer cluster, E(C) = sum of its members' energies; lower energy means higher confidence, so the score
is -E(C). We divide by K (mean energy times vote share): the ranking within a question set is unchanged, and the
pool (K=4) and the tests (K=8) share one scale, so the calibrator fitted on the pool transfers. Raw logits come
from ``uqbench.pipeline.energy``. Responses without an answer form singleton clusters. Training-free; calibrated on the
validation pool like every other estimator.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline


class SemEnergy(Baseline):
    name = "SemEnergy"
    paper = "SemEnergy"

    def score(self, q):
        if len(q.energy) != q.k:
            raise ValueError(f"no raw logits for {q.uid}; run python -m uqbench.pipeline.energy")
        logit = np.nan_to_num(np.asarray(q.energy, dtype=float), nan=0.0)   # empty responses add no evidence
        total = {}
        for c, v in zip(q.clusters, logit, strict=True):
            total[c] = total.get(c, 0.0) + v
        return np.array([total[c] for c in q.clusters]) / q.k
