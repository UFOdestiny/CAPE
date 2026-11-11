"""Self-Certainty (Kang et al., NeurIPS 2025, "Scalable Best-of-N Selection ... via Self-Certainty").

Response score = mean over generated tokens of KL(U ‖ p_i) = −(1/V) Σ_j log(V · p_i(j)), computed
from the full-vocabulary next-token distribution in the replay stage (V = tokenizer vocabulary,
padding rows of the output matrix excluded). Training-free.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline


class SelfCertainty(Baseline):
    name = "SelfCertainty"
    paper = "SelfCertainty"

    def score(self, q):
        return np.array([float(r["self_certainty"].mean()) for r in q.replay])
