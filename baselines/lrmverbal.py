"""LRMVerbal (Yoon et al., 2025, "Reasoning Models Better Express Their Confidence") — shared-trace proxy.

The original method asks for a 10-class verbal confidence inside the solving prompt. To score the
same responses as every other baseline, we use the paper's two-turn variant on the shared trace:
after the complete response a second user turn asks for the Listing-1 confidence class (decoding
constrained to the class names; see ``uqbench.pipeline.followup``). Classes map to bin midpoints
0.05 … 0.95. This is a proxy of the original single-turn method.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline

FALLBACK = 0.5


class LRMVerbal(Baseline):
    name = "LRMVerbal"
    paper = "LRMVerbal"
    probabilistic = True

    def score(self, q):
        return np.array([f["verbal_conf"] if f["verbal_conf"] is not None else FALLBACK for f in q.followup])
