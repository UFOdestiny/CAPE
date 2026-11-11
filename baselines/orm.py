"""Outcome reward model (ORM): a general open reward model scores each (question, response) pair.

Skywork-Reward-V2-Llama-3.1-8B (Liu et al., 2025, "Skywork-Reward-V2: Scaling Preference Data Curation
via Human-AI Synergy") reads the prompt and the backbone's visible answer (``uqbench.pipeline.reward``); its scalar
reward is the response's score, calibrated on the validation pool like every other estimator.
Training-free with respect to our pool.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline


class ORM(Baseline):
    name = "ORM"
    paper = "SkyworkRewardV2"

    def score(self, q):
        if len(q.reward) != q.k or any(r is None for r in q.reward):
            raise ValueError(f"no reward-model scores for {q.uid}; run python -m uqbench.pipeline.reward")
        return np.asarray(q.reward, dtype=float)
