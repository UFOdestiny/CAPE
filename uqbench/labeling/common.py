"""Helpers shared by the labeling stages."""
from __future__ import annotations

import numpy as np


def remap_judge(row: dict, old: list[int], new: list[int]) -> dict:
    """Map an answer-set judge output onto merged cluster ids; merged clusters add their probabilities."""
    to_new = {o: n for o, n in zip(old, new, strict=True)}
    acc: dict[int, float] = {}
    for c, lp in zip(row["clusters"], row["logp"], strict=True):
        n = to_new[c]
        acc[n] = float(np.logaddexp(acc[n], lp)) if n in acc else lp
    return {**row, "clusters": list(acc), "logp": list(acc.values())}
