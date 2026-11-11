"""Common interface for response-level confidence estimators.

A baseline maps each :class:`~uqbench.records.Question` to K confidence scores (one per sampled
response, higher = more likely correct). Learned or tuned components are fitted only on the
training pool (``fit``); test questions are scored without access to labels.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from uqbench.metrics import auroc
from uqbench.records import Question


class Baseline:
    name: str = ""
    paper: str = ""            # short citation key, see docs/baselines/<name>.md
    probabilistic = False      # True if the raw score is meant as P(correct)
    main = True                # shown in the main table (False = variant / ablation row)
    reference = False          # matched-budget classic reference row (not a 2025+ baseline)

    def fit(self, train: list[Question], val: list[Question]) -> None:
        """Fit learned parts or select hyper-parameters (labels of the pool only)."""

    def score(self, q: Question) -> np.ndarray:
        raise NotImplementedError

    def score_all(self, questions: Iterable[Question]) -> np.ndarray:
        return np.concatenate([np.asarray(self.score(q), dtype=float) for q in questions])

    def describe(self) -> dict:
        """Fitted hyper-parameters, recorded next to the results."""
        return {}


def labels_of(questions: Iterable[Question]) -> np.ndarray:
    return np.concatenate([q.labels for q in questions])


def select_by_auroc(candidates: dict, questions: list[Question], score_fn) -> tuple:
    """Return (best_key, table) maximising validation AUROC of ``score_fn(key, q)``."""
    y = labels_of(questions)
    table = {key: auroc(np.concatenate([score_fn(key, q) for q in questions]), y) for key in candidates}
    best = max(table, key=lambda k: (np.nan_to_num(table[k], nan=-1.0)))
    return best, table


def vote_share(weights: np.ndarray, clusters: list[int]) -> np.ndarray:
    """Weighted share of each response's answer cluster among all K responses."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return np.full(len(clusters), 1.0 / len(clusters))
    share = {}
    for w, c in zip(weights, clusters, strict=True):
        share[c] = share.get(c, 0.0) + w
    return np.array([share[c] / total for c in clusters])
