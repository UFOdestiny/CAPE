"""Common interface for response-level confidence estimators (CAPE and the optional baselines).

An estimator maps each :class:`~uqbench.records.Question` to K confidence scores (one per sampled response,
higher = more likely correct). Learned or tuned components are fitted only on the training pool (``fit``); test
questions are scored without access to labels.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from uqbench.records import Question


class Estimator:
    name: str = ""
    paper: str = ""            # short citation key ("ours" for CAPE)
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
