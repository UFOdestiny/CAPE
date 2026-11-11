"""Post-hoc calibration maps fitted on the validation pool (identical procedure for every method)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class Platt:
    """Logistic map p = σ(a · z + b) on the standardised score z (Platt, 1999)."""

    def fit(self, scores, labels) -> Platt:
        scores = np.asarray(scores, dtype=float)
        self.mu, self.sd = scores.mean(), scores.std() or 1.0
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(scores)[:, None], labels)
        return self

    def _z(self, scores) -> np.ndarray:
        return (np.asarray(scores, dtype=float) - self.mu) / self.sd

    def __call__(self, scores) -> np.ndarray:
        return self.lr.predict_proba(self._z(scores)[:, None])[:, 1]


class Identity:
    """Use a method's native probabilities unchanged (clipped to [0, 1])."""

    def __call__(self, scores) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def _nll(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_calibrator(scores, labels, groups, probabilistic: bool):
    """Platt for score-valued methods. For probability-valued methods, keep the native
    probabilities unless 2-fold cross-fitted Platt (folds by question) has lower validation NLL.
    Returns (calibrator fitted on all validation data, name)."""
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels, float), np.asarray(groups)
    if not probabilistic:
        return Platt().fit(scores, labels), "platt"
    fold = np.isin(groups, np.unique(groups)[::2])
    crossfit = np.empty_like(scores)
    for a, b in ((fold, ~fold), (~fold, fold)):
        crossfit[b] = Platt().fit(scores[a], labels[a])(scores[b])
    if _nll(crossfit, labels) < _nll(Identity()(scores), labels):
        return Platt().fit(scores, labels), "platt"
    return Identity(), "identity"
