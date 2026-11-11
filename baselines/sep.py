"""Semantic entropy probes (Kossen et al., 2024, "Semantic Entropy Probes: Robust and Cheap
Hallucination Detection in LLMs").

A linear probe on a single response's hidden states predicts whether its question has high semantic
entropy; no correctness labels are used to train it. Training targets come from the pool: the discrete
semantic entropy of each pool question's K answer clusters, binarised at the median over training
questions (the paper's best-split threshold, here the median). Features are the answer-position hidden
states of a middle and the last stored layer (as for CAPE's probe), standardised and reduced to their
top principal components; the probe is a logistic regression whose strength is chosen by validation
AUROC for correctness. Score = -P(high semantic entropy), a single-trace estimator at test time.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from baselines.base import Baseline, labels_of

COMPONENTS = 256
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
SEED = 0


def _features(q) -> np.ndarray:
    from cape.model import hidden_features

    return hidden_features(q)


def semantic_entropy(q) -> float:
    shares = np.array(list(Counter(q.clusters).values()), dtype=float) / q.k
    return float(-(shares * np.log(shares)).sum())


class SEP(Baseline):
    name = "SEP"
    paper = "SEP"

    def fit(self, train, val):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        from uqbench.metrics import auroc

        h = np.concatenate([_features(q) for q in train])
        se = np.array([semantic_entropy(q) for q in train])
        self.threshold = float(np.median(se))
        target = np.repeat(se > self.threshold, [q.k for q in train]).astype(int)
        if target.min() == target.max():   # degenerate split (e.g. median at the maximum): use >=
            target = np.repeat(se >= self.threshold, [q.k for q in train]).astype(int)
        self.mu, self.sd = h.mean(0), h.std(0) + 1e-6
        self.pca = PCA(min(COMPONENTS, h.shape[1]), svd_solver="randomized", random_state=SEED)
        z = self.pca.fit_transform((h - self.mu) / self.sd)
        zv = self.pca.transform((np.concatenate([_features(q) for q in val]) - self.mu) / self.sd)
        yv = labels_of(val)
        best = None
        self.val_auroc = {}
        if target.min() == target.max():   # no variation in semantic entropy: constant score
            self.C, self.model, self.val_auroc = None, None, {}
            return
        for c in C_GRID:
            m = LogisticRegression(C=c, max_iter=3000).fit(z, target)
            a = auroc(-m.decision_function(zv), yv)
            self.val_auroc[c] = a
            if best is None or a > best[0]:
                best = (a, c, m)
        _, self.C, self.model = best

    def score(self, q):
        if self.model is None:
            return np.zeros(q.k)
        z = self.pca.transform((_features(q) - self.mu) / self.sd)
        return -self.model.decision_function(z)

    def describe(self):
        return {"C": self.C, "se_threshold": self.threshold, "val_auroc": self.val_auroc}
