"""Simple fusions of existing signals (CPU only).

* ``DeepConf×SC``: the validation-calibrated DeepConf probability of a response times the vote share of
  its answer, the plainest way to inject agreement into a single-trace score.
* ``CoCoA`` (Vashurin et al., 2025, "Uncertainty Quantification for LLMs through Minimum Bayes Risk:
  Bridging Confidence and Consistency"): uncertainty u(y_k) * mean_j (1 - s(y_k, y_j)), the product of a
  response's own confidence-based uncertainty and its expected dissimilarity to the other samples.
  u is the mean token negative log-probability (the paper's perplexity variant) and s is either
  ROUGE-L between final responses or answer equivalence; the choice is made by validation AUROC.
  Score = -uncertainty.
* ``Stacker``: a supervised per-response logistic regression on the 22 trace statistics CAPE uses plus
  five vote features (share of the response's answer, top share, whether it is in the top cluster,
  clusters per sample, answered), with the L2 strength chosen by validation log loss. It has neither
  CAPE's answer-level likelihood nor the projection or the hidden-state gate.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline, labels_of
from baselines.deepconf import DeepConf
from baselines.reference import SC
from baselines.stacker import FeatureStacker


class DeepConfSC(Baseline):
    name = "DeepConf×SC"
    paper = "DeepConf"
    probabilistic = True

    def __init__(self):
        self.deepconf, self.sc = DeepConf(), SC()

    def fit(self, train, val):
        from sklearn.linear_model import LogisticRegression

        self.deepconf.fit(train, val)
        s = self.deepconf.score_all(val)[:, None]
        self.mu, self.sd = s.mean(), s.std() + 1e-9
        self.platt = LogisticRegression(C=1e6, max_iter=1000).fit((s - self.mu) / self.sd, labels_of(val))

    def score(self, q):
        s = (self.deepconf.score(q)[:, None] - self.mu) / self.sd
        return self.platt.predict_proba(s)[:, 1] * self.sc.score(q)

    def describe(self):
        return {"variant": self.deepconf.variant}


class CoCoA(Baseline):
    name = "CoCoA"
    paper = "CoCoA"

    def __init__(self):
        from baselines.snne import SNNE

        self.sims = {"rougeL": SNNE("rougeL"), "answer": SNNE("math_eq")}
        self.similarity = None
        self.val_auroc = {}

    def _score(self, similarity, q):
        nll = np.array([-float(r["chosen_lp"].mean()) for r in q.replay])
        s = self.sims[similarity]._matrix(q)
        k = q.k
        dissim = (k - s.sum(axis=1)) / max(k - 1, 1)   # mean over the other samples of 1 - s (s_ii = 1)
        return -nll * np.clip(dissim, 0.0, None)

    def fit(self, train, val):
        from baselines.base import select_by_auroc

        self.similarity, self.val_auroc = select_by_auroc(tuple(self.sims), val, self._score)

    def score(self, q):
        return self._score(self.similarity, q)

    def describe(self):
        return {"similarity": self.similarity, "val_auroc": self.val_auroc}


class Stacker(FeatureStacker):
    name = "Stacker"
    paper = "Stacker"

    def __init__(self):
        super().__init__("none", "Stacker")

    def describe(self):
        return {"C": self.plain["C"]}
