"""Feature-matched logistic stacker shared by baselines and research controls."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from baselines.base import Baseline, labels_of

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def agreement_features(q) -> np.ndarray:
    has = np.array([s["pred"] is not None for s in q.samples])
    counts = Counter(c for c, h in zip(q.clusters, has, strict=True) if h)
    share = np.array([counts[c] / q.k if h else 0.0 for c, h in zip(q.clusters, has, strict=True)])
    top = share.max()
    return np.c_[share, np.full(q.k, top), (share == top) & has, np.full(q.k, len(counts) / q.k), has]


class FeatureStacker(Baseline):
    probabilistic = True

    def __init__(self, probe: str = "none", name: str = "Stacker"):
        self.probe_mode = probe      # "none" | "always" | "gated"
        self.name = name

    def _x(self, q, with_probe: bool) -> np.ndarray:
        from cape.model import scalar_features

        x = np.concatenate([scalar_features(q), agreement_features(q)], axis=1)
        if with_probe:
            x = np.concatenate([x, self.probe.logits(q)[:, None]], axis=1)
        return x

    def _fit_lr(self, train, val, with_probe):
        x = np.concatenate([self._x(q, with_probe) for q in train])
        xv = np.concatenate([self._x(q, with_probe) for q in val])
        y, yv = labels_of(train), labels_of(val)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        best = None
        for c in C_GRID:
            m = LogisticRegression(C=c, max_iter=5000).fit((x - mu) / sd, y)
            p = np.clip(m.predict_proba((xv - mu) / sd)[:, 1], 1e-6, 1 - 1e-6)
            nll = float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
            if best is None or nll < best[0]:
                best = (nll, c, m)
        return {"mu": mu, "sd": sd, "model": best[2], "C": best[1]}

    def fit(self, train, val) -> None:
        self.formats = {q.task for q in train}
        if self.probe_mode != "none":
            from cape.model import HiddenProbe

            self.probe = HiddenProbe().fit(train, val)
            self.with_probe = self._fit_lr(train, val, True)
        if self.probe_mode != "always":
            self.plain = self._fit_lr(train, val, False)

    def _logit(self, q, with_probe: bool) -> np.ndarray:
        m = self.with_probe if with_probe else self.plain
        return m["model"].decision_function((self._x(q, with_probe) - m["mu"]) / m["sd"])

    def score(self, q) -> np.ndarray:
        if self.probe_mode != "gated":
            return 1 / (1 + np.exp(-self._logit(q, self.probe_mode == "always")))
        # CAPE's support gate: format check times the conformal test of the hidden-state residual,
        # pooling the two models' logits with weight w (the logarithmic pool of a binary outcome)
        from cape.model import SUPPORT_ALPHA

        w = float(q.task in self.formats) * min(1.0, self.probe.p_value(q) / SUPPORT_ALPHA)
        z = self._logit(q, False) if w == 0 else (1 - w) * self._logit(q, False) + w * self._logit(q, True)
        return 1 / (1 + np.exp(-z))
