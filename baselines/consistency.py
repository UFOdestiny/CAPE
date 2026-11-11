"""Sample-consistency baselines.

* ``DeepConf-Vote`` (Fu et al., 2025, "Deep Think with Confidence", offline mode): keep the top η% of a
  question's traces by trace confidence and vote with confidence weights; a response's score is the
  weighted share of its own answer among the kept traces. The trace-confidence variant and η are chosen
  on the validation pool by AUROC (the paper's η = 10% and 90%, plus 50% and 100% = no filtering, since
  K is far smaller than the paper's 512 traces).
* ``ConsistCal`` (Lyu et al., AAAI 2025, "Calibrating Large Language Models with Sample Consistency"):
  confidence from the consistency of the K samples, with the paper's three measures adapted to a
  response: agreement (share of the response's answer), first-second distance (share of its answer
  minus that of the most frequent other answer) and entropy (negative normalised entropy of the answer
  distribution). The measure is chosen on the validation pool by AUROC; probabilities come from the
  shared Platt calibration.

Responses without an extractable answer score 0 in both.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from baselines.base import Baseline, select_by_auroc
from baselines.deepconf import VARIANTS, trace_confidence

ETAS = (0.1, 0.5, 0.9, 1.0)


def _answered(q) -> np.ndarray:
    return np.array([s["pred"] is not None for s in q.samples])


class DeepConfVote(Baseline):
    name = "DeepConf-Vote"
    paper = "DeepConf"
    probabilistic = True

    def __init__(self):
        self.config = None
        self.val_auroc = {}

    def _score(self, config, q) -> np.ndarray:
        variant, eta = config
        has = _answered(q)
        conf = np.array([trace_confidence(r["conf_topk"], variant) for r in q.replay])
        idx = np.flatnonzero(has)
        if len(idx) == 0:
            return np.zeros(q.k)
        keep = idx[np.argsort(-conf[idx], kind="stable")[: max(1, int(np.ceil(eta * len(idx))))]]
        weight = Counter()
        for i in keep:
            weight[q.clusters[i]] += max(conf[i], 0.0)
        total = sum(weight.values())
        if total <= 0:
            return np.where(has, 1.0 / len(idx), 0.0)
        return np.array([weight[c] / total if h else 0.0 for c, h in zip(q.clusters, has, strict=True)])

    def fit(self, train, val):
        grid = [(v, e) for v in VARIANTS for e in ETAS]
        self.config, table = select_by_auroc(grid, val, self._score)
        self.val_auroc = {f"{v}/{e}": a for (v, e), a in table.items()}

    def score(self, q):
        return self._score(self.config, q)

    def describe(self):
        return {"variant": self.config[0], "eta": self.config[1], "val_auroc": self.val_auroc}


class ConsistCal(Baseline):
    name = "ConsistCal"
    paper = "ConsistCal"

    MEASURES = ("agreement", "fsd", "entropy")

    def __init__(self):
        self.measure = None
        self.val_auroc = {}

    def _score(self, measure, q) -> np.ndarray:
        has = _answered(q)
        counts = Counter(c for c, h in zip(q.clusters, has, strict=True) if h)
        if not counts:
            return np.zeros(q.k)
        if measure == "entropy":
            p = np.array(list(counts.values()), dtype=float) / sum(counts.values())
            h = -(p * np.log(p)).sum() / np.log(q.k)
            return np.where(has, -h, -1.0)
        out = []
        for c, h in zip(q.clusters, has, strict=True):
            if not h:
                out.append(-1.0 if measure == "fsd" else 0.0)
                continue
            own = counts[c] / q.k
            if measure == "agreement":
                out.append(own)
            else:
                other = max((n for cc, n in counts.items() if cc != c), default=0) / q.k
                out.append(own - other)
        return np.array(out)

    def fit(self, train, val):
        self.measure, self.val_auroc = select_by_auroc(self.MEASURES, val, self._score)

    def score(self, q):
        return self._score(self.measure, q)

    def describe(self):
        return {"measure": self.measure, "val_auroc": self.val_auroc}
