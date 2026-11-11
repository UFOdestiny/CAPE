"""Evaluation metrics for confidence scores (higher score = more likely correct).

Discrimination: AUROC, AUPRC of errors. Selective prediction: AURC with exact expected value
under random tie-breaking (so vote-fraction ties never depend on row order). Probabilistic
quality (requires scores in [0, 1]): Brier, ECE, NLL.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_EPS = 1e-6


def _arrays(conf, labels):
    conf = np.asarray(conf, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if conf.shape != labels.shape:
        raise ValueError(f"shape mismatch: {conf.shape} vs {labels.shape}")
    if not np.all(np.isfinite(conf)):
        raise ValueError("confidence scores must be finite")
    return conf, labels


def accuracy(labels) -> float:
    return float(np.mean(labels))


def auroc(conf, labels) -> float:
    """Mann–Whitney AUROC (ties count 1/2); NaN when only one class is present."""
    conf, labels = _arrays(conf, labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, conf))


def auprc_error(conf, labels) -> float:
    """Average precision for detecting errors, ranking by low confidence."""
    conf, labels = _arrays(conf, labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(1 - labels, -conf))


def brier(prob, labels) -> float:
    prob, labels = _arrays(prob, labels)
    return float(np.mean((prob - labels) ** 2))


def nll(prob, labels) -> float:
    prob, labels = _arrays(prob, labels)
    p = np.clip(prob, _EPS, 1 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def ece(prob, labels, n_bins: int = 15, strategy: str = "uniform") -> float:
    """Expected calibration error with equal-width (``uniform``) or equal-mass (``quantile``) bins."""
    prob, labels = _arrays(prob, labels)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(prob, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError(strategy)
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(labels[mask].mean() - prob[mask].mean())
    return float(total)


def risk_coverage(conf, labels) -> tuple[np.ndarray, np.ndarray]:
    """Expected selective risk at coverage k/N, k = 1..N, under random tie-breaking."""
    conf, labels = _arrays(conf, labels)
    order = np.argsort(-conf, kind="stable")
    s, err = conf[order], 1.0 - labels[order]
    n = len(s)
    cum_err = np.empty(n)
    start, before = 0, 0.0
    while start < n:
        end = start
        while end + 1 < n and s[end + 1] == s[start]:
            end += 1
        size, group_err = end - start + 1, err[start : end + 1].sum()
        cum_err[start : end + 1] = before + group_err * np.arange(1, size + 1) / size
        before += group_err
        start = end + 1
    k = np.arange(1, n + 1)
    return k / n, cum_err / k


def aurc(conf, labels) -> float:
    return float(np.mean(risk_coverage(conf, labels)[1]))


PROB_METRICS = {"brier": brier, "ece": ece, "nll": nll}
RANK_METRICS = {"auroc": auroc, "aurc": aurc, "auprc_err": auprc_error}


def evaluate(conf, labels, prob=None) -> dict[str, float]:
    """Ranking metrics from ``conf``; probability metrics from ``prob`` (if given)."""
    out = {"n": int(len(labels)), "acc": accuracy(labels)}
    out.update({name: fn(conf, labels) for name, fn in RANK_METRICS.items()})
    if prob is not None:
        out.update({name: fn(prob, labels) for name, fn in PROB_METRICS.items()})
    return out
