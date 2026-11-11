"""Single-trace statistics of a response, computed from the replayed token statistics and the follow-up probes.

CAPE's default evidence is the hidden-state probe (``cape.model``); these statistics enter its answer model only in
the ablations that add trace evidence (``AnswerModel(features=...)``), and they are shared with the feature-matched
controls. Token confidence C_i is the negative mean of the top-20 log-probabilities at step i (DeepConf, Fu et al.,
2025); window statistics use groups of ``WINDOW`` tokens with stride 1, and the head/tail gain uses the first and
last ``TAIL`` fraction of the trace (CDG, Wang et al., 2026).
"""
from __future__ import annotations

import numpy as np

WINDOW = 2048   # sliding-window group size of the DeepConf trace confidences
TAIL = 0.10     # head / tail fraction of the confidence-dynamics gain

CORE_FEATURES = ["self_certainty", "entropy", "mean_logprob", "deepconf_bottom10", "deepconf_tail", "cdg_gain",
                 "log_length", "trace_closed", "ptrue_logodds", "verbal_conf"]
TRACE_FEATURES = ["answer_logprob", "answer_min_logprob", "answer_entropy", "answer_tokens", "tail_entropy",
                  "tail_self_certainty", "tail_logprob", "head_entropy", "deepconf_lowest", "deepconf_mean",
                  "low_prob_fraction", "entropy_q90"]
FEATURES = CORE_FEATURES + TRACE_FEATURES   # every single-trace statistic (controls and ablations)


def group_confidences(conf: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(conf) <= window:
        return np.array([conf.mean()])
    csum = np.concatenate([[0.0], np.cumsum(conf, dtype=np.float64)])
    return (csum[window:] - csum[:-window]) / window


def trace_confidence(conf: np.ndarray, variant: str) -> float:
    if variant == "mean":
        return float(conf.mean())
    if variant == "tail":
        return float(conf[-WINDOW:].mean())
    groups = group_confidences(conf)
    if variant == "lowest":
        return float(groups.min())
    if variant == "bottom10":
        m = max(1, int(np.ceil(0.1 * len(groups))))
        return float(np.sort(groups)[:m].mean())
    raise ValueError(variant)


def dynamics(conf: np.ndarray) -> tuple[float, float]:
    """(mean confidence, head→tail gain)."""
    n_cut = max(1, int(len(conf) * TAIL))
    return float(conf.mean()), float(conf[-n_cut:].mean() - conf[:n_cut].mean())


def trace_statistics(r: dict) -> list[float]:
    """Final-answer segment (after the reasoning block), last/first 10% of the trace, whole-trace tails."""
    lp, ent, sc, conf = r["chosen_lp"], r["entropy"], r["self_certainty"], r["conf_topk"]
    n = len(lp)
    end = min(max(int(r["positions"]["think_end"]), 0), n)
    seg = slice(end, n) if n - end >= 2 else slice(max(n - 32, 0), n)
    tail, head = slice(int(n * 0.9), n), slice(0, max(int(n * 0.1), 1))
    return [float(lp[seg].mean()), float(lp[seg].min()), float(ent[seg].mean()), float(n - end),
            float(ent[tail].mean()), float(sc[tail].mean()), float(lp[tail].mean()), float(ent[head].mean()),
            trace_confidence(conf, "lowest"), trace_confidence(conf, "mean"),
            float((lp < np.log(0.1)).mean()), float(np.quantile(ent, 0.9))]


def scalar_features(q) -> np.ndarray:
    """(K, len(FEATURES)) single-trace features, memoised on the Question object (window statistics
    over long traces are expensive; subsets are new objects, so they never share a cache)."""
    cached = getattr(q, "_cape_features", None)
    if cached is not None:
        return cached
    rows = []
    for s, r, f in zip(q.samples, q.replay, q.followup, strict=True):
        conf = r["conf_topk"]
        lp1 = f["ptrue_logp1"] if f["ptrue_logp1"] is not None else -20.0
        lp0 = f["ptrue_logp0"] if f["ptrue_logp0"] is not None else -20.0
        rows.append([
            float(r["self_certainty"].mean()), float(r["entropy"].mean()), float(r["chosen_lp"].mean()),
            trace_confidence(conf, "bottom10"), trace_confidence(conf, "tail"), dynamics(conf)[1],
            float(np.log(r["length"])), float(s["closed"]), lp1 - lp0,
            f["verbal_conf"] if f["verbal_conf"] is not None else 0.5,
            *trace_statistics(r),
        ])
    out = np.array(rows, dtype=np.float64)
    q._cape_features = out
    return out
