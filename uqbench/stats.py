"""Statistics shared by the reports: paired bootstrap, multiple testing and the agreement-floor rate."""
from __future__ import annotations

import numpy as np

from uqbench import metrics
from uqbench.config import load_yaml, scores_dir


def paired_bootstrap(model: str, dataset: str, a: str, b: str, n_boot: int, seed: int, metric: str = "auroc") -> dict:
    """Question-level paired bootstrap of the improvement of method a over method b (both score the same
    responses): AUROC(a) − AUROC(b) on raw scores, or Brier(b) − Brier(a) on calibrated probabilities, so a
    positive delta always favours a."""
    sets = load_yaml("benchmark")["math_sets"] if dataset == "math_all" else [dataset]
    paths = [scores_dir(model) / f"{d}.npz" for d in sets]
    zs = [np.load(p) for p in paths if p.exists()]  # partial runs (ablations) pool only what they have
    y = np.concatenate([z["labels"] for z in zs])
    key = "" if metric == "auroc" else "__prob"
    sa, sb = np.concatenate([z[a + key] for z in zs]), np.concatenate([z[b + key] for z in zs])
    gain = ((lambda s1, s2, t: metrics.auroc(s1, t) - metrics.auroc(s2, t)) if metric == "auroc"
            else (lambda s1, s2, t: metrics.brier(s2, t) - metrics.brier(s1, t)))
    uid = np.concatenate([z["uid"] for z in zs])
    _, inv = np.unique(uid, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(gain(sa[idx], sb[idx], y[idx]))
    diffs = np.array(diffs)
    return {"delta": gain(sa, sb, y), "lo": np.percentile(diffs, 2.5),
            "hi": np.percentile(diffs, 97.5), "p_le0": float((diffs <= 0).mean())}


def bh_threshold(tests: list[tuple[float, float]], q: float = 0.05) -> float:
    """Largest p-value rejected by Benjamini-Hochberg at level q (-1 if none)."""
    t = sorted(p for p, _ in tests)
    ok = [p for i, p in enumerate(t, 1) if p <= q * i / len(t)]
    return max(ok) if ok else -1.0
