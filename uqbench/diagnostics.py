"""Diagnostics behind the theory, computed from the scored test sets.

For each test set: answer-structure statistics (no-answer rate, clusters per question, unanimous and
unanimous-wrong rates — the latter lower-bounds the Brier score of any normalized vote — and the share of questions
whose correct responses fall into several clusters), the label-free Brier certificate of the projection, and the
mean confidence on unanimous-correct, unanimous-wrong and split questions. Called by ``uqbench.pipeline.evaluate``;
results go to ``results/<model>/{answer_structure,certificate,null_mass}.csv``.
"""
from __future__ import annotations

import numpy as np


def structure_stats(questions) -> dict:
    n_clusters, unanimous, unanimous_wrong, split_gold, no_answer = [], 0, 0, 0, 0
    for q in questions:
        has = [s["pred"] is not None for s in q.samples]
        no_answer += q.k - sum(has)
        ids = {c for c, h in zip(q.clusters, has, strict=True) if h}
        n_clusters.append(len(ids))
        if len(ids) == 1 and all(has):
            unanimous += 1
            unanimous_wrong += int(q.labels.max() == 0)
        correct_ids = {c for c, y in zip(q.clusters, q.labels, strict=True) if y == 1}
        split_gold += int(len(correct_ids) > 1)
    n = len(questions)
    return {"questions": n, "no_answer_rate": no_answer / sum(q.k for q in questions),
            "clusters_per_q": float(np.mean(n_clusters)), "unanimous_rate": unanimous / n,
            "unanimous_wrong_rate": unanimous_wrong / n, "gold_split_rate": split_gold / n}


def labels_coherent(q) -> bool:
    """Whether the realised labels satisfy (A1)-(A2) for this question's clustering."""
    has = np.array([s["pred"] is not None for s in q.samples])
    clusters = np.array(q.clusters)
    if q.labels[~has].any():
        return False
    correct = 0
    for c in set(clusters[has]):
        member = q.labels[(clusters == c) & has]
        if len(set(member)) > 1:
            return False
        correct += int(member.max() == 1)
    return correct <= 1


def certificate(questions, p_before: np.ndarray, p_after: np.ndarray) -> dict:
    """Label-free certificate: for coherent p_after = Π(p_before) and labels satisfying (A1)-(A2), the Brier
    gain on every question is at least ||p_before − p_after||², a label-free lower bound. Returns
    certified and realised gains (per response), how often the bound held overall and on
    questions with coherent labels, and the share of questions with incoherent labels."""
    certified = realised = held = held_coh = n_coh = 0.0
    start = 0
    for q in questions:
        a, b = p_before[start : start + q.k], p_after[start : start + q.k]
        start += q.k
        d = float(((a - b) ** 2).sum())
        g = float(((a - q.labels) ** 2).sum() - ((b - q.labels) ** 2).sum())
        ok = g >= d - 1e-9
        coherent = labels_coherent(q)
        certified, realised, held = certified + d, realised + g, held + ok
        n_coh, held_coh = n_coh + coherent, held_coh + (ok and coherent)
    n = sum(q.k for q in questions)
    return {"certified_gain": certified / n, "realised_gain": realised / n, "bound_held": held / len(questions),
            "bound_held_coherent": held_coh / max(n_coh, 1), "incoherent_labels": 1 - n_coh / len(questions)}


def null_mass_rows(name: str, questions, prob: np.ndarray, method: str) -> list[dict]:
    """Mean confidence a method assigns on unanimous-correct, unanimous-wrong and split questions
    (normalized votes cannot tell unanimous-wrong from unanimous-correct questions)."""
    groups: dict[str, list[float]] = {"unanimous_correct": [], "unanimous_wrong": [], "split": []}
    start = 0
    for q in questions:
        p = prob[start : start + q.k]
        start += q.k
        has = [s["pred"] is not None for s in q.samples]
        unanimous = all(has) and len(set(q.clusters)) == 1
        key = ("unanimous_correct" if q.labels.max() == 1 else "unanimous_wrong") if unanimous else "split"
        groups[key].append(float(p.mean()))
    return [{"dataset": name, "method": method, "group": g, "questions": len(v),
             "mean_confidence": float(np.mean(v)) if v else float("nan")} for g, v in groups.items()]
