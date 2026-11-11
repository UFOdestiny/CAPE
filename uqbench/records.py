"""Per-question view joining generations, replay features, follow-up probes and answer clusters.

This is the only interface baselines see: each :class:`Question` holds the K sampled responses of
one problem together with everything observed about them at inference time. Gold answers are
kept only in ``labels`` (used for training/evaluation, never by a scorer).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from uqbench.artifacts import load_energy, load_followup, load_replay, load_reward
from uqbench.config import load_yaml, run_dir, set_parts
from uqbench.io import read_jsonl, write_jsonl
from uqbench.verify import cluster_answers, split_thinking


@dataclass
class Question:
    uid: str
    dataset: str
    split: str
    task: str
    question: str
    samples: list[dict]                      # text, token_ids, pred, closed, finish_reason
    clusters: list[int]                      # answer-equivalence cluster id per sample
    labels: np.ndarray                       # 1 if sample k is correct (evaluation only)
    replay: list[dict] = field(default_factory=list)
    followup: list[dict] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)   # outcome reward model score per sample
    energy: list[float] = field(default_factory=list)   # mean raw logit of the answer tokens per sample

    @property
    def k(self) -> int:
        return len(self.samples)

    def subset(self, idx: list[int]) -> Question:
        """The same question restricted to samples ``idx`` (an independent K' < K replicate)."""
        pick = lambda xs: [xs[i] for i in idx] if xs else xs  # noqa: E731
        return replace(self, samples=pick(self.samples), clusters=pick(self.clusters), labels=self.labels[idx],
                       replay=pick(self.replay), followup=pick(self.followup), reward=pick(self.reward),
                       energy=pick(self.energy))

    def responses(self) -> list[str]:
        """Final responses after ``</think>`` (empty when the trace was cut off)."""
        return [s["response"] if "response" in s else split_thinking(s["text"])[1] for s in self.samples]


def _clusters(model: str, dataset: str, rows: list[dict]) -> dict[str, list[int]]:
    """Answer-equivalence clusters, cached because math_verify comparisons are slow."""
    path = run_dir(model) / "clusters" / f"{dataset}.jsonl"
    if path.exists():
        return {r["uid"]: r["clusters"] for r in read_jsonl(path)}
    out = {r["uid"]: cluster_answers([s["pred"] for s in r["samples"]], r["task"], r["uid"]) for r in rows}
    write_jsonl(path, ({"uid": u, "clusters": c} for u, c in out.items()))
    return out


def load_generations(model: str, dataset: str, partial: bool = False) -> list[dict]:
    """Raw generation records of a test set: the rows of each of its generated datasets
    (``benchmark.yaml: test_union``). With ``partial``, generated datasets this run lacks are skipped (robustness
    runs sampled only some of them)."""
    paths = [run_dir(model) / "gen" / f"{p}.jsonl" for p in set_parts(dataset)]
    return [r for path in paths if path.exists() or not partial for r in read_jsonl(path)]


def load_questions(model: str, dataset: str, with_replay: bool = True, with_followup: bool = True,
                   partial: bool = False) -> list[Question]:
    """All questions of a test set (the union of its generated datasets, see :func:`load_generations`), with their
    responses and signals; ``Question.dataset`` keeps the generated dataset."""
    parts = [p for p in set_parts(dataset) if not partial or (run_dir(model) / "gen" / f"{p}.jsonl").exists()]
    return [q for p in parts for q in load_generated(model, p, with_replay, with_followup)]


def load_generated(model: str, dataset: str, with_replay: bool = True, with_followup: bool = True) -> list[Question]:
    """The questions of one generated dataset. Question ids that occur more than once (20 ids in the TriviaQA
    validation sample, which repeats questions with different evidence) are dropped entirely, because their
    per-sample signals cannot be told apart."""
    rows = list(read_jsonl(run_dir(model) / "gen" / f"{dataset}.jsonl"))
    counts = Counter(r["uid"] for r in rows)
    rows = [r for r in rows if counts[r["uid"]] == 1]
    clusters = _clusters(model, dataset, rows)
    rep = load_replay(model, dataset) if with_replay else {}
    fol = load_followup(model, dataset) if with_followup else {}
    rew = load_reward(model, dataset) if with_followup else {}
    ene = load_energy(model, dataset) if with_followup else {}
    out = []
    for r in rows:
        k = len(r["samples"])
        out.append(Question(
            uid=r["uid"], dataset=r["dataset"], split=r["split"], task=r["task"], question=r["question"],
            samples=r["samples"], clusters=clusters[r["uid"]],
            labels=np.array([s["correct"] for s in r["samples"]], dtype=float),
            replay=[rep.get((r["uid"], i)) for i in range(k)] if rep else [],
            followup=fol.get(r["uid"], []) if fol else [],
            reward=rew.get(r["uid"], []),
            energy=ene.get(r["uid"], []),
        ))
    return out


def merge_runs(model: str, extra: str, dataset: str, **kw) -> list[Question]:
    """Questions of ``model`` with the responses of an independent run ``extra`` appended (same
    questions, independent seeds; only the questions ``extra`` sampled); answer clusters are recomputed over all
    responses."""
    base = {q.uid: q for q in load_questions(model, dataset, **kw)}
    both = lambda a, b: a + b if a and b else []  # noqa: E731  (a signal one run lacks, e.g. energy, is dropped)
    out = []
    for q2 in load_questions(extra, dataset, partial=True, **kw):
        q1 = base.get(q2.uid)
        if q1 is None:
            continue
        samples = q1.samples + q2.samples
        out.append(replace(q1, samples=samples, labels=np.concatenate([q1.labels, q2.labels]),
                           clusters=cluster_answers([s["pred"] for s in samples], q1.task, q1.uid),
                           replay=both(q1.replay, q2.replay), followup=both(q1.followup, q2.followup),
                           reward=both(q1.reward, q2.reward), energy=both(q1.energy, q2.energy)))
    return out


def load_pool(model: str, **kw) -> list[Question]:
    """The training pool: the questions of every pool in ``uqbench/configs/benchmark.yaml: training_pools``."""
    bench = load_yaml("benchmark")
    return [q for name in bench["training_pools"] for q in load_questions(model, name, **kw)]
