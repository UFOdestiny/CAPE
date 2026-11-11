"""Read persisted inference signals without importing pipeline execution modules.

The NPZ and JSONL schemas are shared with the writers in uqbench.pipeline.
"""
from __future__ import annotations

import numpy as np

from uqbench.config import run_dir
from uqbench.io import read_jsonl

POSITIONS = ("answer", "think_end", "last")
TOKEN_STATS = ("chosen_lp", "entropy", "self_certainty", "conf_topk")


def load_replay(model: str, dataset: str) -> dict:
    """Merge shards into ``{(uid, k): {"length", "positions", "hidden", <token stats>...}}``."""
    out = {}
    for path in sorted((run_dir(model) / "replay").glob(f"{dataset}.*.npz")):
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}  # read each array once (NpzFile re-reads per access)
        offsets = np.concatenate([[0], np.cumsum(arrays["length"])])
        for i, (uid, k) in enumerate(zip(arrays["uid"], arrays["sample"], strict=True)):
            item = {s: arrays[s][offsets[i] : offsets[i + 1]] for s in TOKEN_STATS}
            item.update(length=int(arrays["length"][i]),
                        positions=dict(zip(POSITIONS, arrays["positions"][i], strict=True)),
                        hidden=arrays["hidden"][i], layers=arrays["layers"])
            out[(str(uid), int(k))] = item
    return out


def load_energy(model: str, dataset: str) -> dict[str, list[float]]:
    """``{uid: [mean raw logit over the answer tokens of each sample]}`` (NaN when a response is empty)."""
    out: dict[str, dict[int, float]] = {}
    for path in sorted((run_dir(model) / "energy").glob(f"{dataset}.*.npz")):
        with np.load(path) as z:
            for uid, k, v in zip(z["uid"], z["sample"], z["answer_logit"], strict=True):
                out.setdefault(str(uid), {})[int(k)] = float(v)
    return {u: [d[k] for k in sorted(d)] for u, d in out.items()}


def load_followup(model: str, dataset: str) -> dict[str, list[dict]]:
    out = {}
    for path in sorted((run_dir(model) / "followup" / "_shards").glob(f"{dataset}.*.jsonl")):
        out.update({r["uid"]: r["samples"] for r in read_jsonl(path)})
    return out


def load_reward(model: str, dataset: str) -> dict[str, list[float]]:
    out = {}
    for path in sorted((run_dir(model) / "reward" / "_shards").glob(f"{dataset}.*.jsonl")):
        out.update({r["uid"]: r["rewards"] for r in read_jsonl(path)})
    return out
