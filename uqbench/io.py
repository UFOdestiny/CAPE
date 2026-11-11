"""Small JSONL helpers and multi-GPU worker launching shared by the pipeline stages."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write atomically: rows go to ``<path>.tmp`` and are renamed on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_uids(path: Path) -> set[str]:
    return {r["uid"] for r in read_jsonl(path)} if path.exists() else set()


def launch_workers(module: str, args: list[str], gpus: list[str], log_dir: Path) -> None:
    """Run ``python -m <module> <args> --shard i --num-shards n`` once per GPU and wait."""
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, gpu in enumerate(gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        cmd = [sys.executable, "-m", module, *args, "--shard", str(i), "--num-shards", str(len(gpus))]
        log = open(log_dir / f"{module.split('.')[-1]}.shard{i}.log", "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for i, (proc, log) in enumerate(procs):
        if proc.wait() != 0:
            failed.append(i)
        log.close()
    if failed:
        raise RuntimeError(f"{module}: shards {failed} failed; see logs in {log_dir}")
