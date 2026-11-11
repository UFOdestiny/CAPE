"""GPU stage: sample K reasoning responses per question with vLLM.

Responses are stored with their exact prompt/output token ids so that later stages (teacher-forced
replay, follow-up probes) condition on the identical token sequence. No log-probs are requested
here: per-token statistics are computed exactly in the replay stage, which keeps generation fast.

Usage::

    python -m uqbench.pipeline.generate --model qwen3-8b --gpus 0,1,2,3 [--datasets math500 aime] [--limit 8]
"""
from __future__ import annotations

import argparse

from uqbench.config import generated, load_yaml, model_config, model_path, run_dir
from uqbench.io import append_jsonl, done_uids, launch_workers, read_jsonl, write_jsonl
from uqbench.pipeline import data
from uqbench.verify import extract_answer, is_correct, split_response


def build_prompt_ids(tokenizer, record: dict, chat_kwargs: dict) -> list[int]:
    messages = [{"role": "user", "content": data.user_prompt(record)}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **chat_kwargs)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def jobs(datasets: list[str], limit: int) -> list[tuple[int, str, dict, int]]:
    """(job index, dataset, record, K) for the requested datasets. The index runs over the full
    canonical ``generation_order`` so seeds and shards do not depend on which datasets are run."""
    bench = load_yaml("benchmark")
    unknown = set(datasets) - set(bench["generation_order"])
    if unknown:
        raise ValueError(f"add {sorted(unknown)} to generation_order in configs/benchmark.yaml")
    out, index = [], 0
    for name in bench["generation_order"]:
        k = bench["samples"]["pool" if name in bench["training_pools"] else "test"]
        rows = data.load(name)
        for r in rows[:limit] if limit else rows:
            if name in datasets:
                out.append((index, name, r, k))
            index += 1
    return out


def shared_prefixes(source: str, name: str, fraction: float) -> dict[str, dict]:
    """Shared-prefix (dependent) sampling: every response of a question continues the first
    response of ``source`` after its first L tokens, L = ``fraction`` x the median length of the
    other responses. L does not depend on the first response, so the cut is a stopping time and
    each continued response is still an exact sample from the model; only the dependence between
    the K responses changes. A first response no longer than L is reused whole."""
    out = {}
    for row in read_jsonl(run_dir(source) / "gen" / f"{name}.jsonl"):
        first = row["samples"][0]
        others = sorted(len(s["token_ids"]) for s in row["samples"][1:])
        cut = int(round(fraction * others[(len(others) - 1) // 2]))
        whole = len(first["token_ids"]) <= cut
        out[row["uid"]] = {"ids": first["token_ids"] if whole else first["token_ids"][:cut],
                           "whole": first if whole else None}
    return out


def worker(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg, bench = model_config(args.model), load_yaml("benchmark")
    path = model_path(cfg["path"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    out_dir = run_dir(args.model) / "gen" / "_shards"
    mine = [job for job in jobs(args.datasets, args.limit) if job[0] % args.num_shards == args.shard]
    llm = LLM(path, dtype="bfloat16", max_model_len=cfg["max_model_len"], gpu_memory_utilization=0.92,
              max_num_seqs=512, seed=bench["seed"], enable_prefix_caching=True)
    # One generate() call over all datasets keeps the GPU busy through long-sequence tails.
    todo = []
    for name in args.datasets:
        finished = done_uids(out_dir / f"{name}.{args.shard}.jsonl")
        todo += [j for j in mine if j[1] == name and j[2]["uid"] not in finished]
    if not todo:
        return
    budget = cfg.get("max_tokens", bench["max_tokens"])
    offset = cfg.get("seed_offset", 0)  # independent extra samples of the same backbone
    skip_special = cfg.get("reasoning_format") != "harmony"
    prefix = {}
    if "prefix" in cfg:
        for name in args.datasets:
            prefix.update(shared_prefixes(cfg["prefix"]["of"], name, cfg["prefix"]["fraction"]))
    rows, prompts, params = [], [], []
    for idx, name, record, k in todo:
        ids = build_prompt_ids(tokenizer, record, cfg["chat_template_kwargs"])
        head = prefix.get(record["uid"], {"ids": [], "whole": None})
        rows.append((name, record, ids, head, k))
        if head["whole"] is None:
            prompts.append(TokensPrompt(prompt_token_ids=ids + head["ids"]))
            params.append(SamplingParams(n=k, max_tokens=budget - len(head["ids"]), seed=bench["seed"] + offset + idx,
                                         skip_special_tokens=skip_special, **cfg["sampling"]))
    outputs = iter(llm.generate(prompts, params))
    done = {}
    for name, record, ids, head, k in rows:
        if head["whole"] is not None:
            s = head["whole"]
            samples = [{"token_ids": s["token_ids"], "text": s["text"], "finish_reason": s["finish_reason"]}] * k
        elif head["ids"]:
            samples = [{"token_ids": head["ids"] + list(o.token_ids), "finish_reason": o.finish_reason,
                        "text": tokenizer.decode(head["ids"] + list(o.token_ids), skip_special_tokens=skip_special)}
                       for o in next(outputs).outputs]
        else:
            samples = [{"token_ids": list(o.token_ids), "text": o.text, "finish_reason": o.finish_reason}
                       for o in next(outputs).outputs]
        if head["ids"]:
            for s in samples:
                s["prefix_len"] = len(head["ids"])
        done.setdefault(name, []).append({"uid": record["uid"], "prompt_ids": ids, "samples": samples})
    for name in args.datasets:
        append_jsonl(out_dir / f"{name}.{args.shard}.jsonl", done.get(name, []))


def merge(model: str, datasets: list[str], limit: int) -> None:
    """Join shards in dataset order and attach answers and correctness labels."""
    gen_dir = run_dir(model) / "gen"
    fmt = model_config(model).get("reasoning_format", "think")
    for name in datasets:
        raw = {}
        for shard in sorted((gen_dir / "_shards").glob(f"{name}.*.jsonl")):
            raw.update({r["uid"]: r for r in read_jsonl(shard)})
        rows = []
        for record in (data.load(name)[:limit] if limit else data.load(name)):
            gen = raw[record["uid"]]
            for s in gen["samples"]:
                _, response, closed = split_response(s["text"], fmt)
                s["closed"] = closed if fmt != "none" else s["finish_reason"] == "stop"
                s["response"] = response
                s["pred"] = extract_answer(response, record["task"]) if closed else None
                s["correct"] = int(is_correct(s["pred"], record["answer"], record["task"]))
            rows.append({**record, "prompt_ids": gen["prompt_ids"], "samples": gen["samples"]})
        write_jsonl(gen_dir / f"{name}.jsonl", rows)
        n = sum(len(r["samples"]) for r in rows)
        acc = sum(s["correct"] for r in rows for s in r["samples"]) / n
        unclosed = sum(not s["closed"] for r in rows for s in r["samples"]) / n
        print(f"[generate] {model}/{name}: {len(rows)} questions, {n} responses, "
              f"acc={acc:.3f}, unclosed={unclosed:.3f}")


def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=generated(bench["test_sets"] + bench["training_pools"]))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--limit", type=int, default=0, help="first N questions per dataset (smoke tests)")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true", help="re-join existing shards")
    args = ap.parse_args()
    if args.shard >= 0:
        return worker(args)
    if not args.merge_only:
        passthrough = ["--model", args.model, "--datasets", *args.datasets, "--limit", str(args.limit)]
        launch_workers("uqbench.pipeline.generate", passthrough, args.gpus.split(","), run_dir(args.model) / "logs")
    merge(args.model, args.datasets, args.limit)


if __name__ == "__main__":
    main()
