"""Answer-set judge (GPU stage, ablations only): one short prompt per question over its distinct sampled answers.

The backbone sees the question and the distinct final answers of its K responses (one per answer
cluster, in a per-question shuffled order) plus "None of the above is correct", and we read the
next-token log-probabilities of the option numbers after the pre-filled "The correct option is (".
The answers are shown without the reasoning traces, so the probe is short; ``--think-tokens`` lets
the backbone reason first. Used only by the ablation ``python -m uqbench.studies.ablation --studies judge``
(CAPE with ``use_judge=True``, and the judge alone, which resembles clustered self-assessment).

Usage::

    python -m uqbench.pipeline.judge --model qwen3-8b --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import random

from uqbench.config import generated, load_yaml, model_config, model_path, run_dir, set_parts
from uqbench.io import launch_workers, read_jsonl, write_jsonl
from uqbench.pipeline.followup import encode_with_specials
from uqbench.records import load_generated

MAX_OPTIONS = 9                    # single-token option numbers 1..9 (<= 8 clusters + "none")
NO_LOGP = -20.0                    # option number outside the top-20 log-probabilities
PREFILL = "The correct option is ("


def judge_prompt(question: str, answers: list[str], think: bool = False) -> str:
    lines = [f"({i + 1}) {a}" for i, a in enumerate(answers)]
    lines.append(f"({len(answers) + 1}) None of the above is correct.")
    return (f"Question:\n{question}\n\nIndependent solutions to this question gave the following final "
            "answers:\n" + "\n".join(lines) + "\n\nWhich option is correct? "
            + ("Think briefly, then give the option number." if think else "Reply with the option number only."))


def chat_ids(tokenizer, text: str, chat_kwargs: dict) -> list[int]:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True, **chat_kwargs)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def close_reasoning(fmt: str) -> str:
    """Text that ends a (possibly truncated) reasoning segment and opens the final answer."""
    if fmt == "harmony":
        return "<|end|><|start|>assistant<|channel|>final<|message|>" + PREFILL
    if fmt == "think":
        return "\n</think>\n\n" + PREFILL
    return "\n\n" + PREFILL


def prompt_ids(tokenizer, text: str, chat_kwargs: dict, fmt: str) -> list[int]:
    ids = chat_ids(tokenizer, text, chat_kwargs)
    if fmt == "harmony":
        tail = "<|channel|>final<|message|>" + PREFILL
    elif fmt == "think":
        opened = tokenizer.decode(ids[-4:]).rstrip().endswith("<think>")
        tail = ("\n</think>\n\n" if opened else "<think>\n\n</think>\n\n") + PREFILL
    else:
        tail = PREFILL
    return ids + encode_with_specials(tokenizer, tail)


def candidates(q) -> tuple[list[int], list[str]]:
    """Distinct answer clusters (one representative answer each) in a per-question shuffled order."""
    reps = {}
    for c, s in zip(q.clusters, q.samples, strict=True):
        if s["pred"] is not None:
            reps.setdefault(c, str(s["pred"]))
    order = sorted(reps, key=lambda c: (-q.clusters.count(c), c))[: MAX_OPTIONS - 1]
    random.Random(q.uid).shuffle(order)
    return order, [reps[c] for c in order]


def worker(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg, bench = model_config(args.model), load_yaml("benchmark")
    path = model_path(cfg["path"])
    tok = AutoTokenizer.from_pretrained(path)
    fmt = cfg.get("reasoning_format", "think")
    digits = [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, MAX_OPTIONS + 1)]
    llm = LLM(path, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.9, max_num_seqs=256,
              seed=bench["seed"], enable_prefix_caching=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    think = args.think_tokens > 0
    if think:
        stop_ids = [i for i in (tok.convert_tokens_to_ids(t) for t in ("<|end|>", "<|return|>"))
                    if isinstance(i, int) and fmt == "harmony"]
        sp_think = SamplingParams(max_tokens=args.think_tokens, temperature=0.0, skip_special_tokens=False,
                                  stop=["</think>"] if fmt == "think" else None, stop_token_ids=stop_ids or None)
    out_dir = run_dir(args.model) / ("judge_think" if think else "judge") / "_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        out_path = out_dir / f"{name}.{args.shard}.jsonl"
        if out_path.exists():
            continue
        qs = [q for i, q in enumerate(load_generated(args.model, name, with_replay=False, with_followup=False))
              if i % args.num_shards == args.shard]
        items = [(q, *candidates(q)) for q in qs]
        items = [it for it in items if it[1]]
        if think:
            heads = [chat_ids(tok, judge_prompt(q.question, ans, think=True), cfg["chat_template_kwargs"])
                     for q, _, ans in items]
            drafts = llm.generate([TokensPrompt(prompt_token_ids=h) for h in heads], sp_think)
            tail = encode_with_specials(tok, close_reasoning(fmt))
            prompts = [TokensPrompt(prompt_token_ids=h + list(d.outputs[0].token_ids) + tail)
                       for h, d in zip(heads, drafts, strict=True)]
        else:
            prompts = [TokensPrompt(prompt_token_ids=prompt_ids(tok, judge_prompt(q.question, ans),
                                                                cfg["chat_template_kwargs"], fmt))
                       for q, _, ans in items]
        outputs = llm.generate(prompts, sp)
        rows = []
        for (q, order, _), out in zip(items, outputs, strict=True):
            lp = out.outputs[0].logprobs[0]
            logps = [float(lp[d].logprob) if d in lp else NO_LOGP for d in digits[: len(order) + 1]]
            rows.append({"uid": q.uid, "clusters": order, "logp": logps[:-1], "none_logp": logps[-1]})
        write_jsonl(out_path, rows)
        print(f"[judge] {name} shard {args.shard}: {len(rows)} questions", flush=True)


def load(model: str, dataset: str, think: bool = False) -> dict[str, dict]:
    """Judge outputs of a test set's generated datasets (or of a pool), by question id; generated datasets that
    were not judged contribute nothing."""
    out = {}
    shards = run_dir(model) / ("judge_think" if think else "judge") / "_shards"
    for part in set_parts(dataset):
        for path in sorted(shards.glob(f"{part}.*.jsonl")):
            out.update({r["uid"]: r for r in read_jsonl(path)})
    return out


def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=generated(bench["test_sets"] + [bench["pool"]]))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--think-tokens", type=int, default=0, help="reasoning budget before answering (0: none)")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.shard >= 0:
        return worker(args)
    launch_workers("uqbench.pipeline.judge", ["--model", args.model, "--datasets", *args.datasets,
                                     "--think-tokens", str(args.think_tokens)],
                   args.gpus.split(","), run_dir(args.model) / "logs")


if __name__ == "__main__":
    main()
