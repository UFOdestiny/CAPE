"""Outcome reward model (GPU stage): one scalar reward per (question, response).

A general open reward model (Skywork-Reward-V2-Llama-3.1-8B, 2025) reads the user prompt the backbone
saw and the backbone's visible answer (the text after its reasoning; the end of the trace when the
reasoning was cut off), truncated from the left to ``MAX_TOKENS`` so that the final answer is kept.
The reward is the ORM baseline (``baselines/orm.py``), calibrated and projected like every other estimator.

Usage::

    python -m uqbench.pipeline.reward --model qwen3-8b --gpus 0,1,2,3
    python -m uqbench.pipeline.reward --model qwen3-8b --gpus 0,1,2,3 --datasets math_heldout gpqa_heldout
"""
from __future__ import annotations

import argparse

from uqbench.config import MODELS_DIR, generated, load_yaml, run_dir
from uqbench.io import launch_workers, read_jsonl, write_jsonl
from uqbench.pipeline.data import user_prompt
from uqbench.verify import split_thinking

REWARD_MODEL = "Skywork-Reward-V2-Llama-3.1-8B"
MAX_TOKENS = 4096           # response tokens kept (the end of the visible answer)
BATCH_TOKENS = 64_000       # padded tokens per forward pass


def visible_answer(sample: dict) -> str:
    text = sample.get("response") or split_thinking(sample["text"])[1]
    return text if text.strip() else sample["text"]


def worker(args) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = str(MODELS_DIR / REWARD_MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", num_labels=1).cuda().eval()
    out_dir = run_dir(args.model) / "reward" / "_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        out_path = out_dir / f"{name}.{args.shard}.jsonl"
        if out_path.exists():
            continue
        rows = [r for i, r in enumerate(read_jsonl(run_dir(args.model) / "gen" / f"{name}.jsonl"))
                if i % args.num_shards == args.shard]
        items = []   # (row index, sample index, token ids)
        for i, r in enumerate(rows):
            for k, s in enumerate(r["samples"]):
                answer = visible_answer(s)
                ids = tok.encode(answer, add_special_tokens=False)
                if len(ids) > MAX_TOKENS:   # keep the end, where the final answer is
                    answer = tok.decode(ids[-MAX_TOKENS:], clean_up_tokenization_spaces=False)
                conv = [{"role": "user", "content": user_prompt(r)}, {"role": "assistant", "content": answer}]
                text = tok.apply_chat_template(conv, tokenize=False)
                items.append((i, k, tok(text, add_special_tokens=False)["input_ids"]))
        rewards = [[None] * len(r["samples"]) for r in rows]
        order = sorted(range(len(items)), key=lambda j: len(items[j][2]))
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and len(items[order[end]][2]) * (end - start + 1) <= BATCH_TOKENS:
                end += 1
            batch = [items[j][2] for j in order[start:end]]
            width = max(map(len, batch))
            ids = torch.full((len(batch), width), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for b, seq in enumerate(batch):   # right padding; the head reads the last real token
                ids[b, :len(seq)], mask[b, :len(seq)] = torch.tensor(seq), 1
            with torch.no_grad():
                score = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits[:, 0].float().cpu()
            for j, v in zip(order[start:end], score.tolist(), strict=True):
                i, k, _ = items[j]
                rewards[i][k] = v
            start = end
        write_jsonl(out_path, ({"uid": r["uid"], "rewards": rw} for r, rw in zip(rows, rewards, strict=True)))
        print(f"[reward] {name} shard {args.shard}: {len(items)} responses", flush=True)




def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=generated(bench["test_sets"] + bench["training_pools"]))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.shard >= 0:
        return worker(args)
    launch_workers("uqbench.pipeline.reward", ["--model", args.model, "--datasets", *args.datasets],
                   args.gpus.split(","), run_dir(args.model) / "logs")


if __name__ == "__main__":
    main()
