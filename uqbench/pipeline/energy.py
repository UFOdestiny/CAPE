"""Teacher-forced pass that keeps the raw logit of every sampled token, for the semantic-energy baseline.

Semantic energy (Ma et al., 2025) takes the energy of a token as its negative raw logit (before the softmax), the
energy of a response as the mean over the tokens of its answer (the reasoning is discarded, as in their think
mode), and scores a response by the summed energy of its answer cluster. ``uqbench.pipeline.replay`` keeps only
normalized
statistics, so this stage runs one forward pass per response and computes the raw logit of each sampled token as
the dot product of the final hidden state with that token's output embedding (no full-vocabulary projection).

Outputs per dataset and shard: ``energy/<dataset>.<shard>.npz`` with, per response, the mean raw logit over the
answer tokens (after ``</think>``, or after the final-channel marker for Harmony outputs) and over all tokens.

Usage::

    python -m uqbench.pipeline.energy --model qwen3-8b --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse

import numpy as np

from uqbench.config import generated, load_yaml, model_config, model_path, run_dir
from uqbench.io import launch_workers, read_jsonl
from uqbench.pipeline.replay import find_positions
from uqbench.verify import HARMONY_FINAL, THINK_END


def chosen_logits(model, ids, n_prompt: int, chunk: int | None):
    """Raw logit of every sampled response token (positions n_prompt .. end), from the final hidden states."""
    import torch

    weight = model.lm_head.weight
    targets = ids[0, n_prompt:]
    if not chunk:
        last = model.model(input_ids=ids).last_hidden_state[0, n_prompt - 1:-1]
        return (last.float() * weight[targets].float()).sum(-1)
    from transformers import DynamicCache

    cache, parts, total = DynamicCache(config=model.config), [], ids.shape[1]
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        out = model.model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True)
        lo, hi = max(start, n_prompt - 1), min(end, total - 1)
        if lo < hi:
            h = out.last_hidden_state[0, lo - start:hi - start]
            parts.append((h.float() * weight[ids[0, lo + 1:hi + 1]].float()).sum(-1))
        cache = out.past_key_values
        del out
    return torch.cat(parts)


def worker(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = model_config(args.model)
    path = model_path(cfg["path"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    rcfg = cfg.get("replay", {})
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=rcfg.get("attn_implementation", "flash_attention_2"),
        device_map="cuda").eval()
    harmony = cfg.get("reasoning_format") == "harmony"
    marker, skip_special = (HARMONY_FINAL, False) if harmony else (THINK_END, True)
    eos = set(tokenizer.convert_tokens_to_ids([tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"])) - {None}
    out_dir = run_dir(args.model) / "energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        out_path = out_dir / f"{name}.{args.shard}.npz"
        src = run_dir(args.model) / "gen" / f"{name}.jsonl"
        if out_path.exists() or not src.exists():
            continue
        rows = list(read_jsonl(src))
        units = [(r, k) for i, r in enumerate(rows) if i % args.num_shards == args.shard
                 for k in range(len(r["samples"]))]
        uids, kidx, answer, full, n_answer = [], [], [], [], []
        for r, k in units:
            out_ids = r["samples"][k]["token_ids"]
            if not out_ids:
                z = np.zeros(0, np.float32)
            else:
                ids = torch.tensor([r["prompt_ids"] + out_ids], device=model.device)
                with torch.no_grad():
                    z = chosen_logits(model, ids, len(r["prompt_ids"]), rcfg.get("chunk")).cpu().numpy()
            pos = find_positions(tokenizer, out_ids, eos, marker, skip_special) if out_ids else None
            # answer tokens: after the reasoning marker up to the last non-EOS token; whole response if no marker
            ans = z[pos["think_end"] + 1:pos["last"] + 1] if pos and pos["think_end"] < pos["last"] else z
            uids.append(r["uid"])
            kidx.append(k)
            answer.append(float(ans.mean()) if len(ans) else np.nan)
            full.append(float(z.mean()) if len(z) else np.nan)
            n_answer.append(len(ans))
        np.savez(out_path, uid=np.array(uids), sample=np.array(kidx), answer_logit=np.array(answer),
                 mean_logit=np.array(full), answer_tokens=np.array(n_answer))
        print(f"[energy] {name} shard {args.shard}: {len(units)} responses", flush=True)




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
    launch_workers("uqbench.pipeline.energy", ["--model", args.model, "--datasets", *args.datasets],
                   args.gpus.split(","), run_dir(args.model) / "logs")


if __name__ == "__main__":
    main()
