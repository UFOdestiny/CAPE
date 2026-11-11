"""GPU stage: teacher-forced replay of every generated response with HF transformers.

For each response we run one forward pass over ``prompt_ids + output_ids`` (the exact sampled
tokens) and keep:

* per-token statistics of the full-vocabulary next-token distribution (raw logits, T = 1):
  ``chosen_lp`` (log-prob of the sampled token), ``entropy``, ``self_certainty`` = KL(U || p)
  and ``conf_topk`` = -mean of the top-k log-probs (DeepConf token confidence, k = 20);
* hidden states at three positions (last token of the final ``\\boxed{}`` answer, end of
  ``</think>``, last response token) for layers ``stride, 2*stride, ..., L`` and the last layer.

Outputs per dataset and shard: ``replay/<dataset>.<shard>.npz``.

Usage::

    python -m uqbench.pipeline.replay --model qwen3-8b --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from uqbench.artifacts import POSITIONS, TOKEN_STATS
from uqbench.config import generated, load_yaml, model_config, model_path, run_dir
from uqbench.io import launch_workers, read_jsonl
from uqbench.verify import HARMONY_FINAL, THINK_END

LOGIT_CHUNK = 4096


def char_to_token(tokenizer, ids: list[int], char_end: int, skip_special: bool = True) -> int:
    """Index of the token whose decoded prefix first reaches ``char_end`` characters."""
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=skip_special)) >= char_end:
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


def find_positions(tokenizer, ids: list[int], eos_ids: set[int], marker: str = THINK_END,
                   skip_special: bool = True) -> dict[str, int]:
    """Token indices (within the response) of the answer end, reasoning end (``marker``) and last
    token. Harmony outputs use ``marker=HARMONY_FINAL`` with special tokens kept in the text."""
    last = len(ids) - 1
    while last > 0 and ids[last] in eos_ids:
        last -= 1
    text = tokenizer.decode(ids, skip_special_tokens=skip_special)
    think = text.rfind(marker)
    think_end = char_to_token(tokenizer, ids, think + len(marker), skip_special) if think >= 0 else last
    answer = last
    box = text.rfind("\\boxed")
    if think >= 0 and box > think:
        depth, j = 0, text.find("{", box)
        while 0 <= j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                answer = char_to_token(tokenizer, ids, j + 1, skip_special)
                break
            j += 1
    return {"answer": min(answer, last), "think_end": min(think_end, last), "last": last}


def token_stats(logits, targets, vocab: int, topk: int) -> dict[str, np.ndarray]:
    import torch

    logp = torch.log_softmax(logits[:, :vocab].float(), dim=-1)
    if int(targets.max()) >= vocab:  # sampled id in the padded embedding rows (seen with gpt-oss at T=1)
        chosen = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    else:
        chosen = logp.gather(1, targets[:, None]).squeeze(1)
    out = {
        "chosen_lp": chosen,
        "entropy": -(logp.exp() * logp).sum(-1),
        "self_certainty": -logp.mean(-1) - math.log(vocab),
        "conf_topk": -logp.topk(topk, dim=-1).values.mean(-1),
    }
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


def chunked_features(model, ids, n_prompt: int, abs_pos: list[int], layers: list[int], vocab: int,
                     topk: int, chunk: int):
    """Incremental prefill with a KV cache (``chunk`` tokens at a time) for models whose attention
    kernel has no memory-efficient implementation (e.g. gpt-oss attention sinks → eager)."""
    import torch
    from transformers import DynamicCache

    total = ids.shape[1]
    cache = DynamicCache(config=model.config)
    stats = {s: [] for s in TOKEN_STATS}
    picked = {}
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        out = model.model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
        for j, p in enumerate(abs_pos):
            if start <= p < end:
                picked[j] = torch.stack([out.hidden_states[layer][0, p - start] for layer in layers])
        lo, hi = max(start, n_prompt - 1), min(end, total - 1)
        if lo < hi:
            logits = model.lm_head(out.last_hidden_state[0, lo - start : hi - start])
            for s, v in token_stats(logits, ids[0, lo + 1 : hi + 1], vocab, topk).items():
                stats[s].append(v)
        cache = out.past_key_values
        del out
    hidden = torch.stack([picked[j] for j in range(len(abs_pos))])
    return {s: np.concatenate(v) for s, v in stats.items()}, hidden


def worker(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, bench = model_config(args.model), load_yaml("benchmark")
    path = model_path(cfg["path"]) if not args.model_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    rcfg = cfg.get("replay", {})
    attn = rcfg.get("attn_implementation", "flash_attention_2") if torch.cuda.is_available() else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
    chunk = rcfg.get("chunk")
    harmony = cfg.get("reasoning_format") == "harmony"
    marker, skip_special = (HARMONY_FINAL, False) if harmony else (THINK_END, True)
    n_layers = model.config.num_hidden_layers
    stride = bench["replay"]["layer_stride"]
    layers = sorted(set(range(stride, n_layers + 1, stride)) | {n_layers})
    vocab, topk = len(tokenizer), bench["replay"]["topk"]
    eos = set(tokenizer.convert_tokens_to_ids([tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"])) - {None}
    out_dir = run_dir(args.model) / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        out_path = out_dir / f"{name}.{args.shard}.npz"
        if out_path.exists():
            continue
        rows = list(read_jsonl(run_dir(args.model) / "gen" / f"{name}.jsonl"))
        units = [(r, k) for i, r in enumerate(rows) if i % args.num_shards == args.shard
                 for k in range(len(r["samples"]))]
        uids, kidx, lengths, pos_rows, hidden = [], [], [], [], []
        stats = {k: [] for k in TOKEN_STATS}
        for r, k in units:
            out_ids = r["samples"][k]["token_ids"]
            ids = torch.tensor([r["prompt_ids"] + out_ids], device=model.device)
            n_prompt = len(r["prompt_ids"])
            pos = find_positions(tokenizer, out_ids, eos, marker, skip_special)
            if chunk:
                with torch.no_grad():
                    feats, h = chunked_features(model, ids, n_prompt, [n_prompt + pos[p] for p in POSITIONS],
                                                layers, vocab, topk, chunk)
                hidden.append(h.half().cpu().numpy())
                for s in TOKEN_STATS:
                    stats[s].append(feats[s])
                uids.append(r["uid"])
                kidx.append(k)
                lengths.append(len(out_ids))
                pos_rows.append([pos[p] for p in POSITIONS])
                continue
            with torch.no_grad():
                out = model.model(input_ids=ids, output_hidden_states=True)
                hs, last_hidden = out.hidden_states, out.last_hidden_state[0]
                chunks = {s: [] for s in TOKEN_STATS}
                targets = ids[0, n_prompt:]
                for start in range(0, len(out_ids), LOGIT_CHUNK):
                    end = min(start + LOGIT_CHUNK, len(out_ids))
                    logits = model.lm_head(last_hidden[n_prompt - 1 + start : n_prompt - 1 + end])
                    for s, v in token_stats(logits, targets[start:end], vocab, topk).items():
                        chunks[s].append(v)
                idx = torch.tensor([n_prompt + pos[p] for p in POSITIONS], device=model.device)
                hidden.append(torch.stack([hs[layer][0, idx] for layer in layers], dim=1).half().cpu().numpy())
            for s in TOKEN_STATS:
                stats[s].append(np.concatenate(chunks[s]))
            uids.append(r["uid"])
            kidx.append(k)
            lengths.append(len(out_ids))
            pos_rows.append([pos[p] for p in POSITIONS])
            del hs, out
        np.savez(out_path, uid=np.array(uids), sample=np.array(kidx), length=np.array(lengths),
                 positions=np.array(pos_rows), layers=np.array(layers),
                 hidden=np.stack(hidden) if hidden else np.zeros((0, 3, len(layers), 1), np.float16),
                 **{s: np.concatenate(v) if v else np.zeros(0, np.float32) for s, v in stats.items()})
        print(f"[replay] {name} shard {args.shard}: {len(units)} responses", flush=True)




def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default="", help="override checkpoint (tests)")
    ap.add_argument("--datasets", nargs="+", default=generated(bench["test_sets"] + bench["training_pools"]))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.shard >= 0:
        return worker(args)
    passthrough = ["--model", args.model, "--datasets", *args.datasets]
    if args.model_path:
        passthrough += ["--model-path", args.model_path]
    launch_workers("uqbench.pipeline.replay", passthrough, args.gpus.split(","), run_dir(args.model) / "logs")


if __name__ == "__main__":
    main()
