"""GPU stage: short follow-up continuations conditioned on each complete response (vLLM).

Two probes share the response prefix (so prefix caching makes them cheap):

* ``ptrue`` (CISC, Taubenfeld et al. 2025): the CISC confidence prompt is appended directly after
  the response and we read P("1") / P("0") of the next token.
* ``verbal`` (LRMVerbal, Yoon et al. 2025, shared-trace proxy): after the complete response a
  second user turn asks for the 10-class verbal confidence of Listing 1; the assistant turn is
  pre-filled with a closed think block and ``**Confidence**:``; greedy decoding is constrained to
  the ten class names (structured outputs) so that every response receives a class.

Usage::

    python -m uqbench.pipeline.followup --model qwen3-8b --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import re

from uqbench.config import generated, load_yaml, model_config, model_path, run_dir
from uqbench.io import launch_workers, read_jsonl, write_jsonl

PTRUE_SUFFIX = ("\n\nNow I will rate my confidence in the proposed answer as either 0 or 1. "
                "Proposed confidence: (")

VERBAL_CLASSES = {
    "almost no chance": 0.05, "highly unlikely": 0.15, "chances are slight": 0.25, "unlikely": 0.35,
    "less than even": 0.45, "better than even": 0.55, "likely": 0.65, "very good chance": 0.75,
    "highly likely": 0.85, "almost certain": 0.95,
}
VERBAL_PROMPT = (
    "Now, thoroughly assess your confidence in your answer above by evaluating your thinking process "
    "so far. Classify your confidence into one of the following classes based on how likely your "
    "answer is to be correct:\n"
    '- "Almost no chance" (0.0–0.1)\n- "Highly unlikely" (0.1–0.2)\n- "Chances are slight" (0.2–0.3)\n'
    '- "Unlikely" (0.3–0.4)\n- "Less than even" (0.4–0.5)\n- "Better than even" (0.5–0.6)\n'
    '- "Likely" (0.6–0.7)\n- "Very good chance" (0.7–0.8)\n- "Highly likely" (0.8–0.9)\n'
    '- "Almost certain" (0.9–1.0)\n'
    "Each category reflects the probability that your answer is correct. Format your confidence as "
    "**Confidence**: $CLASS where CLASS is one of the names (only the names without the probability "
    "ranges) of the classes above."
)
CLOSED_THINK = "<think>\n\n</think>\n\n"
ANSWER_PREFILL = "**Confidence**:"


def verbal_turn(tokenizer, chat_kwargs: dict, reasoning_format: str = "think") -> str:
    """Text appended after a finished response: end of the assistant turn, the confidence question
    as a new user turn, and an assistant turn pre-filled with a closed think block. The turn
    delimiters are read off the model's own chat template."""
    marker_a, marker_q = "\x00ANSWER\x00", "\x00QUESTION\x00"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": marker_a},
         {"role": "user", "content": marker_q}], tokenize=False, add_generation_prompt=True, **chat_kwargs)
    turn = text.split(marker_a, 1)[1].replace(marker_q, VERBAL_PROMPT)
    if reasoning_format == "none":
        return turn + ANSWER_PREFILL
    if reasoning_format == "harmony":
        return turn + "<|channel|>final<|message|>" + ANSWER_PREFILL
    if turn.rstrip().endswith("<think>"):  # template already opens a think block
        return turn + "\n</think>\n\n" + ANSWER_PREFILL
    return turn + CLOSED_THINK + ANSWER_PREFILL
_CLASS_RE = re.compile("|".join(sorted(VERBAL_CLASSES, key=len, reverse=True)), re.IGNORECASE)


def parse_verbal(text: str) -> float | None:
    """Midpoint of the first verbal class mentioned (longest names matched first)."""
    m = _CLASS_RE.search(text)
    return VERBAL_CLASSES[m.group(0).lower()] if m else None


def encode_with_specials(tokenizer, text: str) -> list[int]:
    """Encode text whose control tokens (e.g. ``</s>[INST]``) must map to their special ids;
    plain segments are encoded without adding BOS/EOS."""
    specials = {t for t in getattr(tokenizer, "all_special_tokens", [])}
    specials |= {t for t in ("[INST]", "[/INST]", "<|im_start|>", "<|im_end|>") if t in tokenizer.get_vocab()}
    specials = sorted((t for t in specials if t and t in text), key=len, reverse=True)
    if not specials:
        return tokenizer.encode(text, add_special_tokens=False)
    pattern = re.compile("(" + "|".join(re.escape(t) for t in specials) + ")")
    ids = []
    for part in pattern.split(text):
        if not part:
            continue
        ids += [tokenizer.convert_tokens_to_ids(part)] if part in specials else \
            tokenizer.encode(part, add_special_tokens=False)
    return ids


def worker(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    cfg, bench = model_config(args.model), load_yaml("benchmark")
    path = args.model_path or model_path(cfg["path"])
    tok = AutoTokenizer.from_pretrained(path)
    eos = set(tok.convert_tokens_to_ids([tok.eos_token, "<|im_end|>", "<|endoftext|>"])) - {None}
    ptrue_ids = tok.encode(PTRUE_SUFFIX, add_special_tokens=False)
    turn = verbal_turn(tok, cfg["chat_template_kwargs"], cfg.get("reasoning_format", "think"))
    verbal_ids = encode_with_specials(tok, turn)
    one, zero = tok.convert_tokens_to_ids("1"), tok.convert_tokens_to_ids("0")
    llm = LLM(path, dtype="bfloat16", max_model_len=args.max_model_len or cfg["max_model_len"],
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=256, seed=bench["seed"], enable_prefix_caching=True)
    sp_ptrue = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    choices = [" " + c.title() for c in VERBAL_CLASSES]
    sp_verbal = SamplingParams(max_tokens=16, temperature=0.0,
                               structured_outputs=StructuredOutputsParams(choice=choices))
    limit = llm.llm_engine.model_config.max_model_len
    room = limit - max(len(ptrue_ids) + sp_ptrue.max_tokens, len(verbal_ids) + sp_verbal.max_tokens)
    out_dir = run_dir(args.model) / "followup" / "_shards"
    for name in args.datasets:
        out_path = out_dir / f"{name}.{args.shard}.jsonl"
        if out_path.exists():
            continue
        rows = [r for i, r in enumerate(read_jsonl(run_dir(args.model) / "gen" / f"{name}.jsonl"))
                if i % args.num_shards == args.shard]
        prompts, params, keys, cut = [], [], [], 0
        for r in rows:
            for k, s in enumerate(r["samples"]):
                base = r["prompt_ids"] + s["token_ids"]
                while base and base[-1] in eos:
                    base = base[:-1]
                if len(base) > room:   # a response that fills the budget: drop the middle of its reasoning
                    head = len(r["prompt_ids"]) + (room - len(r["prompt_ids"])) // 2
                    base, cut = base[:head] + base[len(base) - (room - head):], cut + 1
                prompts += [TokensPrompt(prompt_token_ids=base + ptrue_ids),
                            TokensPrompt(prompt_token_ids=base + verbal_ids)]
                params += [sp_ptrue, sp_verbal]
                keys.append((r["uid"], k))
        outputs = llm.generate(prompts, params)
        results: dict[str, list] = {}
        for i, (uid, _) in enumerate(keys):
            lp = outputs[2 * i].outputs[0].logprobs[0]
            text = outputs[2 * i + 1].outputs[0].text
            p1 = float(lp[one].logprob) if one in lp else None
            p0 = float(lp[zero].logprob) if zero in lp else None
            results.setdefault(uid, []).append(
                {"ptrue_logp1": p1, "ptrue_logp0": p0, "verbal_text": text, "verbal_conf": parse_verbal(text)})
        write_jsonl(out_path, ({"uid": uid, "samples": v} for uid, v in results.items()))
        print(f"[followup] {name} shard {args.shard}: {len(keys)} responses, {cut} shortened to fit", flush=True)




def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=generated(bench["test_sets"] + bench["training_pools"]))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--model-path", default="", help="override checkpoint (tests)")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=0, help="override (tests)")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.shard >= 0:
        return worker(args)
    passthrough = ["--model", args.model, "--datasets", *args.datasets, "--gpu-mem", str(args.gpu_mem),
                   "--max-model-len", str(args.max_model_len)]
    if args.model_path:
        passthrough += ["--model-path", args.model_path]
    launch_workers("uqbench.pipeline.followup", passthrough, args.gpus.split(","), run_dir(args.model) / "logs")


if __name__ == "__main__":
    main()
