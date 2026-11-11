# DeepConf: Deep Think with Confidence

> These are reading notes on the original paper and the official implementation; for the adaptation and settings currently used in this repository, see [Experimental protocol](../protocol.md).
- Paper: ICLR 2026 (poster; OpenReview id `8LqHs0KIM7`; camera-ready: https://proceedings.iclr.cc/paper_files/paper/2026/hash/98257285340854262185500e59bc0f28-Abstract-Conference.html ), also a NeurIPS 2025 Workshop on Efficient Reasoning poster; arXiv 2508.15260 (v1, 2025-08-21). Authors: Yichao Fu, Xuewei Wang, Hao Zhang, Yuandong Tian, Jiawei Zhao (Meta AI / UCSD; Fu and Zhao contributed equally). Note: arXiv v1, the citation in the official README, and this project's `latex/reference.bib` all omit Hao Zhang and list the venue as arXiv; they need to be changed to ICLR 2026. These notes follow the ICLR camera-ready (`docs/paper/DeepConf.pdf` / `.md`).
- Official code: https://github.com/facebookresearch/deepconf (MIT; Python, a `DeepThinkLLM` wrapper on top of vLLM (`requirements.txt` pins `vllm==0.10.2`); online early stopping uses vLLM v1's per-request custom logits processor (`AdapterLogitsProcessor`), so no vLLM source changes are needed; grading depends on Dynasor's `math_equal`; also installable via `pip install deepconf`; last push 2026-07-17). Usable: the implementation is complete, including `deepconf/{wrapper,processors,utils,outputs}.py` and `examples/`. However, the repository has no script reproducing the paper's 64-resampling-run tables, and does not implement consensus-based adaptive stopping (see the "Differences between paper and code" in §1). Project page: jiaweizzhao.github.io/deepconf
- Category: single-trace logit (trace-level top-k confidence) + multi-sample aggregation (confidence-weighted voting, filtering, online early stopping)
- Access requirements: gray-box; only needs the top-k logprobs at each generation position (k=20), no full vocabulary; no training; no extra model. Online mode needs to intervene on logits during decoding (vLLM logits processor).

## 1. Method design

**Core idea**: first obtain token-level confidence from top-k statistics of the model's own next-token distribution, then aggregate locally within a trace (sliding-window groups, worst 10% of groups, worst group, tail) to obtain a trace-level quality score. Offline, the score is used for weighted majority voting and top-η% filtering; online, a sliding-window confidence falling below a threshold terminates low-quality traces early. The paper's main goals are the voting accuracy of parallel thinking and token cost, not calibration or UQ evaluation.

**Formulas** (Sec. 2-3.1, Eq. 1-7):
- Token entropy (Eq. 1, background only, not used by DeepConf itself): H_i = −Σ_j P_i(j) log P_i(j).
- **Token confidence** (Eq. 2): C_i = −(1/k) Σ_{j=1..k} log P_i(j), where P_i(j) is the probability of the top-k token at position i. The paper states "We compute confidence using the top-20 token probabilities", i.e. **k=20**, and the top-k used for computing confidence may differ from the top-k used for sampling (Sec. 2). The sharper the distribution, the smaller the probabilities of lower-ranked tokens, and the larger C_i. A larger C_i means higher confidence.
- **Average trace confidence** (Eq. 3): C_avg = (1/N) Σ_{i=1..N} C_i, where N is the total number of generated tokens. The paper says it is "also termed self-certainty", but this is only a top-k proxy for the self-certainty of Kang et al. 2025; see the pitfalls in §6.
- **Group confidence** (Eq. 4): C_{G_i} = (1/|G_i|) Σ_{t∈G_i} C_t, where G_i is an overlapping sliding window consisting of the current token and the preceding tokens, n tokens in total (the text gives n=1024 or 2048 as examples; the experiments default to n=2048).
- **Bottom-10% group confidence** (Eq. 5): C_bottom-10(t) = the mean of the lowest 10% of group confidences in the trace.
- **Lowest group confidence** (Eq. 6): C_least(t) = min_{G_j} C_{G_j}, the extreme case of bottom-10%. This is the quantity used for online early stopping.
- **Tail confidence** (Eq. 7): C_tail(t) = the mean of C_i over the last T_tail tokens, with T_tail fixed at 2048. Appendix C.4 additionally has Tail(10%) (the last 10% of tokens) and Head(10%).
- Appendix C.4 also has B(q%) (bottom-q% groups, q∈{10,50}) and L(w) (lowest group, w∈{512,1K,2K}).

**Offline thinking** (Sec. 3.2, Alg. 1):
- Majority voting: V(a)=Σ_t 1[answer(t)=a].
- Confidence-weighted voting: V(a)=Σ_t C_t·1[answer(t)=a], take the argmax.
- Confidence filtering: before weighted voting, keep only the traces with top-η% confidence within each question's sampled set, η∈{10%, 90%}. For example, with K=512 and η=10%, about 51 traces are kept (Sec. 4.2).

**Online thinking** (Sec. 3.3, Alg. 2):
- Offline warmup: for each question, first fully generate N_init=16 traces, compute their lowest group confidence, and set the threshold s = Percentile_{100−η}. DeepConf-low uses η=10% (90th percentile), DeepConf-high uses η=90% (10th percentile). The implementation description in Sec. 4.1 is s = min_{t∈T_top} C_t.
- During generation, as soon as a trace's current group confidence (window 2048) falls below s, the trace is terminated; terminated traces do not participate in voting. After a trace completes, it is filtered once more with the same s.
- Adaptive sampling: β = V(â)/Σ_a V(a). When β ≥ τ (τ=0.95), stop sampling for that question; otherwise sample until the budget B is reached. B∈{32,64,128,256,512} (Table 12).

**Hyperparameter summary**: k=20, window n=2048, tail of 2048 tokens, bottom fraction 10%, η∈{10%,90%}, N_init=16, τ=0.95 (Table 12). The paper claims "no hyperparameter tuning is needed", but Table 5 shows that the best η varies with model and dataset.

**Scored object and value range**: the scored object is the whole generated trace (all generated tokens of thinking + final answer); an answer-level weighted vote share β can also be derived from it. C_i is not a probability; its lower bound is ln k (about 3.0 for k=20, attained when the top-k probabilities are equal), it has no upper bound, and trace-level values in the paper's Fig. 2 range roughly over 10-22. Computing ECE/Brier requires an additional calibration mapping, which the paper does not do.

**Differences between paper and official code** (based on GitHub HEAD, checked 2026-10-07):
1. Token confidence computation (`utils.compute_confidence`): takes `-mean` over all entries of the `logprobs` dictionary returned by vLLM, rounded to 3 decimals. With vLLM `logprobs=20`, if the sampled token is not in the top-20, the returned dictionary contains this extra token, so the effective k may be 21. The online logits processor (`processors.ConfPerReqLogitsProcessor`) instead applies softmax over the full vocabulary and strictly takes the top-20.
2. Sliding windows use only full windows (stride 1). When a trace is shorter than the window, the mean over all tokens is used instead (offline). The online processor can only trigger early stopping after the window is full. Bottom-10% takes the mean of `max(1, int(0.1·#windows))` windows.
3. Online threshold: the code uses `np.percentile(min_confs, 100 − confidence_percentile)`, with linear interpolation, while the paper says to take the min. The parameter `confidence_percentile` is in fact η. Its default is 90 in `DeepThinkLLM.deepthink` (corresponding to DeepConf-high) and 10 in `examples/example_online.py` (corresponding to DeepConf-low), which is easy to confuse.
4. The code **does not implement consensus-based adaptive stopping with τ=0.95**: the wrapper generates `total_budget − warmup` traces in one batch and only does per-trace early stopping, which is equivalent to the paper's "budget-only" variant.
5. In online mode with `compute_multiple_voting=True` (default), `final_answer` is overwritten by the **unweighted** majority vote over the filtered traces (`voting_results['majority']`), whereas the paper's default is weighted voting. Also, `compute_all_voting_results` only provides mean/tail/bottom/min weighting, plus top-10% filtering for tail and bottom-window, without the 90% variants of the paper's Table 1, and without Mean@10 or L(2K)@10.
6. The seed is `time.time_ns()`, not fixed. Online early stopping is enabled only on CUDA devices; on non-CUDA devices the processor returns None, silently degrading to no early stopping.
7. The example scripts provide no prompt template for Qwen3 and do not append the "\boxed{}" instruction (the paper's App. G says it is appended).
8. The paper's own notation is inconsistent: App. E writes the online stopping threshold as the "consensus threshold τ" ("set τ to the 90th percentile"), contradicting the s/τ definitions in Sec. 3.3.

## 2. Dataset setup
- **AIME24** (AoPS 2024 AIME I+II), **AIME25** (AoPS 2025 AIME I+II; the code example uses HF `MathArena/aime_2025`), **BRUMO25** (Brown Univ. Math Olympiad 2025), **HMMT25** (HMMT Feb 2025; Fig. 2 states 30 questions, the App. figures label it HMMT_FEB25), **GPQA-Diamond** (graduate-level STEM multiple choice). Sources are in Sec. 4.1; all are listed on MathArena.
- Number of questions: only HMMT25 explicitly states 30 questions (Fig. 2); for the other datasets it is **not reported** (AIME I+II implies 30 questions). No subsampling; full sets are used.
- Split: not reported (all are single test sets).
- Answer format: the answer must be placed in `\boxed{...}` and extracted in post-processing (App. G). The code's `extract_answer` takes the bracket-balanced content of the last `boxed`. GPQA uses option letters, and the code compares single letters case-insensitively.
- Correctness labeling: the paper **does not report** the grading function. The code uses Dynasor's `math_equal(answer, ground_truth)`, after first stripping `\text{}` (`quick_parse`).
- GPQA-D results are given only for DeepSeek-8B, Qwen3-8B, and Qwen3-32B (Table 10 / App. F). GPT-OSS-120B has no GPQA-D in Table 1.

## 3. Experimental protocol
- **Sampling pool**: for each question, **4,096 complete traces** are pre-generated as a common sampling framework. Offline, each run resamples a working set of size K (e.g. K=512) from the pool. Online, resampling from the pool likewise drives generation with early stopping (Sec. 4.1). All metrics are averages over **64 independent resampling runs**.
- **Decoding** (Table 11):
  - DeepSeek-8B: T=0.6, top-p=0.95, no top-k, max length 64k
  - Qwen3-8B / Qwen3-32B: T=0.6, top-p=0.95, top-k=20, max length 32k
  - GPT-OSS-20B / 120B: T=1.0, top-p=1.0, top-k=40, max length 130k
  - Generation stops only at EOS or at the maximum length.
- **Prompt** (App. G): for Qwen3 and GPT-OSS, "Please reason step by step, and put your final answer within \boxed{}." is appended after the question. GPT-OSS additionally keeps the official system prompt and sets reasoning effort = high (no tools). DeepSeek-8B uses the official system prompt with the question in the user message (the code uses the Chinese DeepSeek-R1 system prompt).
- **Metrics**: Pass@1 (single-trace accuracy), Cons@K (unweighted majority vote), Measure@K (confidence-weighted voting), Measure+top-η%@K (cutoff recomputed within the working set for each run), and total number of generated tokens (early-stopped traces count only the tokens generated). The paper **does not report trace-level discrimination or calibration metrics such as AUROC, ECE, or Brier**; the only UQ-related evidence is the histogram of confidence for correct/incorrect traces in Fig. 2 (HMMT25, 30 questions × 4096 traces, model not stated), with no numeric values.
- **Online**: B∈{32,64,128,256,512}, N_init=16, η∈{10,90}, τ=0.95, lowest group confidence window 2048 (Sec. 4.3, Table 12).
- **Ablations**: τ∈{1.0,0.95,0.90,0.85} (Table 3, Qwen3-32B/AIME24); N_init∈{8,16,32} (Table 4); η∈{90,75,50,25,10} (Table 5); comparison of confidence metrics (Tables 6-9).
- **Seeds**: not reported (the paper uses 64 resampling runs instead; the code seed is a timestamp).
- **Compute and hardware**: only token counts are reported; GPU model, time, etc. are **not reported**. The README gives one example: DeepSeek-8B on BRUMO25, a single A100, 8 runs per question, budget of 256 traces, online about 2.7M tokens / 5.3k s, baseline about 5.8M / 13.6k s. An example of token scale: Qwen3-8B AIME25 Cons@512 is 2.77×10^8 tokens, which for 30 questions is about 9.2M tokens/question and about 18k tokens/trace (derived values).

## 4. LLM backbone setup
- Five open-source reasoning models: DeepSeek-8B (i.e. `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, footnote 1), Qwen3-8B, Qwen3-32B (thinking reasoning models), GPT-OSS-20B, GPT-OSS-120B (reasoning effort high, no tools).
- The inference framework is vLLM (the paper's Reproducibility section mentions "detailed vLLM integration"; the code pins `vllm==0.10.2` and enables prefix caching).
- Precision: **not reported**.
- Context length: see Table 11 (32k / 64k / 130k).

## 5. Main results and conclusions (brief, key numbers)
- **Offline (Table 1, K=512, top-10% filtering)**:
  - DeepSeek-8B AIME25: Cons 82.3 → Bottom-10%@10% 87.5 / Tail@10% 87.4; HMMT25: 69.6 → Tail@10% 83.9.
  - Qwen3-32B AIME24: 85.3 → 90.8.
  - GPT-OSS-120B AIME25: 97.0 → Tail@10% **99.9** (Pass@1 is 91.8).
  - η=90% is essentially on par with majority voting.
- **Lowest group conf (App. D, 20 settings)**: Top-10% improves in 12/20 settings (+0.26 to +9.38) and degrades in 8/20 settings (−4.69 to −0.31), with an average of +1.22; Top-90% ranges from −0.21 to +0.73, with an average of +0.17. The degradations come from "confidently wrong".
- **Metric comparison (App. C.4, average over 23 model-dataset pairs)**:
  - Maj 83.0; Mean@10 83.9; L(2K)@10 84.4; B(10%)@10 84.0; Tail(10%)@10 84.6; Tail(2k)@10 84.5.
  - Head(10%) has almost no signal (Head(10%)@10 is 81.9, below Maj).
  - Weighting alone without filtering is almost equal to majority voting (e.g. Mean 83.0, L(2K) 83.0).
- **Qwen3-8B results, most relevant to this project**:
  - AIME24: Maj 80.1 → Mean@10 / L(2K)@10 both 86.7.
  - **AIME25**: Maj 82.6 → Mean@10 74.0, Tail(2k)@10 75.7, L(2K)@10 77.9, **aggressive filtering drops accuracy by 4.7-8.6 pp** (Tables 6-8).
  - Online DeepConf-low on AIME25 gives 78.1 vs Maj 82.5 (−4.4 pp, Table 10).
- **Online (Table 2/10, B=512)**:
  - DeepConf-low saves 43.8-84.7% of tokens (Table 10), e.g. GPT-OSS-120B AIME25 saves −84.7% with accuracy 97.9 vs 97.1; DeepSeek-8B AIME24 saves −77.9% with accuracy 92.5 vs 86.7.
  - DeepConf-high saves 16-59% with essentially unchanged accuracy.
  - τ=0.95 on Qwen3-32B AIME24 has no accuracy loss while saving an additional 52.8-54.7% (B=512, Table 3).
- **Conclusion**: local confidence (tail, bottom, lowest) is no worse than the global mean, and slightly better. Filtering matters more than weighting itself. Aggressive filtering gives the largest gain but carries the "confidently wrong" risk; online early stopping can approximate offline lowest-group filtering.

## 6. Fit to this project's unified benchmark
- **High applicability, good fidelity**: DeepConf natively depends only on top-20 logprobs, and vLLM's `logprobs=20` is exactly its default `max_logprobs` limit, so it can be reproduced faithfully on Qwen3-8B thinking mode. The paper itself evaluates Qwen3-8B, and its decoding setup also matches the official Qwen3 thinking recommendation: T=0.6, top-p=0.95, top-k=20, 32k. No full vocabulary, extra model, or training is needed.
- **Use as a response-level correctness score**:
  - For each trace, directly take C_avg, C_bottom-10, C_least (window 2048), C_tail (2048) as the score; higher means more likely correct. The label is whether the extracted boxed answer equals the gold answer.
  - The paper does not report AUROC, so the discriminative power of these scores has to be measured by ourselves.
  - We recommend reporting both: (a) pooled AUROC across questions; (b) within-question discrimination, e.g. the mean of per-question AUROC, or accuracy after top-η% selection. The reason is that all DeepConf thresholds are per-question relative percentiles, and the original design only serves within-question ranking; cross-question comparability has not been validated (question difficulty and trace length both systematically shift confidence).
  - With multiple samples, the weighted vote share β = V(â)/ΣV can also serve as an answer-level confidence, as a baseline for weighted self-consistency.
  - Online mode is not a UQ scorer. Early-stopped traces have no answer, so it only suits efficiency experiments. For the unified benchmark we recommend using offline scoring only.
- **Per-question cost**: scoring a single trace needs no extra forward pass, only generation with `logprobs=20`; the overhead lies in the serialization, memory, and Python processing of vLLM's logprob objects (about 21×L of them). We recommend keeping only a float array of C_i per token on the generation side, as the official code does. The paper uses 512 traces/question; our benchmark can use N=8-64 traces/question, reusing the same traces to compute all variants.
- **Pitfalls and deviations from the original definitions**:
  1. DeepConf's "average trace confidence" is **not** the self-certainty of Kang et al. 2025: the latter is KL(U‖p), which needs the full-vocabulary −(1/V)Σ_j log(V·p_j), while here only the top-20 −mean log p is used, a top-k proxy. In the unified benchmark the two should be named separately as two baselines and not substituted for each other.
  2. vLLM's logprobs by default come from raw logits (before temperature, top-p, and top-k processing; verify `logprobs_mode` for the vLLM version in use). Whether the paper also uses the raw distribution is **not reported**; the official code uses vLLM's default.
  3. The official code averages over the 20 or 21 entries returned by vLLM (including the sampled token). To be faithful to the paper, we recommend strictly taking the top-20 entries and excluding the extra appended sampled token; both can also be stored for alignment.
  4. Short-trace degeneration: on easier problems such as MATH500 and GSM8K, Qwen3-8B traces are often shorter than 2048 tokens. Then bottom, lowest, and tail all degenerate to C_avg (code behavior). We recommend additionally reporting windows 512 and 1024 (L(512)/L(1K) in App. C.4), and stating the degeneration rule in the notes or the code.
  5. The score takes the mean over all generated tokens of thinking + answer, consistent with the paper and the code. If vLLM has a reasoning parser enabled, confirm that logprobs still cover the `<think>` segment. A truncated trace (reaching max_tokens) has no boxed answer and should be judged incorrect, and the truncation rate should be recorded.
  6. C_i has no upper bound and is not a probability, so ECE/Brier cannot be computed directly; a calibration mapping is needed (e.g. isotonic/Platt on held-out data), which the paper does not provide.
  7. Qwen3-8B AIME25 is the paper's own reported "confidently wrong" counterexample. Our AUROC results may be close to or below 0.5 on some datasets, which is not necessarily an implementation error.
- **Recommended faithful implementation setup**:
  - Qwen3-8B, `enable_thinking=True`, with "Please reason step by step, and put your final answer within \boxed{}." appended after the question
  - vLLM `SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768, logprobs=20)`
  - C_i = −mean(top-20 logprobs), window n=2048, stride 1, full windows only
  - bottom 10%, tail 2048
  - voting variants: Measure@K and top-{10,90}%@K
  - grading via boxed extraction + `math_equal` (Dynasor version or equivalent implementation)

## Appendix: DeepConf-Vote (multi-sample version)
`baselines/consistency.py: DeepConfVote`. Reproduces the paper's offline voting: keep the top-η answered responses by trace confidence, use confidence-weighted voting, and the response's score is the weighted vote share of its answer. The paper uses 512 traces with η of 10% or 90%; this project uses K=8, so the η grid is {0.1, 0.5, 0.9, 1} (1 means weighting only, no filtering), selected together with the confidence variants by val AUROC.
