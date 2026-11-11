# ReasonProbe: Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification

> This note records a reading of the original paper and official implementation; for the adaptation and settings currently used in this repo, see the [experiment protocol](../protocol.md).
- Naming note: the paper does not name the method; this project keeps "ReasonProbe". RMCB (Khanmohammadi et al., EACL 2026) reproduced it as a baseline and calls it **PHSV** (`models/PHSV_*.py` in that repo).
- Paper: COLM 2025 (Second Conference on Language Modeling; the OpenReview search API shows venue = "COLM 2025", forum id O6I0Av7683; the forum page requires human verification and could not be opened directly); arXiv:2504.05419 (v1 only, 2025-04-07, PDF header "Preprint. Under review."; this note is based on arXiv v1; the camera-ready could not be downloaded); https://arxiv.org/abs/2504.05419 ; Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He (NYU / NYU Shanghai)
- Official code: https://github.com/AngelaZZZ-611/reasoning_models_probing (the README calls itself "Official code", and the HF Papers page also links this repo; **no LICENSE file**; Python/PyTorch + HF transformers + bitsandbytes + vLLM generation + spaCy segmentation + Gemini API annotation; usable: contains generation, chunking, Gemini annotation, representation extraction, grid-search training, and test scripts, and provides trained probes (Google Drive); data "Coming soon"; last push 2026-05-14)
- Category: supervised probe on hidden states (intermediate-answer-level / chunk-level correctness probe for long-CoT reasoning models)
- Access requirements: white-box (**last-layer** hidden state, position = the last token of each reasoning chunk; in the code the input is `question + <think>\n + first i chunks joined with "\n\n" + "\n\n"`, taking the last token of the sequence, i.e. the "\n\n" after the chunk); training labels: Gemini 2.0 Flash extracts the intermediate answer from each chunk and compares it with gold; needs an extra LLM (Gemini) for extraction and grading; needs spaCy for segmentation

## 1. Method design
- Data construction (§3.1): take the reasoning text inside `<think>`, split it into segments by "\n\n"; the keywords "wait", "double-check", "alternatively", "make sure", "another way", "verify", "to confirm" (Table 3) mark the start of a new reasoning path, and segments of the same path are merged into a chunk; Gemini 2.0 Flash extracts the intermediate answer of each chunk and judges correctness (Table 6 prompt); adjacent chunks without an answer are merged into the nearest chunk with an answer. This yields {(c_i, y_i)}.
- Input features: the last-layer, last-token hidden state e_i ∈ R^m of each chunk (prefix up to the end of that chunk) (§3.1). No multi-layer features, no pooling.
- Probe structure: two-layer MLP: σ(W2·ReLU(W1 e + b1) + b2), W1 ∈ R^{m×d}; d ∈ {0,16,32}, and d=0 degenerates to a linear probe (§3.2, Table 7; code `MLPProbe`/`LinearProbe`). Many optimal configurations have d=0 (Table 8).
- Loss: weighted BCE, positive-class weight = α·w, where w = ratio of negative to positive training samples and α is a hyperparameter (§3.2; code `BCEWithLogitsLoss(pos_weight=α·w)`).
- Optimization: Adam, batch 64, at most 200 epochs, early stopping on validation loss (§4.1; code has a patience variable).
- Hyperparameter grid (Table 7): lr ∈ {1e-3,1e-4,1e-5}, α ∈ {0.3,0.5,0.7,0.9,1.0,1.5,2.0,3.0}, weight decay ∈ {0.001,0.01,0.1}, d ∈ {0,16,32} (216 combinations in total). Selection rule: rank by validation accuracy and pick the one with the fewest parameters (smallest d) among the top 10 (§4.1). The optimal hyperparameters for each setting are in Table 8.
- Output: sigmoid probability p_i ∈ [0,1] (probability that the intermediate answer is correct); the paper stresses good calibration (ECE<0.1).
- Extended uses: (a) "look-ahead": use the hidden state of an intermediate segment of a chunk to predict the correctness of an upcoming answer (§4.5); (b) early stopping: score each intermediate answer in order, and at the first p_i > Thr truncate and take that intermediate answer as the final answer (§5).

## 2. Dataset settings
- Math: GSM8K, MATH, AIME; logic: KnowLogic (5.4k multiple-choice questions, filtered to a single correct answer) (§4.1).
- Splits (App. A.1): training sets are all downsampled to ≤1000 questions; GSM8K test has about 1317 questions; MATH uses the original training set + the MATH-500 test; AIME trains on AIME 1983–2024 and tests on AIME 2025 (30 questions); KnowLogic uses a random 80/20 split. Samples exceeding the maximum length (30K) are discarded.
- Chunk statistics (Table 4, examples): R1-Distill-Llama-8B MATH train 1000 questions/6259 chunks, test 491 questions/3380 chunks, mean chunk length 615.1 tokens, positives 76.91%; AIME train 922 questions/7158 chunks, test 30 questions/323 chunks, mean 1652 tokens, positives 35.24%. Note: Table 4 does not list statistics for QwQ-32B.
- Correctness label: Gemini 2.0 Flash judges whether the intermediate answer equals gold (Table 6); final-answer correctness is set up separately in §4.4.
- Generations per question: not explicitly reported in the paper (the code README example uses one rollout per question, temperature 0.6).

## 3. Experimental protocol
- Train/validation split: each probing dataset is randomly split 8:2 into train/validation (§4.1); the code splits by .pt file, each containing all chunks of 50 questions, so the actual split is **question-level**. The test set comes from independent test questions.
- ID evaluation: test on the same dataset; OOD: cross-dataset (GSM8K/MATH/KnowLogic train → four test sets) (Table 2, 11–15).
- Metrics: ROC-AUC (Fig. 2), ECE (Naeini et al. 2015), Brier (Table 1), accuracy/precision/recall/macro-F1 (Table 9–10). The number of ECE bins is not reported.
- Decoding: vLLM, max output 30K (App. A.1); temperature not reported in the paper (code default 0.6); the early-stopping experiments use max 10K tokens (§5.1).
- Prompt: `<BOS><|User|>{instruction} Please reason step by step, and put your final answer within \boxed{}.<|Assistant|>` (Table 5); the same prompt is kept when extracting representations.
- Seeds: not reported (code seed=42). Compute cost: not reported.
- Code-level implementation details (not in the paper): representation extraction loads the model with **4-bit NF4 quantization** (bitsandbytes, compute dtype bf16); the batched forward passes no attention_mask (left padding, pad tokens are attended to); two BOS tokens are hand-written in the prompt string, and the tokenizer adds yet another BOS.

## 4. LLM backbone settings
- DeepSeek-R1-Distill-Qwen-1.5B / 7B / 32B, R1-Distill-Llama-8B / 70B (SFT-distilled reasoning models), QwQ-32B (RL reasoning model) (§4.1). Non-reasoning control: Llama-3.1-8B-Instruct (§4.4).
- Precision: not reported in the paper; representation extraction in the code is 4-bit NF4.
- Framework: vLLM for generation; HF transformers for representation extraction.
- Context: max generation 30K tokens; mean chunk length 272–1842 tokens (Table 4); representation extraction needs a forward pass over "question + prefix reasoning", where the prefix can reach thousands to over ten thousand tokens.

## 5. Main results and conclusions (brief, key numbers)
- ID: all probes have ROC-AUC > 0.7 and ECE < 0.1 (§4.2; in Table 1 some AIME ECE values are 0.10–0.14 and KnowLogic Qwen-32B is 0.10, so not all are strictly <0.1). R1-Distill-Qwen-32B has ROC-AUC > 0.9 on AIME. Math is better than logic; larger models are better; the Qwen series has a stronger signal.
- Table 1 (ECE/Brier, MATH): Llama-8B 0.03/0.14, Llama-70B 0.07/0.10, Qwen-1.5B 0.04/0.12, Qwen-7B 0.03/0.10, Qwen-32B 0.06/0.09, QwQ-32B 0.13/0.10.
- OOD (Table 2, R1-Distill-Llama-8B): MATH-trained → GSM8K AUC 0.83/ECE 0.04, → AIME 0.76/0.28, → KnowLogic 0.63/0.08; GSM8K ↔ MATH transfers well; calibration worsens when transferring to AIME; logic ↔ math transfer is unstable. The README suggests using the MATH-trained probe as a general-purpose probe.
- Reasoning vs non-reasoning (Fig. 3, MATH): for reasoning models the intermediate-answer ROC-AUC is 0.85 and the final-answer ROC-AUC 0.86 (values from the figure); the non-reasoning Llama-3.1-8B-Instruct is clearly worse (about 0.66), with higher ECE as well.
- Look-ahead (§4.5, Fig. 4): peak accuracy 79%, dropping by 14%/10%/5% at chunk positions 10%/50%/95%; ECE reaches its lowest value of 0.03 at about the 60% position.
- Early stopping (§5.2, Fig. 5, R1-Distill-Llama-8B/MATH): at Thr=0.85 accuracy is 88.2% (almost the same as without early stopping) and generated tokens drop by about 24%; at Thr≥0.9 accuracy is 88.6% (same as without early stopping) and tokens drop by 19%; at equal token budget accuracy is up to about 5% higher than static early stopping (87.4% vs ~82.5%).

## 6. Fit to this project's unified benchmark
- Domain match: **highest**. It is the only work that directly trains a hidden-state correctness probe on long-CoT reasoning models (R1-Distill/QwQ) + math, and can be regarded as the direct precedent for this project's "probe at the last reasoning step / final answer".
- Response-level adaptation in this project: our target is the final correctness of the whole response, so we recommend two faithful variants:
  (a) **Final-answer version** (consistent with the paper's §4.4 "Reasoning model (Final)"): take the last-layer hidden state of the last token of the final response (or the last token of the `\boxed{}` answer after `</think>`), with label = final-answer correctness;
  (b) **Last-chunk version**: chunk according to the paper's rules and take the end token ("\n\n") of the last "chunk containing an answer" before `</think>`.
  Intermediate-answer-level training (needs Gemini extraction) is costly and has noisy labels; it can serve as an additional experiment and is not required.
- Extraction position suggestion: the last layer (the paper's setting) as the main one; additionally report multiple layers (e.g. every 4 layers) as an ablation, to align with the layer selection of SEP/SupUQ.
- Cost: vLLM does not expose hidden states, so one HF teacher-forcing forward over prompt+trace is needed (2k–16k tokens). The final-answer version needs only one forward per response; if the chunk version runs "one forward per chunk prefix" like the official code, the cost is O(#chunks × length), and it should instead use a single forward and index the hidden states at each chunk-end position (equivalent for a causal model, except that the tokenization boundary of the trailing "\n\n" must be aligned).
- Capacity alignment: the paper searches tiny MLPs with d ∈ {0,16,32}, whose capacity differs a lot from SEP (linear) / SupUQ (RF or MLP) / SAPLMA (256-128-64 MLP); we suggest reporting two tiers, "linear + small MLP", and using the same validation set and the same hyperparameter budget for all learned baselines.
- Recommended faithful implementation settings: last layer, last token; weighted BCE (pos_weight = α·neg/pos); Adam, batch 64, ≤200 epochs, early stopping on validation loss; grid per Table 7; selection rule "smallest d among the top 10 by validation accuracy"; question-level 8:2 split; extract in bf16 (instead of 4-bit) and note the difference from the official code; pass attention_mask.
- Risks/red flags:
  1. In the appendix Tables 11–15 (OOD tables for different models) the ID diagonal values are identical across 5 models (GSM8K 0.82/0.04, MATH 0.84/0.03, KnowLogic 0.67/0.07), inconsistent with the model-specific ECE in Table 1 (e.g. the QwQ-32B MATH ECE is 0.13 in Table 1 but 0.03 on the Table 15 diagonal), apparently copied from Table 2; be careful when citing OOD numbers.
  2. The AIME test has only 25–30 questions and 104–323 chunks, so the AUC variance is large.
  3. Labels depend on Gemini's extraction of intermediate answers, with noise unevaluated; chunk segmentation depends on keywords, and keyword coverage must be rechecked for the Qwen3 thinking style.
  4. The code has no LICENSE; representation extraction uses 4-bit quantization and passes no attention_mask (not stated in the paper).
  5. Only the last layer is used, with no layer sweep; works such as SEP show the optimal layer is often in the middle-to-late layers, so a layer ablation must be added in this project.
