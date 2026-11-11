# Unified Experimental Protocol

All methods score the same sampled responses, learn and select only on the training pool, and are evaluated on frozen evaluation sets. The single source of truth for backbones, datasets and the method list is `uqbench/configs/benchmark.yaml` (decoding settings are in `uqbench/configs/models.yaml`). Code locations are given in parentheses.

## 1. Task and Labels

- **Task**: response-level correctness estimation. Given a question x and one complete sampled response (reasoning + final answer), estimate P(the response is correct).
- **Labels** (`uqbench/verify.py`):
  - Math: take the last `\boxed{}` in the final response after the reasoning block and test equivalence with `math_verify`; strip a leading definition (`z(t) =`, `E_v =`) and test once more; when `math_verify` judges them unequal, run a numeric comparison (`numeric_equal`): scientific notation and percent notation, a leading `x =` / `\approx`, and units wrapped in text are all treated as the same value; when both sides are written as approximate quantities (with a decimal point or exponent), a relative error ≤ 1% counts as equal, and integers must match exactly; when both sides have at least 3 significant digits, equality after rounding to the coarser precision also counts as correct (20.4 and 20.39). These rules are also used for clustering. For (answer, reference answer) pairs that the rules still judge wrong and that are not two pure numeric values, Claude Haiku 5.5 (adaptive thinking, Batch API, `uqbench/labeling/regrade.py`) sees the question and the reference answer and decides whether the answer is correct: equivalent expressions, equations and notations count as correct; when the question specifies a form, range or particular case, it must be satisfied; when the reference answer is an exact value, exact equality is required, and a 1% error is allowed only when the reference answer is a decimal approximation. The LLM judgment changes only the label, not the answer cluster. Labeling runs after generation (`scripts/run_labels.sh`) without rerunning GPU stages; every change is logged in `OUTPUT_DIR/<model>/regrade.jsonl`.
  - Multiple choice: compare the option letter.
  - Open-ended QA: exact match against the alias list after normalization. For responses whose reasoning block is closed but that answer in a single sentence without `\boxed{}`, Claude Haiku 5.5 (Batch API, `uqbench/labeling/extract.py`) copies the short answer given at the end of the final response verbatim (without seeing the reference answer and without judging correctness); the answer is then graded by the same exact match and clustered by the same rules.
  - Responses whose reasoning block is not closed (exceeds the 32k budget) or from which no answer can be extracted are kept and judged wrong; their proportion is reported separately.
- **Answer clusters**: responses to the same question are clustered by the deterministic equivalence rules above; for math questions, distinct answers that the rules did not merge are then judged pairwise for equivalence by Claude Haiku 5.5 without seeing the reference answer (`uqbench/labeling/regrade.py pairs-*`, results in `OUTPUT_DIR/math_equiv/equivalent.jsonl`, read by `cluster_answers` by question id; runs sampled together are judged together), so that grading and clustering use the same equivalence relation; responses without an answer each form their own cluster.
- **Evaluation unit**: all K responses of each question are evaluation units. Question-level methods (SNNE, SE) give the same score to all responses of a question.

## 2. Data (`uqbench/pipeline/data.py`)

| Role | Dataset (test set name) | Questions | Notes |
|---|---|---:|---|
| Training pool `math_pool` | MATH train | 2,998 | Stratified by level × subject, deduplicated against all test questions |
| Training pool `mixed_pool` | MMLU-Pro non-math + TriviaQA | 1,000 | 500 questions each, disjoint from all test questions |
| Test | MATH-500 (`math500`) | 500 | |
| Test | MATH (non-500) (`math_rest`) | 498 | MATH test questions not in MATH-500, stratified by level × subject |
| Test | OlympiadBench (`olympiadbench`, OE_TO_maths_en_COMP) | 581 | Text-only English; multi-answer questions removed; questions with units get a unit hint |
| Test | Minerva-Math (`minerva`) | 272 | |
| Test | AIME 2024-2026 (`aime`) | 90 | Merged into one set |
| Test | GPQA-Diamond (`gpqa`) | 198 | Four-way multiple choice; option order shuffled with a fixed seed |
| Test | GPQA-Ext. (`gpqa_ext`) | 348 | All GPQA-Extended questions not in Diamond; same as above |
| Test | MMLU-Pro non-math categories (`mmlupro`) | 2,003 | Two disjoint samples stratified by category (1,002 + 1,001), 10-way |
| Test | SuperGPQA (`supergpqa`) | 1,997 | Two disjoint samples stratified by discipline (999 + 998), up to 10-way |
| Test | TriviaQA rc.nocontext validation (`triviaqa`) | 1,960 | Two disjoint random samples (1,000 + 1,000), after removing 20 question ids that appear twice in the first sample |

- **Training pool** (`benchmark.yaml: training_pools` = `math_pool + mixed_pool`): K=4 per question, split 80/20 by question into train / val. All learned methods (CAPE, Combiner, the learned parts and calibration maps of each baseline) are fitted and selected on the same training pool. `pool: math_pool` is used only for the answer-set judge ablation.
- **Test sets and generated datasets**: the GPU stages run per generated dataset (`benchmark.yaml: generation_order`), and the CPU stages run per test set. `benchmark.yaml: test_union` lists the test sets composed of several generated datasets: `math_rest` = `math_heldout`, `gpqa_ext` = `gpqa_heldout`, `mmlupro` / `supergpqa` / `triviaqa` = the dataset of the same name ∪ `*_heldout` (`uqbench.pipeline.data` writes all generated datasets at once; the later-added part is deduplicated by question text against the first sample and the training pools); the other test sets are simply the generated dataset of the same name. Question ids carry the generated dataset name and remain unique after merging; the bootstrap resamples by question.
- `math_all` merges the five math test sets (`math_sets`: MATH-500, MATH (non-500), OlympiadBench, Minerva, AIME) by number of responses (pooled), without macro averaging.

## 3. Result Locations

Main results are written to `results/<model>/`, and per-response scores to `$OUTPUT_DIR/<model>/scores/test/`.

## 4. Backbones and Decoding (`uqbench/configs/models.yaml`)

| Model | Format | Decoding (model-card recommendation) |
|---|---|---|
| Qwen3-8B | thinking | T=0.6, top-p=0.95, top-k=20 |
| DeepSeek-R1-Distill-Qwen-7B | thinking | T=0.6, top-p=0.95 |
| gpt-oss-20b | harmony, reasoning effort = medium | T=1.0, top-p=1.0 |

- Samples per question: K=8 on evaluation sets, K=4 on the training pool; generation budget 32,768 tokens.
- Prompt: `{question}` + reason step by step and put the final answer in `\boxed{}` (option letter for multiple choice, short entity for QA).
- Sampling seed = seed + global index of the question in `generation_order` (+ optional `seed_offset`); new datasets may only be appended to the end.
- The ablation backbone (`ablation_backbone`, Qwen3-8B) has an additional sampling run with independent seeds, `qwen3-8b-x8`: responses 9-16 on MATH-500, OlympiadBench, Minerva, AIME and GPQA-Diamond, used only for the K=16 experiments (`kscale`).
- Dependence intervention (`qwen3-8b-pf25/50/75`, `uqbench.pipeline.generate.shared_prefixes`): all K responses of each question continue the first L tokens of the first response of the main run, with L = f × the median length of the remaining responses (f = 0.25 / 0.5 / 0.75); when the first response is not longer than L, it is reused whole. L does not depend on the responses being continued, so every response is still an exact sample of the model and only the dependence between responses changes. Six test sets: MATH-500, OlympiadBench, Minerva, AIME, GPQA-Diamond, TriviaQA.

## 5. Pipeline (`uqbench/`; every stage can resume from a checkpoint and can be run on its own with `python -m uqbench <stage> --model ...`)

| Stage | Module | Contents |
|---|---|---|
| GPU 1 generation | `generate` | vLLM sampling; saves the exact token ids of the prompt and output; no logprobs; extracts answers and grades them when merging |
| GPU 2 replay | `replay` | One HF teacher-forcing forward pass computing per-token full-vocabulary statistics: log-prob of the sampled token, entropy, Self-Certainty KL(U‖p), top-20 confidence; saves hidden states at 3 positions (last answer token, end of reasoning block, last token) × every 4th layer + the final layer |
| GPU 3 energy | `energy` | Teacher-forced forward pass; raw logit of the sampled token = final-layer hidden state · output embedding of that token (without projecting the whole vocabulary); saves the mean raw logit over the answer part and over all tokens of each response, the input of the SemEnergy baseline |
| GPU 4 follow-up | `followup` | Short continuations from a shared prefix: the P("1") prompt of CISC; the 10-level verbalized confidence of LRMVerbal (decoding constrained to the 10 class names) |
| GPU 5 reward | `reward` | A general outcome reward model (Skywork-Reward-V2-Llama-3.1-8B) scores (prompt, visible part of the response), i.e. the ORM baseline |
| CPU evaluation | `evaluate` | Fit and select on the training pool, score the test sets, metrics and bootstrap CIs; writes `metrics.csv` and `$OUTPUT_DIR/<model>/scores/test/*.npz` |
| CPU controls | `controls` | Feature-aligned combiner ladder and within-cluster evidence correlation (ICC) |
| CPU decision | `decision` | Unanimous-question separation, before/after projection comparison, robust certificate |
| CPU within-question ranking | `within_question` | Within-question AUROC |
| CPU refinement | `refinement` | Cross-fitted isotonic recalibrated Brier, "none correct" detection, standalone ∅ classifier, selective answering, projection steps; writes per-question `questions.csv` |
| CPU attribution | `attribution` | Runs after `refinement`, reads only stored scores and does not refit: safety of rule one (`attribution_safety.csv`: fraction of questions on which the projection and "calibrated probability × vote share" each raise the Brier score, clipping trigger rate, gain covered by the certificate); ∅ detection of rule two by stratum (all, unanimous, split, 1/2/≥3 answers) (`attribution_null.csv`: residual mass of CAPE, the standalone ∅ classifier, the coherent combiner and the projected DeepConf, 1 − top answer, and two CAPE variants with a constant ∅ logit and a per-response loss); paired bootstrap of CAPE against the residual mass of the coherent combiner (`attribution_null_paired.csv`). |
| CPU audit | `audit` | How many consistent errors a more lenient grading would accept (`verifier_audit.csv`) |
| Dependence intervention | `dependence` | Structure of the shared-prefix runs (accuracy, ICC, none-correct rate, consistent-error rate, effective sample size), metrics of each method under unchanged fits, CAPE cross-fitted by case (learned vote coefficient and ∅ constant) and per-question paired bootstrap |
| Ablation only | `judge`, `kscale`, `ablation` | Answer-set judge (may reason ≤1,024 tokens first; run only on the first sample of each test set and the two training pools); K=1…16 lower-bound experiment (MATH-500, OlympiadBench, Minerva, AIME, GPQA); CAPE ablations (§8) |

Replay uses raw logits (T=1), consistent with vLLM's raw logprobs. The attention sink of gpt-oss has no flash-attention implementation, so replay uses a blockwise eager-attention forward pass (2,048 tokens per block).

**Scripts** (`scripts/`; all read `.env` through `scripts/environment.sh`):

- `download_data.sh`, `download_models.sh`: raw data, processed datasets and backbones.
- `run_gpu.sh <model> [gpus=0,1,2,3]`: generate → replay → energy → followup → reward on all generated datasets.
- `run_labels.sh <model> [<model> ...]`: LLM-assisted labeling (math regrading, answer equivalence, short-answer extraction); pass runs sampled together at once.
- `run_cpu.sh <model>`: evaluate → controls → decision → within_question → refinement → attribution → audit. Defaults to 3 BLAS threads: the float32 PCA of the CAPE probe and SEP varies in the last digits with the thread count, and is bitwise reproducible at the same thread count.
- `run_benchmark.sh <model> [gpus=0,1,2,3]`: runs `run_gpu.sh`, `run_labels.sh`, `run_cpu.sh` in order.
- `run_ablations.sh [gpus=0,1,2,3]` (after `run_benchmark.sh` has finished for all backbones): x8 GPU stages, labeling and kscale for the ablation backbone; the answer-set judge for the three backbones; the ablation study judge.
- `run_dependence.sh [gpus=0,1,2,3]`: the three shared-prefix runs of the dependence intervention (generation, replay, energy, labeling) and the `dependence` study.
- `submit_benchmark.sh <model>`: reads `SLURM_ACCOUNT` and `SLURM_PARTITION` from `.env` and submits `slurm_benchmark.sbatch` (`run_benchmark.sh` on one 4-GPU node).

The dataset and backbone lists are read by `scripts/datasets.sh` from `uqbench/configs/benchmark.yaml`; `PYTHON` can specify the interpreter.

## 6. Methods (`baselines/`, `cape/`; for the selection rationale see [baselines/README.md](baselines/README.md))

The list is in `uqbench/configs/benchmark.yaml: methods`: `baselines` (table row order), `references` (SC, SE), `constructed` (our constructed controls DeepConf×SC, Combiner), `single_response` (per-response scoring estimators).

| Method | Setting | Parts learned / selected on the training pool |
|---|---|---|
| DeepConf | confidence = −mean top-20 log-prob; mean / bottom-10% / lowest / tail, window 2,048 | Variant selected by val AUROC |
| CDG | s = C̄ + βΔC, where ΔC is the confidence difference between the last 10% and first 10% of tokens | β estimated on train by the paper's rule |
| SelfCertainty | Token average of the full-vocabulary KL(U‖p) | None |
| LRMVerbal | Shared-trajectory proxy version (the paper's two-round variant); class names mapped to interval midpoints | None |
| CISC | Vote share of the same answer weighted by within-question softmax(c/T) of the P("1") confidence | T selected by val AUROC on the paper's 80-point grid [1e-4, 1e4] |
| ReasonProbe | Final-layer hidden state at the last answer token; MLP d∈{0,16,32}, weighted BCE | The paper's 216-configuration grid; among the top 10 by val accuracy, the one with the smallest d |
| SNNE | ROUGE-L on the final response after the reasoning block, τ=1, with the self-similarity term; mean inside the log (minus log K) so that a calibration map fitted at K=4 can be used at K=8 | None |
| DeepConf-Vote | DeepConf offline voting: keep the top-η responses with an answer by trajectory confidence, confidence-weighted vote; the response score is the weighted vote share of its answer | Confidence variant × η∈{0.1, 0.5, 0.9, 1}, selected by val AUROC |
| ConsistCal | The sample-consistency measures of Lyu et al. 2025 (agreement, first-second distance, normalized entropy) made per-response | Measure selected by val AUROC |
| SEP | Semantic entropy probe: logistic regression on PCA-256 of answer-position hidden states (middle layer + final layer), predicting whether the question's semantic entropy is above the training median; not trained with correctness labels | C selected by val AUROC |
| ORM | Scalar reward of a general outcome reward model (`uqbench/pipeline/reward.py`) | None |
| SemEnergy | Sum over the responses in a cluster of the mean raw logit of the answer part, i.e. −E(C) (`uqbench/pipeline/energy.py`) | None |
| CoCoA | Vashurin et al. 2025: the response's own mean token NLL × its mean dissimilarity to the other responses | Similarity (ROUGE-L or answer equivalence) selected by val AUROC |
| DeepConf×SC (control) | Calibrated DeepConf probability times the vote share of the response's answer | DeepConf variant and calibration map |
| Combiner (control) | Per-response logistic regression: CAPE's 22 trace statistics + 5 vote features, no projection, no gating | L2 strength selected by val log loss |
| SC†, SE† | Vote share; discrete semantic entropy (entropy of answer clusters), using the same K responses as the multi-sample methods | None |
| CAPE | `cape/model.py`, see [method notes](method.md) §3 | The two answer models, probe, L2 strength |
| CAPE variants (ablation rows, `main=False`) | Fitted together with the main CAPE in `uqbench/pipeline/evaluate.py`: scalar evidence, core evidence (only the 10 baseline signals), w/o support gate, learned α, w/o null, constant null (`null_features=False`), per-response loss (`objective="response"`), w/o votes, + difficulty | Same as CAPE |

† Same-budget references, not counted as 2025+ baselines.

- **`+Coh`**: project the method's calibrated probability onto the coherent set (`cape/coherence.py: CoherentProjection`). After projection the raw projection is used directly, with no second calibration map (the theory and the certificate describe exactly this object; a second map would leave the coherent set).
- **Feature-aligned controls** (`uqbench/studies/controls.py`): Combiner → projection → add the hidden-state probe → add exactly the same support gate as CAPE (`w = 1[format]·min(1, p/α)`, weighting the logits of the two logistic models by w), i.e. the "coherent combiner"; and the modular implementation "coherent combiner + standalone ∅ classifier (trained on the same training pool with CAPE's evidence) + renormalization".

## 7. Metrics and Statistics (`uqbench/metrics.py`, `uqbench/calibration.py`, `uqbench/stats.py`)

- **Main metrics**: Acc (backbone accuracy), AUROC↑ (ties counted as 1/2), Brier↓, ECE↓ (15 equal-width bins).
- **Supplementary**: AURC↓ (exact expectation under random tie-breaking), AUPRC (errors), NLL.
- **Probabilization**:
  - Methods that output scores use a Platt map fitted on val.
  - Methods that output probabilities choose between "keep the native probability" and "Platt" by val NLL, where the NLL of Platt is 2-fold cross-fitted by question.
  - `+Coh` uses the raw projection (§6).
  - Brier / ECE of the native probability are also reported (`brier_raw`, `ece_raw`).
- **Confidence intervals**: cluster bootstrap by question, 2,000 resamples; significance uses a question-level paired bootstrap, with Benjamini-Hochberg 5% FDR across pairs.
- **Winning cells**: CAPE is no worse than the best of all baselines and references in that cell (test set × metric). All 30 (backbone, test set) pairs are counted (`benchmark.yaml: low_parse` = 1.0); responses with no extractable answer count as wrong (about 76% of R1-Distill / TriviaQA responses have no `\boxed{}` answer).

## 8. Additional Analyses and Ablations

- Label-free Brier certificate and robust certificate; `+Coh` for every method.
- K=1…16 coherence lower-bound experiment (ablation backbone).
- Mean confidence on unanimously correct / unanimously wrong / split questions; "none correct" detection and escalation by ∅ mass; AURC of selective answering.
- CAPE ablations: the component variants are the ablation rows of `evaluate`; the features study of `uqbench.studies.ablation` is run only on the ablation backbone, and the judge study on the three backbones (only on questions the judge has run on, i.e. the first sample of each test set, excluding `math_rest` and `gpqa_ext`). For conclusions see [method notes](method.md) §5.
- Other forms of the answer model (`uqbench.studies.ablation --studies forms`, three backbones): a learned within-cluster correlation r (cluster evidence multiplied by κ(n) = n/(1+(n−1)r)); a question-dependent ∅ logit (mean evidence, vote entropy, length, unclosed rate, number of clusters/K, no-answer rate).
- Grader-independent clustering (`--studies equivalence`, math sets and TriviaQA, three backbones): math answers are clustered by whitespace-stripped strings, or by strings after additional LaTeX surface normalization, and short answers by DeBERTa-large-MNLI bidirectional entailment (`uqbench/pipeline/nli.py`); labels are unchanged. CAPE and SC are refitted on the reclustered training pool; the calibrated probabilities of published estimators are reprojected onto the new clusters, and the robust certificate is checked question by question.
- Cost-based answering (`uqbench/studies/abstention.py`): for each question, the method answers with its most likely answer or abstains; a correct answer scores +1, a wrong answer −(1−α)/α, abstention 0; α ∈ {0.01, …, 0.5}. The plug-in rule "answer if probability ≥ 1−α" is the Bayes decision for a calibrated prediction and has no tunable parameter; also reported are, for each method, the threshold selected on the validation part of the training pool, the post-hoc optimal threshold (regret), and a version in which everything answers the majority answer. Paired bootstrap over the questions shared by the three backbones.

## 9. Result Files

- **Per backbone** (`results/<model>/`): `main_table.md`, `metrics.csv` (all methods and variants, CIs), `hyperparameters.json`, `answer_structure.csv`, `certificate.csv`, `robust_certificate.csv`, `null_mass.csv`, `unanimous.csv`, `coherence_effect.csv`, `controls.csv`, `evidence_icc.csv`, `within_question.csv`, `refinement.csv`, `null_signal.csv`, `questions.csv`, `selective_qa.csv`, `projection_steps.csv`, `stacker_null.csv`, `attribution_*.csv`, `verifier_audit.csv`, `kscale.csv`, `ablation_features.csv`, `ablation_judge.csv`, `ablation_forms.csv`, `ablation_equivalence{,_projection}.csv`, `dependence_{structure,iid_curve,methods,refit,refit_params,bootstrap}.csv` (ablation backbone); cross-backbone `results/abstention{,_paired}.csv` and `results/robustness_*.csv`.
- **Per-response scores**: `$OUTPUT_DIR/<model>/scores/test/<test set>.npz` (plus `controls_<test set>.npz` and `val.npz` for the validation part of the training pool).
- **Summary**: `results/summary.md` (`python -m uqbench.reporting.report`); it also exports all metrics for every (backbone, test set, method): `results/<model>/per_dataset/<dataset>.csv` (one row per method) and the merged `results/all_metrics.csv`; per-response judgments are in `results/<model>/responses/<dataset>.parquet` (generated dataset, question id, sample index, extracted answer, answer cluster, whether correct, calibrated probability `p:` and raw score `raw:` of each method, probabilities of controls and modular implementations, and the per-question "none correct" quantity `q:`); CAPE's rank against all competing methods (including projected versions) on every (backbone, test set), the strongest competitor per metric, and the gap to the main rivals are in `results/cape_vs_competitors.csv`.
- **Paper**: `python -m visualization.build` writes tables to `latex/tables/<label>-panel-<i>.tex` and figures to `latex/figs/<label>.pdf` (`PAPER_DIR`); numbers in the paper are written by hand in the text.

## 10. Known Limitations

- LRMVerbal is a proxy version: it does not reproduce the original single-round form of estimating confidence while solving.
- SNNE and SE are question-level scores and cannot distinguish responses within the same question.
- The original CDG paper excluded Qwen3; we report it as usual and do not tune it for Qwen3.
- The training pool contains only math, multiple choice and short-answer QA; on other formats all learned components and calibration maps are transferred across domains.
