# Failed and Abandoned Attempts

> Research record: keeps the judgments, numbers, and trade-offs of the time. The current way to run things is defined by the [reproduction guide](reproduction.md) and the [experiment protocol](protocol.md); generator names that have since been deleted appear in the text only for traceability.

Approaches that were tried but not adopted, and engineering pitfalls encountered. "Winning cells" are defined in [experiment protocol](protocol.md) §7. Numbers in the tables come from the settings of the time; they indicate direction and are not to be compared with paper numbers.

## 1. Answer Model and Calibration

| Attempt | Result | Decision |
|---|---|---|
| Learn the consistency-prior exponent α on the training pool | Learned α ≈ 1.2–1.3, over-trusting votes on multiple-choice questions: with a finite option set, accidental agreement among wrong answers is far more common than in open-ended math. R1 GPQA Brier 0.189→0.221 | Fix α = 1 (exchangeable prior); the learned-α version is reported as an ablation row |
| Apply Platt calibration to all methods uniformly | Degrades probabilities that were already calibrated: CAPE math ECE on Qwen3-8B 0.034→0.054 | Choose between native probabilities and cross-fitted Platt by val NLL (protocol §7) |
| Label-shift adaptation with EM, adjusting the ∅ mass | Diverges on harder test sets; the label-shift assumption (only the class prior changes) does not hold | Abandoned |
| Unlabeled EM adaptation to an "all wrong" prior shift (Saerens et al. 2002 style, adjusting π∅ by the all-wrong rate estimated on the new dataset) | Diverges on the development set; the estimated rate tends to 1 on most out-of-domain sets, Brier 0.129→0.301. EM assumes the likelihood ratio of π∅ is unchanged on new data, which fails out of domain | Abandoned |
| Add hidden-state features directly (32-dim PCA) without a gate | Slightly better in-distribution, OOD calibration collapses (Qwen GPQA Brier 0.532): few wrong samples in the training pool, so high-dimensional features overfit | Switch to the support gate (§3); within-question centering removes the collapse but is not better overall |
| ∅ logit with difficulty features (length, truncation rate, number of clusters/K, no-answer rate) | With the math-only pool, R1 GPQA Brier 0.189→0.198; on the mixed pool none-correct AUROC is slightly higher (0.798) but the development-set mean Brier is worse (0.1268 vs 0.1231) | Not adopted under the pre-agreed primary metric (development-set mean Brier); kept as an ablation row |
| Early features: answer-span probability, 8-dim hidden PCA | No stable gain on the development backbone | Not adopted |
| Selective prediction with a risk threshold selected on val | The training pool is too easy, and the threshold degrades on OOD test sets | Switch to "answer only if P(correct) ≥ 1−α"; the val-threshold version is kept only in result files |

## 2. Variants Compared on the Test Sets (Negative Results Only)

The variants below were compared directly on the test sets, before the support gate, against the CAPE of the time (10 scalar features, no probe), so they are recorded only as negative results and cannot be reported as improvements; the method therefore stays unchanged.

| Variant | Winning cells (Qwen / R1 / gpt-oss) | Conclusion |
|---|---|---|
| CAPE of the time | 20/27 · 19/24 · 17/27 | Baseline |
| Cluster evidence via log-mean-exp | 20 · 19 · 18 | Only +1 on gpt-oss |
| Separate feature map for ∅ | 20 · 19 · 18 | Same |
| Within-question relative features (subtract question mean) | 20 · 19 · 17 | No change |
| Sum the evidence, drop the vote-count prior (naive Bayes) | 15 · 9 · 14 | Clearly worse: correlated samples are treated as independent evidence |
| ∅ with log max-vote share | 20 · 18 · 15 | Worse |
| K-invariant entropy in ∅ | 20 · 19 · 15 | Worse |
| Logit temperature scaling (fit on val) | 20 · 19 · 18 | Only +1 on gpt-oss |
| Forced Platt | 18 · 15 · 17 | Worse |
| Within-question centered hidden states (16-dim / 4-dim) | 18 / 20 · 19 / 19 · 19 / 19 | Gain only on gpt-oss |
| Answer-set judge as extra evidence (no reasoning / reasoning ≤1,024 tokens) | 18/18 · 19/16 · 19/18 | Effect on AUROC and Brier ≤ 0.012; far weaker than SC when used alone |

Round 7 (2026-10-21, in response to reviewer comments, after the regrade and LLM-assisted clustering, mean over 30 pairs; `uqbench.studies.ablation --studies forms`):

| Variant | AUROC / Brier / ECE | ∅ AUROC / ∅ Brier | Conclusion |
|---|---|---|---|
| CAPE (cluster evidence averaged, r → 1; constant u_∅) | 0.8995 / 0.1026 / 0.0564 | 0.805 / 0.115 | Baseline |
| Learned within-cluster correlation r (cluster evidence multiplied by κ(n) = n/(1+(n−1)r)) | 0.8985 / 0.1042 / 0.0629 | 0.807 / 0.117 | Learned r = 0.19–0.22, far below the measured within-cluster correlation 0.58–0.90; overconfident when extrapolating from K = 4 training to K = 8; per-pair AUROC 14 wins 16 losses, Brier 12 wins 18 losses |
| Question-dependent u_∅ (mean evidence, vote entropy, length, unclosed rate, number of clusters/K, no-answer rate) | 0.8987 / 0.1071 / 0.0667 | 0.811 / 0.113 | Slightly better ∅ detection, worse answer-level Brier and ECE; the log K term is unidentifiable on a training pool where K is always 4 |

Conclusion: almost all of the gain comes from the posterior structure (consistency + ∅ + evidence fusion), not from more complex evidence; hidden-state evidence added directly hurts OOD, which is exactly what the support gate targets. The only reliably effective enhancement is a small amount of in-domain training data, which does not change the method itself.

## 3. Support Gate and Evidence (Compared on the Development Set Only)

Development set: training-pool val, the views of val subsampled to K=2/1, and `mixed_pool`.

| Attempt | Development-set result | Decision |
|---|---|---|
| Add to ∅ a missing mass that decays with K (Pitman–Yor / Good–Turing form, τ global or varying with evidence) | Learned τ is only 0.03–0.4; Brier changes ≤ 0.003 on val, K extrapolation (train K≤2 → K=4), and mixed | Not adopted. The error rate of consistent answers does follow ε + c/K, but the existing ∅ already absorbs most of it |
| Finite-option chance term (misplaced mass split 1/L to each seen option) | Qwen multiple-choice ECE 0.082→0.066, R1 0.137→0.143 | Not adopted: direction depends on the backbone |
| Cluster evidence scaled by n_c^γ (between mean and sum) | gpt-oss OOD improves, Qwen and R1 worsen | Not adopted |
| Training-time subsampling augmentation with K'=1..4 | Brier clearly better at K=1, 2, equal or slightly worse at K=4 | Not adopted (main evaluation is K=8) |
| Test-time per-dataset feature standardization (unlabeled) | All OOD worse: also erases real difficulty differences | Abandoned |
| Hidden-state probe directly as evidence, no gate | In-domain val AUROC 0.89→0.98, but Qwen mixed multiple-choice Brier 0.139→0.55 | Switch to the support gate |
| Conformal gate on hidden-state residual only (ignoring answer format) | Mean support on multiple-choice is still 0.15–0.45, Qwen multiple-choice Brier 0.196 | Switch to a "format × residual" gate |
| Enable the probe on math questions only (trace model uses the full training pool) | none-correct AUROC 0.781, below the default gate's 0.793; trace evidence only 0.737 | Not adopted: the probe supports ∅ detection equally on multiple-choice and QA |
| Enlarge the training pool to K=8 (4 more independent answers per math question) | Worse than K=4 on the Qwen3-8B development set (mean Brier 0.1388 vs 0.1367, AUROC 0.870 vs 0.875, none-correct AUROC 0.782 vs 0.789) | Not adopted under the pre-agreed rule; the extra answers were not used in any reported result |

## 3b. Round 4 Simplifications and New Evidence (Compared on the Training Pool and Development Set Only)

Selection data: 5-fold cross-validation on the training pool, training-pool leave-one-source-out (train on two sources, evaluate on the third), and the development set `dev` (1,602 questions, disjoint from all test sets and the training pool: MATH train Level 5 500, SuperGPQA 600, MMLU-Pro 202, TriviaQA 300, K = 8; used for selection only, not in the paper). Numbers are for Qwen3-8B.

| Attempt | Result | Decision |
|---|---|---|
| Cross-trajectory forced answering (each trajectory closes its reasoning at 25/50/75/100% and force-continues on every candidate answer, giving log p(c ∣ trajectory)); cross-trajectory support of a cluster goes into u_c, and the mass of each trajectory falling outside the sampled answers goes into u_∅ | Leave-one-source-out AUROC +0.047, ∅ detection 0.74→0.83 (the extreme split where the source format is entirely absent and the gate is off); cross-validation ±0.002; on the development set not better than round 3 CAPE (Brier 0.1404 vs 0.1389), and removing the "external mass" has no effect | Not adopted: the leave-one-source-out gain does not transfer to real shift; GPU cost is about one replay |
| Add question-level inputs of a modular ∅ classifier to u_∅ (votes + trace statistics and probe mean/min/max/max-cluster mean) | Cross-validation and development set equal or slightly worse | Not adopted |
| Per-answer logit of the gated stacker as g | Development-set Brier 0.1453 (worse) | Not adopted |
| Log-pool of CAPE and the modular implementation | Equal to the modular implementation | Not adopted |
| K-transfer ∅ prior (fit per-question accuracy with a beta-binomial on the training pool, shift u_∅ by the change in log-odds of P(∅∣K)) | Development-set Brier 0.1370 vs 0.1363; the prior shift for K = 4→8 is only about −0.15, and the difference between the development set and training pool is mainly difficulty (∅ rate 0.20 vs 0.095) | Not adopted |
| Learned ∅ feature (θ₁·ḡ + θ₂·H) | Development-set Brier 0.0025 worse than constant ∅; better only on leave-one-source-out | Switch to constant u_∅ (the Bayes form of Proposition 6) |
| Hidden-state principal components directly in g, trained jointly with the answer-level likelihood (replacing the separately trained per-answer probe) | Worse on both development sets (Qwen AUROC −0.005 to −0.026) | Not adopted |
| 22 trace statistics / 10 baseline signals (including P(True) and verbalized confidence) | Development set equal to the 4 token statistics (0.1369, 0.1356 vs 0.1363) | Keep only the 4 token statistics, saving the follow-up reasoning |

## 3c. Round 6: OOD Calibration (Compared on the Development Set Only)

Diagnosis: on the development-set SuperGPQA (OOD proxy), CAPE's ∅ rate matches the actual one (Qwen 0.29 vs 0.30, R1 0.35 vs 0.38); the error comes from being too sharp between answers: underconfident in low bins, overconfident by +0.15 to +0.29 in high bins. The cause is that the probe discriminates less well on hard scientific multiple-choice questions (Qwen probe AUROC: math 0.95, MMLU-Pro 0.81, SuperGPQA 0.65), while the single ψ is dominated by 3,000 math questions. On the test-set OOD, the direction of the signed error differs across backbones (R1 is underconfident on GPQA), so a global temperature cannot fix it. The selected version: a per-answer-format evidence weight ψ_f, with probe positions moved to the middle layer at the answer token + the 3/4-depth layer at the end-of-reasoning token. The table below gives the development-set SuperGPQA Brier (Qwen / R1), based on the round 5 CAPE (0.2381 / 0.1974) unless noted otherwise.

| Attempt | Result | Decision |
|---|---|---|
| Shrink the probe weight ψ (×0.4 to 0.8) | Overconfidence worsens on R1 (signed error +0.075→+0.111), Brier unchanged | Not adopted: votes themselves are more overconfident on SuperGPQA |
| Support gate with Mahalanobis / residual distance | SuperGPQA hidden-state distances are comparable to MMLU-Pro, p < 0.05 for only 3% | Not adopted: the shift is difficulty and prior, not outside the support |
| Learned ∅ feature, per-format θ₀ | 0.2393 / 0.1954, or equal | Not adopted |
| Leave-one-source-out cross-fitted probe logit (answer model learns the transferred ψ) | 0.2950 / 0.2452, ID AUROC drops to 0.76 | Not adopted |
| Stronger regularization (C = 1e-3), PC 64 / 128 / 32, Laplace-harmonized probe, per-answer target, learned α, Platt | Best PC64 gives 0.2342 / 0.1929 but ID Brier +0.005; the rest equal or worse | Not adopted |
| Probe with per-format balanced sample weights | OOD AUROC +0.01, Qwen ID Brier 0.0796→0.0865 | Not adopted |
| ψ_f (one probe weight each for math / multiple-choice / short answer) | 0.2305 / 0.1915, ID unchanged | Adopted |
| Probe positions: answer token + end-of-reasoning token | Qwen 0.2208, R1 0.1897, gpt-oss 0.2182 (round 5: 0.2174; under gpt-oss's harmony format, position 1 is the end of the analysis channel) | Adopted (lowest mean Brier over the three backbones: 0.2096; answer token only 0.2118, three positions 0.2106) |
| Single test-set evaluation after adopting ψ_f + new probe positions | Best in eight of the nine cells of Table 1; OOD AUROC 0.7907, Combiner (Stacker) 0.7911 | The owner accepted 8/9 and stopped exploring |
| Afterwards compared on the development set only: the two hidden-state views (answer position, end-of-reasoning position) each as a column of evidence; plus 4 token statistics | Development-set OOD AUROC averaged over the three backbones +0.006 (0.741→0.749), Brier 0.2096→0.2068 | Not evaluated on the test set, not adopted |

## 4. Decision-Making and Joint Modeling

- **Answer / resample / abstain policy routed by ∅ mass** (the null-aware policy of `uqbench.decide`): a third threshold decides, by the mass left for ∅, whether to abstain immediately or keep resampling. For CAPE, the coherent combiner, and projected DeepConf, utility differs from the confidence-only policy by within ±0.5 percentage points. Reason: what gets routed are unanswered questions, whose ∅ mass is highly correlated with the confidence of the selected answer, and resampling is cheap, so the third threshold has almost nothing to exploit.
- **Cross-model cascade routing** (analysis only, not in the paper): a small model (Qwen3-8B or R1-Distill-7B) answers, and the fraction of questions with the strongest none-correct evidence is handed to gpt-oss-20b. Routing by CAPE's π∅ is no better than routing by vote share, and routing by projected DeepConf's ∅ mass is the worst. The routing gain depends on "the large model is right and the small model is wrong", not just "the small model has no correct sample"; the detection advantage of π∅ did not translate into cascade accuracy.
- **Answer search with a verification budget** (2026-10-21, prototype, labels before the regrade, 30 pairs): candidate answers are handed to an exact verifier (gold labels) in decreasing order of the method's cluster probability, each verification costing λ, a correct answer scoring 1; the stopping rule is "the conditional probability that a correct answer remains among the remaining candidates, Σ unverified π / (Σ unverified π + π∅) < t" or "the next candidate's probability < t". (a) With the best t per pair on the test set: utilities of the 11 methods are all within ±0.005, CAPE does not dominate; with only the first rule, projected per-answer methods never stop (the residual mass of Proposition 2 is 0), and at λ = 0.5 their utility is 0.05–0.12 vs 0.25 for CAPE, but the gap disappears once they are given the second rule. (b) Untuned insertion rule (verify only if the next candidate's conditional probability ≥ λ): at λ ≥ 0.5 CAPE is best (0.2585), equal to the modular implementation (0.2579); at λ = 0.2–0.3 CAPE is below voting and CAPE w/o ∅ (0.524 vs 0.533): CAPE's π∅ is conservative and stops too early. Neither design is adopted; cost-based answering is used instead (`uqbench/studies/abstention.py`).
- **Joint model vs modular implementation** (`uqbench.studies.ablation --studies modular`): CAPE is compared with "coherent stacker + separate ∅ classifier + renormalization" on 5%–100% of the training pool. The modular implementation is better on small pools, and the two are close on the full pool (CAPE mean slightly higher, per-pair win rate about 1/4). No case was found where the joint model is reliably ahead, so the paper presents CAPE as a single-model implementation of two principles.

## 5. Attribution Results That Do Not Support Us

The numbers below come from the results when `+Coh` still had a second calibration map applied.

- **Within-cluster evidence correlation r is not a measurable mechanism**: in Propositions 3 and 4, ∅ depends on r, but per pair the "difference between the stacker's residual mass and CAPE's ∅ Brier" is unrelated to the measured within-cluster correlation (ICC 0.83–0.93) (Spearman −0.16, not significant). The paper uses r only as an example in the theoretical model and does not claim it as a mechanism in the data.
- **CAPE's Brier advantage over the coherent stacker comes entirely from calibration**: CORP decomposition ΔBrier −0.0030 = ΔMCB −0.0026 + ΔDSC +0.0004 (interval contains 0). After a 2×3 swap of the between-answer shape and the ∅ source, the gain is split roughly evenly between the two, with no clean attribution.
- **A separate ∅ classifier in-distribution is no weaker than CAPE's ∅**: on the Brier of the ∅ event, CAPE is significantly worse on 7/23 pairs of the development set and 6/14 pairs of the held-out set; CAPE is more stable only on code (not in the training pool).
- **Going below the voting bound p̂(1−p̂) on unanimous questions is not an effect of rule two**: CAPE is significantly below the bound on 8/43 pairs, the coherent stacker on 11/43, SC on 0. What works is trace evidence and rule one.
- **The ∅ Brier gain is mostly calibration**: with Platt recalibration of either read-out on half of the questions in each pair (a control biased toward the baseline), development-set 1−top answer 0.127 vs CAPE 0.124, new questions 0.130 vs 0.129; AUROC within unanimous and split questions is also close. What remains is cross-question-type ranking (development set 0.771 vs 0.793).
- **On new questions the ∅ feature learned by CAPE is worse than a constant ∅ logit**: ∅ Brier 0.148 vs 0.141 (split questions 0.180 vs 0.169). The difficulty features of the ∅ logit (deviation (ii)) do not help outside the training pool.
- **The recalibrated product DeepConf×SC is on par with projected DeepConf**: 60–74% of the projection gain is discrimination (i.e., fusion with votes). Raw projection beats it on low-accuracy pairs (10/11) and is worse on easy pairs (Spearman −0.51 with accuracy). The paper only claims that projection does not raise Brier per question, whereas the product does.

## 6. Unfavorable or Neutral Results in Robustness and Decision Analysis

From `uqbench.studies.robustness` (`results/robustness_*.csv`); the paper reports only the positive conclusions. Paired intervals resample questions, with the questions of each test set drawn jointly across the three backbones.

- **AUROC is not significant against our own controls**: the mean AUROC gain over 30 pairs is +0.0003 [−0.0008, +0.0015] against the coherent combiner and +0.0020 [−0.0001, +0.0040] against the modular implementation.
- **Brier does not dominate everywhere on OOD**: OOD mean Brier +0.0020 [−0.0009, +0.0049] against the modular implementation and −0.0010 [−0.0053, +0.0033] against projected SEP (SEP + projection has the better point estimate); OOD AUROC −0.002 against the Combiner and −0.003 against the modular implementation (both not significant).
- **Aggregation**: weighted by number of questions, ECE ties with projected ReasonProbe (0.0691); mean AUROC is highest under 13 of 15 aggregations, and the modular implementation is higher when Minerva or Qwen3-8B is dropped.
- **Decisions are on par with the coherent combiner**: the intervals of the differences from the coherent combiner in selective-answering accuracy, AURC, and 50% / 80% coverage accuracy all contain 0 (at 80% coverage the coherent combiner is 0.03 percentage points higher); with adaptive stopping, the coherent combiner is likewise 0.6 percentage points above SC@8. SC stopping early by itself uses only 3.8 samples, with accuracy 0.1 percentage points below SC@8.
- **Questions violating (A1)–(A2)**: among 215 (backbone, question) pairs, projection raises Brier on 110–183 (depending on the estimator); together this offsets only 0.6%–3.3% of the published estimators' gains (the paper reports only the latter).
- **Scorer-independent string clustering** (`--studies equivalence`, after the regrade and LLM-assisted clustering): when math answers are clustered by LaTeX-normalized strings, the fraction of math questions where the label disagrees with the cluster rises from 1.2% to 13% (gpt-oss 23%); the Brier improvement of the 7 single-answer estimators after projection retains only 52% (0.117 → 0.088; 0.061 with LLM-assisted clustering), and projection raises Brier in 25 of 105 cells (24 on gpt-oss); whitespace-only string clustering retains 21% with 32 cells raised, and the mean of the robust lower bound is negative. TriviaQA with NLI clustering retains 95% with no cell raised. The win/loss cell counts of CAPE against projected DeepConf are unchanged (AUROC 16/18).
- **SuperGPQA**: on Qwen3-8B CAPE is 3rd of 18 methods in AUROC (−0.013 against the Combiner, FDR-significant) and 3rd in Brier; on R1-Distill 2nd with no significant difference from the strongest competitor; on gpt-oss 1st on all three metrics with a significant lead. The owner decided not to remove this dataset for now (2026-10-18).

## 6b. Unfavorable or Neutral Results in the Dependence Intervention (Shared-Prefix Sampling)

From `uqbench.studies.dependence` (`results/qwen3-8b/dependence_*.csv`): Qwen3-8B, 6 test sets, K answers share the first f ∈ {0.25, 0.5, 0.75} of the reasoning of the main run's first answer (the distribution of a single answer is unchanged; only the dependence between answers changes). Intervals are 95% intervals from question resampling, with the same question drawn jointly across conditions.

- **r does not rise with dependence**: when learned together with α, r is 0.50 / 0.27 / 0.38 / 0.30 at f = 0 / 0.25 / 0.5 / 0.75 (5-fold standard deviation 0.10–0.23), trading off against α, so it cannot serve as a measure of dependence strength. When only α is learned, α falls monotonically (1.09 / 0.70 / 0.47 / 0.44).
- **Same-probe projection is equally robust as CAPE**: the AUROC difference CAPE − Probe+Coh has an interval containing 0 in every condition; the Brier difference at f = 0.75 is +0.0042 [+0.0011, +0.0072] (Probe+Coh is better) and not significant in the other conditions.
- **The ∅ calibration advantage of frozen CAPE shrinks with dependence**: Brier of the no-correct-answer event, CAPE minus projected-probe 1−top, is −0.011 / −0.004 / −0.002 [−0.005, +0.001] / +0.001 [−0.002, +0.003]. On unanimous questions frozen CAPE underestimates "all wrong" (at f = 0.75 predicted 0.151, actual 0.184; 0.186 after refitting per condition); the difference in ∅ Brier between frozen and refitted is significant only at f = 0.75 (+0.0017 [+0.0002, +0.0032]).
- **Dependence does not increase linearly with f**: most of the dependence comes in the 0 → 0.25 step (effective sample size 8 → 2–4.5), and 0.25 → 0.5 is non-monotonic in the ICC of math500 (0.785 vs 0.772) and the all-wrong rate of GPQA (0.237 vs 0.232).

## 7. Backbones Not Included in the Paper

| Backbone | Status | Reason |
|---|---|---|
| Olmo-3-7B-Think | Originally planned second reasoning backbone | The Olmo3 implementation in vLLM 0.31 ignores per-layer sliding windows (`layer_types`), so trajectories longer than 4,096 tokens are computed wrongly. Replaced by R1-Distill. The runtime patch was not verified end to end and has been moved out of the repo |
| Llama-3.1-8B-Instruct | Run in full (`reasoning_format: none`, 8k budget) | 4 of 8 test sets have more than 20% no-answer. In the remaining cells CAPE wins 11/12, but AUROC is only ≤ 0.011 above CISC/SC: single-answer evidence is weak and CAPE is close to weighted voting; answer selection is worse than majority voting on 6/8 sets. Results of the support-gate version are almost identical (the probe gives almost no gain on instruction models) |
| Ministral-8B-Instruct-2410 | Run in full | 12/27. On TriviaQA ∅ over-abstains: short factual answers shift the evidence distribution away from the math training pool, and the mean confidence on unanimous and correct questions is only 0.86 (SC 0.98), Brier 0.206 vs CISC 0.130 |

By the project owner's decision, the paper reports only the three reasoning backbones in `uqbench/configs/benchmark.yaml: backbones`; the results above are kept in a local archive outside the repo. CAPE's gain depends on informative single-answer evidence, which is most pronounced on reasoning models.

## 8. Engineering Pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| gpt-oss samples ids outside the tokenizer vocabulary at T=1 | About 0.3% of answers contain padding-row tokens, and the replay gather goes out of bounds (CUDA assert) | Take the log-prob of that token from the full logits (`replay.token_stats`) |
| gpt-oss replay | No flash-attention implementation for the attention sink; MoE is slow in HF | Eager attention + incremental forward in chunks of 2,048 tokens; replay takes about 1.5 h |
| flash-attn is compiled only for H200 (sm_90) kernels | Replay errors on A100 / A40 | Use SDPA attention for replay; for vLLM generation put the conda env `bin` on `PATH` (flashinfer compilation needs `ninja`) |
| Instruction models loop until 32k | Generation time dominated by a few long-tail answers | `max_tokens: 8192` (needed only for the `none` format) |
| Ministral special tokens encoded as plain text | Follow-up prompt misaligned | `apply_chat_template(tokenize=True)` + `encode_with_specials` |
| 20 duplicate question ids in TriviaQA validation | Shape mismatch when merging | Drop duplicate ids at load time (960 questions evaluated) |
| Adding a dataset changes job indices | Sampling seeds and shards misaligned | Fix `generation_order`; new datasets can only be appended |
| Replay npz re-reads arrays on every access | 155 GB memory at load | Read each array only once |
| ReasonProbe's grid of 216 configurations | Training one by one is too slow | Batched training (`_ProbeBank`) |
| CAPE feature cache keyed by (dataset, uid, k) | The two half-samples of a replication experiment overwrite each other | Attach the cache to the Question object |
| Probe out-of-fold cache keyed by `id(question)` | When several CAPE variants are fitted in turn in one process, the main CAPE on math sets reads stale probe logits from other instances (the gate is off on OOD sets, which are unaffected) | Cache key changed to a counter token unique to each probe, with an object-identity check; all affected results recomputed |
| Two CAPE modules (an old-version copy and the repo version) compared in one process | Both probe counter tokens start from 0, so the probe-logit caches on Question hit each other and the repo version reads the old probe's logits | Offset the copy's counter; run different implementations in separate processes when comparing them |
| Ablation variants each pick their own calibration map | On gpt-oss "trace model only" chose Platt while CAPE kept native probabilities, so with the gate off the two differ in Brier but not in AUROC | Explained in the paper; not an error |
| GitHub API rate limit | Wrongly concluded that CISC has no official code | Verify the repository via the web page instead |
| DeBERTa-large-MNLI loaded in fp16 | `expected scalar type Float but found Half` in attention | Use fp32 (the model is small, so speed is unaffected) |
| LLM grading without thinking (Haiku 5.5, first regrade) | Spot checks found clear misjudgments: applying a 1% tolerance to exact closed-form answers (two different radicals with close numeric values judged correct); treating a general formula that needs substitution as a concrete-value answer | Prompt distinguishes exact values from decimal approximations and requires the form and case specified by the question, with adaptive thinking enabled; everything regraded (first-run results kept in `OUTPUT_DIR/math_regrade/v1_no_thinking/`, unused) |
| 30-minute CPU limit on the login node | Long processes on the login node get killed | All computation (including tests) goes into Slurm jobs |
| `pkill -f <pattern>` | The pattern matches the shell executing it; vLLM's EngineCore child process lingers and holds GPU memory | Kill processes by PID, and kill EngineCore separately |
