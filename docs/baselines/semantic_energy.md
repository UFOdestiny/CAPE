# SemEnergy: Semantic Energy

> This note records our reading of the original paper and its official implementation; the adaptation and settings currently used in this repository are in the [experimental protocol](../protocol.md).
- Paper: Ma et al., 2025, "Semantic Energy: Detecting LLM Hallucination Beyond Entropy" (arXiv:2508.14496).
- Category: multi-response, training-free. Same motivation as ours (when synonymous wrong answers agree, semantic entropy is 0 and cannot tell them apart), but solved with per-response logit energy.
- Method: the token energy is the negative raw logit (before softmax) of the sampled token; the response energy is the mean over the tokens of the final-answer part (the reasoning part is discarded, as in the paper's think mode); the cluster energy is the sum of the response energies within the cluster, E(C); the confidence of a response is −E(C), and lower energy is more trustworthy.
- Our implementation (`uqbench/pipeline/energy.py`, `baselines/semantic_energy.py`):
  - One teacher-forced forward pass per response; the raw logit is the dot product of the last-layer hidden state and that token's output embedding, without projecting onto the whole vocabulary; the answer part is from after `</think>` (for Harmony, the final-channel marker) to the last non-EOS token, and the whole response when there is no marker.
  - The cluster energy is divided by K (i.e. mean energy x vote share): the ranking is unchanged for the same K; the pool has K=4 and the test has K=8, and without the division test scores are about twice the pool's, so a calibration mapping fitted on the pool would be systematically overconfident.
  - Responses with no answer each form their own cluster; empty responses contribute no evidence. As for the other estimators, a calibration mapping is selected on the validation pool and the output is projected onto the coherent set.
- Reason for inclusion: the closest existing work to ours (chosen by the owner); used to test Proposition 7 (per-response prediction misses ∅): the cluster energy is a sum of per-response evidence, and after projection only the residual mass is left for ∅.
