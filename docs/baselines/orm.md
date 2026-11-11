# ORM: Outcome Reward Model (Skywork-Reward-V2)

> This note records our reading of the original paper and its official implementation; the adaptation and settings currently used in this repository are in the [experimental protocol](../protocol.md).
- Paper: Liu et al., 2025, "Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy" (arXiv:2507.01352); open weights `Skywork/Skywork-Reward-V2-Llama-3.1-8B`.
- Category: single-response, externally trained general-purpose outcome reward model (trained verifier); not trained on our pool.
- Our implementation (`uqbench/pipeline/reward.py`, `baselines/orm.py`):
  - Input: the user prompt the backbone actually saw (`uqbench.pipeline.data.user_prompt`) and the visible part of the response (the text after the reasoning; when the reasoning is truncated, the tail of the whole trajectory), concatenated with the reward model's chat template; responses longer than 4096 tokens keep their tail (where the final answer is).
  - Score: the reward model's scalar output; as for the other estimators, a calibration mapping is selected on the validation pool and the output is projected onto the coherent set.
- Reason for inclusion: a trained-verifier baseline; the owner chose a general ORM (no math-specific PRM and no P(IK) probe).
