# SEP: Semantic Entropy Probes

> This note records our reading of the original paper and its official implementation; the adaptation and settings currently used in this repository are in the [experimental protocol](../protocol.md).
- Paper: Kossen et al., 2024, "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs".
- Category: single-trajectory hidden-state probe; trained on semantic entropy rather than correctness labels.
- Our implementation (`baselines/sep.py`):
  - Features: hidden states at the answer position (middle layer + last layer), standardized, then PCA-256.
  - Target: whether the discrete semantic entropy over the K=4 answer clusters of each training-pool question is above the median over training questions. The median is usually 0, i.e. the target is "whether the answers disagree".
  - Model: logistic regression, with C selected by val correctness AUROC.
  - Score: the negative "high entropy" logit. Only one response is needed at test time.
- Reason for inclusion: a supervised uncertainty probe, added as a year exception with the owner's approval.
