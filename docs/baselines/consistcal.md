# ConsistCal: Calibrating Large Language Models with Sample Consistency

> This note records our reading of the original paper and its official implementation; the adaptation and settings currently used in this repository are in the [experimental protocol](../protocol.md).
- Paper: Lyu et al., AAAI 2025 (39(18): 19260–19268); arXiv 2402.13904.
- Category: multi-sample consistency (training-free); the paper targets the confidence of the majority answer.
- Our adaptation (`baselines/consistency.py`): the paper's three consistency measures are turned into per-response scores. Agreement is the vote share of the response's answer; first-second distance is that vote share minus the largest vote share among the other answers; entropy is the negative normalized entropy of the answer distribution (question level). The measure is selected by val AUROC, and a Platt mapping fitted on val then gives the probability. Responses with no answer get the lowest score.
- Reason for inclusion: a consistency-calibration baseline, added with the owner's approval.
