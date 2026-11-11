# CoCoA: Confidence x Consistency

> This note records our reading of the original paper and its official implementation; the adaptation and settings currently used in this repository are in the [experimental protocol](../protocol.md).
- Paper: Vashurin, Goloburda, Ilina, Rubashevskii, Nakov, Shelmanov, Panov, 2025, "Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency" (arXiv:2502.04964).
- Official code: not checked (the authors' team maintains LM-Polygraph).
- Category: multi-response, training-free; combines single-response confidence and cross-sample consistency in one score.
- Method: from minimum Bayes risk, the uncertainty of response y = u(y) x its expected dissimilarity to the other sampled responses mean_j (1 − s(y, y_j)); u is a confidence-based uncertainty (e.g. perplexity, maximum sequence probability) and s is the similarity between responses.
- Our implementation (`baselines/fusion.py: CoCoA`):
  - Per-response scoring: u is the response's mean token negative log-prob (the paper's perplexity variant, from `uqbench.pipeline.replay`); the dissimilarity is averaged over the other K − 1 responses to the same question.
  - s is the ROUGE-L of the final answers after the reasoning block, or answer equivalence (0/1), chosen by AUROC on the validation pool.
  - Score = −uncertainty; as for the other estimators, a calibration mapping is selected on the validation pool and the output is projected onto the coherent set.
- Reason for inclusion: a published method that fuses confidence and consistency, as a control beyond simple fusions such as DeepConf×SC.
