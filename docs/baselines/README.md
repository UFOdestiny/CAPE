# Baseline Selection

**Rule**: the literature survey covers work from 2024 onward; baselines on the main list must be published in 2025 or later and have official open-source code. The final list is chosen by the project owner from the candidate tiers, and the owner also approves exceptions (SEP) and additional controls. The per-paper notes in this directory summarize the original papers (method, data, protocol, backbones, code check) and were written before implementation; the settings we actually use and our deviations are in the [experimental protocol](../protocol.md) §6.

The list and the table row order are in `uqbench/configs/benchmark.yaml: methods`; implementations are registered in `baselines/__init__.py: build_all`.

## Implemented

| Method | Paper | Official code (GitHub) | Implementation | Note |
|---|---|---|---|---|
| DeepConf | Fu et al., 2025, *Deep Think with Confidence* | facebookresearch/deepconf | `baselines/deepconf.py` | [DeepConf.md](deepconf.md) |
| CDG | Wang et al., ICML 2026, *Inference Time Optimization with Confidence Dynamics* | Accenture/CDG | `baselines/cdg.py` | [CDG.md](cdg.md) |
| SelfCertainty | Kang et al., NeurIPS 2025, *Scalable Best-of-N Selection via Self-Certainty* | backprop07/Self-Certainty | `baselines/selfcertainty.py` | [SelfCertainty.md](selfcertainty.md) |
| LRMVerbal | Yoon et al., 2025, *Reasoning Models Better Express Their Confidence* | MattYoon/reasoning-models-confidence | `baselines/lrmverbal.py` | [LRMVerbal.md](lrmverbal.md) |
| CISC | Taubenfeld et al., Findings ACL 2025, *Confidence Improves Self-Consistency in LLMs* | taubenfeld/CISC | `baselines/cisc.py` | [CISC.md](cisc.md) |
| ReasonProbe | Zhang et al., 2025, *Reasoning Models Know When They're Right* | AngelaZZZ-611/reasoning_models_probing | `baselines/reasonprobe.py` | [ReasonProbe.md](reasonprobe.md) |
| SNNE | Nguyen et al., Findings ACL 2025, *Beyond Semantic Entropy* | BigML-CS-UCLA/SNNE | `baselines/snne.py` | [SNNE.md](snne.md) |
| DeepConf-Vote | Fu et al., 2025 (offline weighted voting of DeepConf) | facebookresearch/deepconf | `baselines/consistency.py` | [DeepConf.md](deepconf.md) |
| ConsistCal | Lyu et al., AAAI 2025, *Calibrating Large Language Models with Sample Consistency* | not checked | `baselines/consistency.py` | [ConsistCal.md](consistcal.md) |
| SEP | Kossen et al., 2024, *Semantic Entropy Probes* (year exception; supervised uncertainty probe) | not checked | `baselines/sep.py` | [SEP.md](sep.md) |
| ORM | Liu et al., 2025, *Skywork-Reward-V2* (general-purpose outcome reward model) | SkyworkAI/Skywork-Reward-V2 | `uqbench/pipeline/reward.py`, `baselines/orm.py` | [ORM.md](orm.md) |
| SemEnergy | Ma et al., 2025, *Semantic Energy* (cluster energy, raw logits) | not checked | `uqbench/pipeline/energy.py`, `baselines/semantic_energy.py` | [SemEnergy.md](semantic_energy.md) |
| CoCoA | Vashurin et al., 2025, *Uncertainty Quantification for LLMs through Minimum Bayes Risk* | not checked | `baselines/fusion.py` | [CoCoA.md](fusion.md) |
| SC†, SE† | Wang et al., 2023; Farquhar et al., Nature 2024 | — | `baselines/reference.py` | [SE.md](semantic_entropy.md) |

† Same-budget references (the same batch of K responses); not counted as 2025+ baselines. Under exact answer equivalence, SE, KLE, EigV and the 0/1 version of SNNE all reduce to functions of the answer-cluster sizes, so only the discrete SE row is reported.

**Controls we constructed** (`uqbench/configs/benchmark.yaml: methods.constructed`; not counted as published methods): DeepConf×SC (the calibrated DeepConf probability times the vote share, `baselines/fusion.py`) and Combiner (per-response logistic regression on 22 trace statistics + 5 voting features, `baselines/stacker.py`, with `baselines/fusion.py` providing the main-table interface); the projected, probe and support-gate versions of the Combiner are in `uqbench/studies/controls.py`.

**Closest precedents to CAPE**: Semantic Energy (Ma et al., 2025; implemented, [SemEnergy.md](semantic_energy.md)) targets the same consistent errors, using per-response logit energy. Clustered Self-Assessment (Cao et al., Findings ACL 2026; [CSA.md](csa.md)) uses answer clusters as options and adds "None of the above", without training; it is not implemented as a baseline, and its closest form appears in the answer-set judge ablation (`uqbench/pipeline/judge.py`, see the [method note](../method.md) §5–6).

## Considered but not implemented

| Category | Methods | Reason not implemented |
|---|---|---|
| No official code | Commit, ConfTime, TraceLength, TracingUQ, SCVC (Del et al.), Mei et al., Probe-SD, J4U | Does not meet the implementation criteria |
| Before 2025 | SE (as reference), EigV, KLE, INSIDE, SupUQ, VerbConf, SAR, Semantic Density | Literature references only (SEP is implemented as an exception, see above) |
| Year boundary | LLMsKnow (arXiv 2024 / ICLR 2025), LM-Polygraph (framework, TACL 2025) | Overlaps with ReasonProbe; LM-Polygraph targets short answers |
| Needs all-layer attention, or long sequences beyond training length | Gnosis, UQAC, RAUQ, UQHead | Memory or out-of-distribution problems on 32k reasoning trajectories; some have no long-CoT results |
| Prefix probing / early stopping | AnswerConv, ConfLeaps, PrematureConf | The original papers are in early-stopping or multiple-choice settings, with no correctness-prediction evaluation |
| Extra model or external verifier | RACE, GCM, SelfDoubt, CoTUQ, RMCB series | Need an extra model, out-of-distribution training data, or have incomplete implementations |
| Training the LLM itself | RLCR, Self-Calibration, CalibSFT, TokUR | Changes the model under test; not a post-hoc estimator |
