# Repository Architecture

Estimators, experiment execution, artifact reading and presentation are kept separate: estimators only receive the unified `Question` record and never launch inference; analyses and figures only read stored artifacts and never rerun sampling.

| Directory or module | Responsibility |
|---|---|
| `cape/model.py` | CAPE: probe, answer model (conditional logit + ∅), support gate |
| `cape/coherence.py` | coherent projection and label-free bound |
| `baselines/` | `Baseline.fit/score` interface, all published estimators and controls (`stacker.py`, `fusion.py`) |
| `uqbench/configs/` | `benchmark.yaml` (protocol, datasets, method list) and `models.yaml` (backbones, decoding, shared-prefix runs) |
| `uqbench/pipeline/` | Data preparation, sampling, replay, energy, follow-up, reward, answer-set judge, NLI clustering, evaluation |
| `uqbench/labeling/` | LLM-assisted labeling: math regrading and answer equivalence (`regrade`), short-answer extraction (`extract`) |
| `uqbench/blackbox/` | Batch API sampling (`batch`) and evaluation (`evaluate`) for the black-box backbone |
| `uqbench/studies/` | Analyses in the paper: controls, decision, within-question ranking, refinement, attribution, audit, same-probe controls, ablation, K scaling, dependence intervention, robustness, cost-based answering |
| `uqbench/reporting/` | Result reading (`results`), cross-model summary and per-response export (`report`) |
| `uqbench/records.py`, `artifacts.py` | Build the K responses per question from stored signals; reasoning-signal readers |
| `uqbench/verify.py` | Answer extraction, equivalence checking, clustering |
| `uqbench/metrics.py`, `calibration.py`, `stats.py`, `diagnostics.py` | Metrics, calibration, bootstrap, coherence diagnostics |
| `uqbench/config.py`, `io.py`, `cli.py` | `.env` and paths, parallel workers and jsonl, unified command entry point |
| `visualization/` | Paper tables (`tables/`) and figures (`figs/`), one function each; `build.py` writes directly into `PAPER_DIR` |
| `scripts/` | Download, run and Slurm entry points; dataset and backbone lists are read from YAML |

Dependency direction: `config/io/artifacts → records → cape/baselines → pipeline/labeling/studies → reporting → visualization`.
`records` loads stored signals only through readers and does not import GPU modules.

## Artifacts

All of the following locations are set by `.env`.

```text
$DATASETS_DIR/
  raw/                         raw downloads
  processed/<dataset>.jsonl    unified question records
$OUTPUT_DIR/<model>/
  gen/<dataset>.jsonl          responses, token ids, answers, completion status and labels
  replay/<dataset>.<shard>.npz token statistics and hidden states
  energy/                      raw logits of answer tokens
  followup/_shards/            P(True) and verbalized confidence
  reward/_shards/              ORM scores
  judge/, judge_think/         answer-set judge
  clusters/, clusters_nli/     answer-cluster caches (grading rules / NLI)
  scores/test/                 per-response scores
  regrade.jsonl                log of labeling changes
  dependence/                  per-question scores of the dependence intervention (ablation backbone)
$OUTPUT_DIR/math_regrade/, math_equiv/, qa_extract/   batches and caches of LLM labeling (reused by answer string)
$RESULTS_DIR/<model>/          per-backbone result tables (see protocol §9)
$RESULTS_DIR/                  summary.md, cross-backbone result tables
$PAPER_DIR/tables/, figs/      paper table panels, figures and panels.tex
```

In model names, `-x8` denotes eight additional independent samples and `-pf25/50/75` denotes the shared-prefix fraction; they are experimental conditions, not version numbers.
