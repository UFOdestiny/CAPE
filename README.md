# CAPE: Coherent Confidence for LLM Reasoning

Code for *Right Together, Wrong Together: Coherent Confidence and the "None Correct" Outcome for LLM Reasoning*.

Sampled responses with equivalent answers are right or wrong together, and all sampled answers can be wrong at once.
CAPE (Coherent Answer Posterior Estimation) estimates the joint outcome directly: a conditional logit over the
distinct answer clusters of a question and a "none correct" outcome, scored from vote shares and a hidden-state probe
of the responses. Its forecasts are coherent by construction. The repository also provides the projection onto the
coherent set, a training-free correction for any per-response confidence that cannot increase the Brier score when
the answer constraints hold.

## Installation

Python 3.11 or newer; run everything from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The GPU stages need a CUDA build of PyTorch and vLLM that matches your hardware; `flash-attn` is optional.

## Configuration

Paths, caches and credentials come from `.env` (process environment > `.env` > defaults; relative paths are anchored to
the repository root). See [`.env.example`](.env.example) for every variable:

| Variable | Default | Content |
|---|---|---|
| `MODELS_DIR` | `models` | Backbone weights |
| `DATASETS_DIR` | `datasets` | Raw downloads and processed datasets |
| `OUTPUT_DIR` | `outputs` | Per-model generations, replay signals, labels and scores |
| `RESULTS_DIR` | `results` | Metric tables |
| `HF_TOKEN` | unset | Gated datasets (GPQA) |
| `ANTHROPIC_API_KEY` | unset | LLM-assisted labels and the black-box backbone |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | unset | Slurm submission |

Protocol settings (test sets, training pools, K, backbones) are in `configs/benchmark.yaml`; decoding settings are in
`configs/models.yaml`.

## Data and models

```bash
bash scripts/download_data.sh      # raw datasets -> $DATASETS_DIR/processed (accept the GPQA terms and set HF_TOKEN first)
bash scripts/download_models.sh    # backbones -> $MODELS_DIR
```

## Running the pipeline

```bash
bash scripts/run_benchmark.sh qwen3-8b 0,1,2,3
```

runs, for one backbone and the given GPUs:

1. `scripts/run_gpu.sh`: sampling (vLLM) and teacher-forced replay (token statistics and hidden states);
2. `scripts/run_labels.sh`: LLM-assisted labels and answer-equivalence clusters (paid Anthropic Message Batches);
3. `scripts/run_cpu.sh`: fits CAPE on the training pools, scores the test sets and writes
   `$RESULTS_DIR/<model>/metrics.csv` (AUROC, AURC, Brier, ECE, NLL with bootstrap intervals).

Every stage is resumable and can be run on its own: `python -m uqbench --help` lists the stages and
`python -m uqbench <stage> --help` their options. `bash scripts/submit_benchmark.sh <model>` submits the same run as a
Slurm job (account and partition from `.env`). `bash scripts/run_blackbox.sh` runs CAPE on an API-only backbone
(Claude Haiku 5.5 through Message Batches), where the evidence is the vote prior and the verbal confidence.

The evaluation also scores the baselines of the paper when a `baselines` package exposing `build_all()` is importable;
the baselines are not part of this repository, and without them CAPE is evaluated alone.

## Using CAPE and the projection

```python
import numpy as np
from cape import project_coherent

p = np.array([0.9, 0.6, 0.7, 0.4])          # calibrated per-response confidences
clusters = [0, 0, 1, 2]                      # answer-equivalence clusters
has_answer = np.array([True, True, True, False])
q = project_coherent(p, clusters, has_answer)   # [0.6, 0.6, 0.4, 0.0]; 1 - (0.6 + 0.4) is left for "none correct"
```

`CAPE().fit(train, val)` learns from labeled `uqbench.records.Question` records (`uqbench.records.load_pool`) and
`CAPE().score(question)` returns one probability per response.

## Repository layout

```text
cape/              CAPE (model.py), the projection onto the coherent set (projection.py), trace statistics (features.py)
uqbench/           Pipeline that runs and evaluates CAPE
  pipeline/        data, generate, replay, evaluate
  labeling/        LLM-assisted labels: math regrading, answer equivalence, short-answer extraction
  blackbox/        API-only backbone through the Anthropic Message Batches API
  records.py       Question records (K responses with their signals, clusters and labels)
  estimator.py     fit/score interface of the estimators
  metrics.py, calibration.py, diagnostics.py, verify.py, config.py
configs/           benchmark.yaml (protocol) and models.yaml (backbones, decoding)
scripts/           Download, run and Slurm entry points
```

## License

Code is released under the [MIT license](LICENSE). Datasets and model weights keep their upstream licenses.
