# CAPE: Coherent Confidence for LLM Reasoning

Reference implementation and benchmark for *Right Together, Wrong Together: Coherent Confidence and the “None
Correct” Outcome for LLM Reasoning*.

Responses with equivalent answers are right or wrong together, and all sampled answers can be wrong at once. This
repository provides

- **coherent projection**: a training-free correction that maps any per-response confidence onto the coherent set
  and cannot increase the per-question Brier score when the answer constraints hold;
- **CAPE** (Coherent Answer Posterior Estimation): a conditional logit over the distinct answers and a “none
  correct” outcome, learned from vote shares and pooled response evidence;
- **a benchmark** that scores 17 estimators on the same sampled responses of three reasoning models and ten test
  sets, with LLM-assisted labels, decision analyses and a dependence intervention.

## Installation

Python 3.11 or newer. Run everything from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paths, caches, credentials
```

GPU stages need a CUDA build of PyTorch and vLLM matching your hardware; `flash-attn` is optional. Paper figures need
a TeX installation with Times fonts.

## Quick start

```python
import numpy as np
from cape import project_coherent

p = np.array([0.9, 0.6, 0.7, 0.4])          # calibrated per-response confidences
clusters = [0, 0, 1, 2]                      # answer-equivalence clusters
has_answer = np.array([True, True, True, False])
q = project_coherent(p, clusters, has_answer)   # [0.6, 0.6, 0.4, 0.0]; 1 - (0.6 + 0.4) is left for "none correct"
```

`CAPE().fit(train, val)` learns from labeled `uqbench.records.Question` records and `score(question)` returns one
probability per response; see [docs/method.md](docs/method.md).

## Reproducing the paper

```bash
bash scripts/download_data.sh                 # raw data and processed datasets
bash scripts/download_models.sh               # backbones

for m in qwen3-8b r1-distill-qwen-7b gpt-oss-20b; do
  bash scripts/run_benchmark.sh "$m" 0,1,2,3  # GPU stages, LLM-assisted labels, CPU analyses
done
bash scripts/run_ablations.sh 0,1,2,3         # K = 16 study and answer-set judge
bash scripts/run_dependence.sh 0,1,2,3        # shared-prefix dependence intervention
bash scripts/run_report.sh                    # summary, robustness, cost-sensitive answering, paper tables/figures
```

Every stage is resumable and can be run on its own: `python -m uqbench --help` lists the stages and
`python -m uqbench <stage> --help` their options. Labels use paid Anthropic Message Batches (`ANTHROPIC_API_KEY`).
[docs/reproduction.md](docs/reproduction.md) also covers the black-box backbone and Slurm submission.

## Repository layout

```text
cape/                  CAPE (model.py) and the coherent projection (coherence.py)
baselines/             Published estimators and controls behind one fit/score interface
uqbench/               Benchmark
  configs/             benchmark.yaml (protocol, datasets, methods) and models.yaml (backbones, decoding)
  pipeline/            data, generate, replay, energy, followup, reward, judge, nli, evaluate
  labeling/            LLM-assisted labels: math regrading, answer equivalence, short-answer extraction
  blackbox/            Black-box backbone via the Anthropic Message Batches API
  studies/             Analyses behind the paper (controls, decision, dependence, robustness, ...)
  reporting/           Result readers and the cross-model summary
  config.py            Paths and settings from .env
  records.py           Question records shared by every estimator
visualization/         Paper tables (tables/) and figures (figs/), one function each; build.py writes them
scripts/               Download, run and Slurm entry points
docs/                  Method, protocol, reproduction, configuration and design notes (Chinese)
```

## Configuration

All paths and credentials come from `.env` (process environment > `.env` > defaults; relative paths are anchored to
the repository root). See [docs/configuration.md](docs/configuration.md).

| Variable | Default | Content |
|---|---|---|
| `MODELS_DIR` | `models` | Backbone weights |
| `DATASETS_DIR` | `datasets` | Raw downloads and processed datasets |
| `OUTPUT_DIR` | `outputs` | Per-model generations, signals, labels and scores |
| `RESULTS_DIR` | `results` | Result tables and the summary |
| `PAPER_DIR` | `latex` | Paper project receiving `tables/` and `figs/` |
| `HF_TOKEN`, `ANTHROPIC_API_KEY` | unset | Gated datasets; LLM-assisted labels and the black-box backbone |

## Documentation

| Document | Content |
|---|---|
| [docs/method.md](docs/method.md) | CAPE, projection, assumptions and theory |
| [docs/protocol.md](docs/protocol.md) | Labels, data, backbones, pipeline, methods, metrics |
| [docs/reproduction.md](docs/reproduction.md) | Commands in order |
| [docs/configuration.md](docs/configuration.md) | Environment and dependencies |
| [docs/architecture.md](docs/architecture.md) | Modules and artifact layout |
| [docs/baselines/](docs/baselines/README.md) | Estimator selection and implementation notes |
| [docs/failures.md](docs/failures.md) | Designs that were tried and not adopted |
| [visualization/README.md](visualization/README.md) | Table and figure conventions |

## License

Code is released under the [MIT license](LICENSE). Datasets and model weights keep their upstream licenses.
