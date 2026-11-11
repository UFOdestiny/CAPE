# Configuration and Environment

## Dependencies

Python ≥ 3.11, run from the repository root (`python -m uqbench ...`, `bash scripts/...`); no package installation is needed.

```bash
pip install -r requirements.txt
```

`requirements.txt` lists all dependencies: core (data, labels, estimators, evaluation), GPU stages (vLLM, transformers), the Anthropic
client and plotting. The GPU stages need a CUDA build of PyTorch and vLLM matching the device; flash-attn is optional, and when only the kernels for one architecture were compiled,
replay on other devices should use SDPA attention (`replay.attn_implementation` in `models.yaml`). Plotting needs a
TeX environment with Times fonts.

## Environment Variables

Copy `.env.example` to `.env`. Priority: process environment → `.env` → defaults; relative paths are resolved against the repository root. Shell scripts call the same Python loader (`uqbench/config.py`) via
`scripts/environment.sh` and never execute shell code in `.env`; `export`, quotes, trailing comments and `${VAR}` are supported, and single quotes are literal. `UQ_ENV_FILE` in the process environment can point to another file, and `PROJECT_ROOT`
changes the base for relative paths.

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `models` | Model weights |
| `DATASETS_DIR` | `datasets` | Raw downloads and processed datasets |
| `OUTPUT_DIR` | `outputs` | Per-model generations, signals, labels and scores |
| `RESULTS_DIR` | `results` | Result tables and summaries |
| `PAPER_DIR` | `latex` | Paper directory that receives the generated `tables/` and `figs/` |
| `PREVIEW_DIR` | `artifacts/previews` | PNG previews of figures |
| `BUILD_DIR` | `artifacts/build` | TikZ compilation files of the framework figure |
| `CONFIG_DIR` | `uqbench/configs` | Alternative `benchmark.yaml` / `models.yaml` |
| `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE` | upstream defaults | Hugging Face caches |
| `HF_TOKEN` | unset | Gated data (GPQA) |
| `ANTHROPIC_API_KEY` | unset | LLM-assisted labeling and black-box backbone (paid) |
| `PYTHON` | `python` | Interpreter used by the scripts |
| `OMP_NUM_THREADS` and other BLAS thread counts | 3 | Setting used for the reported results; bitwise reproducible at the same thread count |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | unset | Cluster parameters for `scripts/submit_benchmark.sh` |

Credentials belong only in the local `.env` or the process environment, never in YAML, logs or commits. Paths are fixed once the module is loaded; after changing a variable, start a new process.
