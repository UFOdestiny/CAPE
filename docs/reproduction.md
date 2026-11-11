# Reproduction

See [protocol.md](protocol.md) for the protocol and [configuration.md](configuration.md) for paths and credentials. All commands run from the repository root;
every stage can resume from a checkpoint and can also be run on its own (`python -m uqbench --help` lists the stages, `python -m uqbench <stage> --help`
gives the arguments). Do not let two jobs write the same output shard at the same time.

## 1. Data and Weights

```bash
bash scripts/download_data.sh      # raw data → $DATASETS_DIR/processed (for GPQA, first accept the terms on HF and set HF_TOKEN)
bash scripts/download_models.sh    # three backbones, ORM reward model, NLI model → $MODELS_DIR
```

## 2. Per Backbone

```bash
for m in qwen3-8b r1-distill-qwen-7b gpt-oss-20b; do
  bash scripts/run_benchmark.sh "$m" 0,1,2,3
done
```

`run_benchmark.sh` runs, in order:

1. `run_gpu.sh`: generate → replay → energy → followup → reward;
2. `run_labels.sh`: LLM-assisted labeling (Claude Haiku 5.5 Batch API, paid, requires `ANTHROPIC_API_KEY`); judgments are cached by answer string;
3. `run_cpu.sh`: evaluate → controls → decision → within_question → refinement → attribution → audit.

## 3. Ablations, Dependence Intervention and Black-Box Backbone

```bash
bash scripts/run_ablations.sh 0,1,2,3    # qwen3-8b-x8 (responses 9-16) and K scaling; answer-set judge; NLI clustering; judge/forms/equivalence ablations
bash scripts/run_dependence.sh 0,1,2,3   # generation, replay, energy, labeling and dependence study for qwen3-8b-pf25/50/75
bash scripts/run_blackbox.sh 0,1,2,3     # Claude Haiku 5.5: Batch sampling and verbalized confidence, ORM, labeling, evaluation
```

The second argument of `run_blackbox.sh` is the Batch API budget (USD); before submitting, it checks the amount already spent and the estimated cost of the pending requests.

## 4. Summary and Paper Tables and Figures

```bash
bash scripts/run_report.sh
```

It runs, in order, `report` (`$RESULTS_DIR/summary.md`, per-test-set CSVs, per-response parquet), `robustness` and `abstention`
(which read the per-response parquet), and `visualization.build` (tables and figures are written directly into `$PAPER_DIR/tables` and `$PAPER_DIR/figs`).
To rebuild only some tables or figures: `python -m visualization.build --only <label or function name>`; see the [visualization notes](../visualization/README.md).

## 5. Slurm

After setting `SLURM_ACCOUNT` and `SLURM_PARTITION` in `.env`:

```bash
bash scripts/submit_benchmark.sh qwen3-8b
```

It submits `scripts/slurm_benchmark.sbatch` (`run_benchmark.sh` on one 4-GPU node). CPU analyses use 3 BLAS threads by default,
and results are bitwise reproducible at the same thread count.
