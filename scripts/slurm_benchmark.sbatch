#!/usr/bin/env bash
# Slurm template (one 4-GPU node). Account and partition are provided by the submit wrapper.
#   bash scripts/submit_benchmark.sh qwen3-8b
#SBATCH --job-name=uqbench
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_benchmark.sh "${MODEL:?set MODEL}" 0,1,2,3
