#!/usr/bin/env bash
# Submit one benchmark job using private cluster settings from .env.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
MODEL="${1:?usage: submit_benchmark.sh <model>}"
sbatch --account="${SLURM_ACCOUNT:?set SLURM_ACCOUNT in .env}" \
  --partition="${SLURM_PARTITION:?set SLURM_PARTITION in .env}" \
  --export="ALL,MODEL=$MODEL" scripts/slurm_benchmark.sbatch
