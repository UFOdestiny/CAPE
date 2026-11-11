#!/usr/bin/env bash
# Dependence intervention on the ablation backbone (after run_benchmark.sh for it): the K responses of every question
# continue the first main-run response after f = 0.25 / 0.5 / 0.75 of the median trace length (models.yaml:
# <backbone>-pf25/50/75), then labels, replay, semantic-energy logits and the study.
# Usage: bash scripts/run_dependence.sh [gpus=0,1,2,3]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
GPUS="${1:-0,1,2,3}"
BASE="$("$PY" -c "from uqbench.config import load_yaml; print(load_yaml('benchmark')['ablation_backbone'])")"
SETS=(math500 olympiadbench minerva aime gpqa triviaqa)
for f in 25 50 75; do
  for stage in generate replay energy; do
    "$PY" -m uqbench.pipeline.$stage --model "$BASE-pf$f" --gpus "$GPUS" --datasets "${SETS[@]}"
  done
  bash scripts/run_labels.sh "$BASE-pf$f"
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
"$PY" -m uqbench.studies.dependence --base "$BASE" --stages structure methods refit bootstrap
