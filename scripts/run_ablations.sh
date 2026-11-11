#!/usr/bin/env bash
# K = 16 study and ablations (after scripts/run_benchmark.sh for every reported backbone).
# Usage: bash scripts/run_ablations.sh [gpus=0,1,2,3]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/datasets.sh
GPUS="${1:-0,1,2,3}"
ABL="$("$PY" -c "from uqbench.config import load_yaml; print(load_yaml('benchmark')['ablation_backbone'])")"

# K = 16 study on the ablation backbone: a second sampling run supplies responses 9-16 of its test sets.
KSETS=(math500 olympiadbench minerva aime gpqa)
for s in generate replay followup reward; do
  "$PY" -m uqbench.pipeline.$s --model "$ABL-x8" --gpus "$GPUS" --datasets "${KSETS[@]}"
done
bash scripts/run_labels.sh "$ABL" "$ABL-x8"   # pooled responses: answer equivalence judged jointly
"$PY" -m uqbench.studies.kscale --model "$ABL" --extra "$ABL-x8"

# Answer-set judge on every reported backbone (GPU; the first sample of each test set and both pools) and NLI
# clusters of the short answers (GPU), then the CPU ablations on every backbone: the judge, other answer-model forms
# and verifier-independent clusters.
FIRST=(math500 olympiadbench minerva aime gpqa mmlupro supergpqa triviaqa)
for MODEL in $(backbones); do
  "$PY" -m uqbench.pipeline.judge --model "$MODEL" --gpus "$GPUS" --datasets "${FIRST[@]}" math_pool mixed_pool
  "$PY" -m uqbench.pipeline.judge --model "$MODEL" --gpus "$GPUS" --think-tokens 1024 --datasets "${FIRST[@]}" math_pool mixed_pool
done
read -r -a MODELS <<< "$(backbones)"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PY" -m uqbench.pipeline.nli --models "${MODELS[@]}"
for MODEL in "${MODELS[@]}"; do
  "$PY" -m uqbench.studies.ablation --model "$MODEL" --studies judge forms equivalence
done
