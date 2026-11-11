#!/usr/bin/env bash
# CPU stages for one backbone, after run_gpu.sh and run_labels.sh: fit every method on the training pools and score
# the test sets (evaluate), then the feature-matched controls, consistent errors and the projection (decision),
# within-question ranking, refinement and "none correct", attribution and the verifier audit.
# Usage: bash scripts/run_cpu.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
MODEL="${1:?usage: run_cpu.sh <model>}"
# The PCA of float32 hidden states (CAPE's probe, SEP) depends on the BLAS thread count in the last digits; the
# reported results use 3 threads (bit-identical reruns at the same setting).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
"$PY" -m uqbench.pipeline.evaluate --model "$MODEL"
for study in controls decision within_question refinement attribution audit; do   # controls first
  "$PY" -m "uqbench.studies.$study" --model "$MODEL"
done
