#!/usr/bin/env bash
# CPU stage for one backbone, after run_gpu.sh and run_labels.sh: fit CAPE (and the optional baselines) on the
# training pools, score the test sets and write the metrics (evaluate).
# Usage: bash scripts/run_cpu.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
MODEL="${1:?usage: run_cpu.sh <model>}"
# The PCA of float32 hidden states (CAPE's probe) depends on the BLAS thread count in the last digits; the
# reported results use 3 threads (bit-identical reruns at the same setting).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
"$PY" -m uqbench.pipeline.evaluate --model "$MODEL"
