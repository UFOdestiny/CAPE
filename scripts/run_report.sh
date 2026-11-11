#!/usr/bin/env bash
# Cross-model results after every backbone's run_benchmark.sh: the summary and per-response exports (report), then
# the analyses that read those exports (robustness, cost-sensitive answering), then the paper's tables and figures.
# Usage: bash scripts/run_report.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
"$PY" -m uqbench.reporting.report
"$PY" -m uqbench.studies.robustness
"$PY" -m uqbench.studies.abstention
"$PY" -m visualization.build
