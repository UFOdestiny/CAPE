#!/usr/bin/env bash
# LLM-assisted labels for generated runs (paid Anthropic Message Batches; needs ANTHROPIC_API_KEY): math answers
# the rules reject are graded against the reference, math answer pairs are judged for equivalence (clusters), and
# short-answer responses without a boxed answer get an extracted answer. Verdicts are cached per answer string.
# Usage: bash scripts/run_labels.sh <model> [<model> ...]
#   Runs whose responses are pooled (e.g. qwen3-8b and qwen3-8b-x8) are passed together.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
[ $# -ge 1 ] || { echo "usage: run_labels.sh <model> [<model> ...]" >&2; exit 1; }
R="$PY -m uqbench.labeling.regrade"
E="$PY -m uqbench.labeling.extract"
$R rules --models "$@"
$R submit --models "$@" && $R status --wait --interval 60
$R apply --models "$@"
group=$(IFS=,; echo "$*")
$R pairs-submit --groups "$group" && $R status --wait --interval 60
$R pairs-apply --models "$@"
$E submit --models "$@" && $E status --wait --interval 60
$E apply --models "$@"
