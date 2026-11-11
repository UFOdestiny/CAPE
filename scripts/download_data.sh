#!/usr/bin/env bash
# Download the raw benchmark datasets into $DATASETS_DIR/raw and build the processed jsonl files.
# GPQA is gated: accept its terms on HuggingFace and export HF_TOKEN first.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
RAW="${DATASETS_DIR:-datasets}/raw"
mkdir -p "$RAW"
fetch() { hf download --repo-type dataset "$1" --local-dir "$RAW/${1//\//_}" "${@:2}" > /dev/null && echo "downloaded $1"; }
for repo in HuggingFaceH4/MATH-500 math-ai/olympiadbench math-ai/minervamath math-ai/aime24 math-ai/aime25 \
            MathArena/aime_2026 Idavidrein/gpqa TIGER-Lab/MMLU-Pro m-a-p/SuperGPQA EleutherAI/hendrycks_math; do
  fetch "$repo"
done
fetch mandarjoshi/trivia_qa --include "rc.nocontext/validation-*"
"$PY" -m uqbench.pipeline.data
