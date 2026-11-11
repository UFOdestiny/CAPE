#!/usr/bin/env bash
# Download model weights into $MODELS_DIR: the backbones (default: the reported backbones in
# uqbench/configs/benchmark.yaml), the outcome reward model (ORM baseline) and the NLI model (equivalence ablation).
# Usage: bash scripts/download_models.sh [qwen3-8b r1-distill-qwen-7b ...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
MODELS="${MODELS_DIR:-models}"
names=("$@")
[ ${#names[@]} -eq 0 ] && mapfile -t names < <("$PY" -c "from uqbench.config import load_yaml; print('\n'.join(load_yaml('benchmark')['backbones']))")
for name in "${names[@]}"; do
  read -r hf_id path < <("$PY" -c "from uqbench.config import model_config as m; import sys; c=m(sys.argv[1]); print(c['hf_id'], c['path'])" "$name")
  hf download "$hf_id" --local-dir "$MODELS/$path" > /dev/null
  echo "downloaded $hf_id -> $MODELS/$path"
done
hf download Skywork/Skywork-Reward-V2-Llama-3.1-8B --local-dir "$MODELS/Skywork-Reward-V2-Llama-3.1-8B" > /dev/null \
  && echo "downloaded the reward model"
hf download microsoft/deberta-large-mnli --local-dir "$MODELS/deberta-large-mnli" > /dev/null && echo "downloaded NLI model"
