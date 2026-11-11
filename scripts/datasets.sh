# Sourced by the run scripts: dataset and backbone lists from configs/benchmark.yaml.
#   sets        generated datasets of the test sets and training pools (test sets in `test_union` expand to parts)
#   backbones   the reported backbones
source scripts/environment.sh
sets() { "$PY" -c "from uqbench.config import generated, load_yaml as y; b = y('benchmark'); \
print(' '.join(generated(b['test_sets'] + b['training_pools'])))"; }
backbones() { "$PY" -c "from uqbench.config import load_yaml; print(' '.join(load_yaml('benchmark')['backbones']))"; }
