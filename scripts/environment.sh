# Sourced by the run scripts after changing to the repository root: exports the paths and settings resolved by
# uqbench.config (process environment > .env > defaults) and sets $PY.
_bootstrap="${PYTHON:-python}"
eval "$("$_bootstrap" -m uqbench.config --shell)"
PY="${PYTHON:-$_bootstrap}"
unset _bootstrap
