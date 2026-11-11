"""Shared environment, paths and benchmark settings.

Precedence: process environment > project .env > defaults. Relative project paths
are anchored to PROJECT_ROOT; CONFIG_DIR defaults to the repository's configs/.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = PACKAGE_DIR.parent   # repository root
REPO_ROOT = Path(os.environ.get('PROJECT_ROOT', _DEFAULT_ROOT)).expanduser().resolve()
_VAR = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read dotenv assignments without executing shell commands.

    Accept export prefixes, quotes, inline comments and ${VAR} expansion.
    Single-quoted values are literal; process variables override file variables.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        line = line.removeprefix('export ').strip()
        if '=' not in line:
            raise ValueError(f'{path.name}:{number}: expected KEY=VALUE')
        key, value = line.split('=', 1)
        key, value = key.strip(), value.strip()
        if not _KEY.fullmatch(key):
            raise ValueError(f'{path.name}:{number}: invalid environment variable name')
        literal = value.startswith("'")
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith('#')):
                raise ValueError(f'{path.name}:{number}: malformed quoted value')
            value = value[1:end]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
        if not literal:
            value = _VAR.sub(lambda m: os.environ.get(m[1], values.get(m[1], '')), value)
        values[key] = value
    return values


def load_env(path: Path | None = None) -> None:
    """Load UQ_ENV_FILE or the project .env without replacing exported values."""
    if path is None:
        path = Path(os.environ.get('UQ_ENV_FILE', REPO_ROOT / '.env')).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    for key, value in _parse_env_file(path).items():
        os.environ.setdefault(key, value)


load_env()
# PROJECT_ROOT may also be supplied by the initially selected dotenv file.
REPO_ROOT = Path(os.environ.get('PROJECT_ROOT', REPO_ROOT)).expanduser()
if not REPO_ROOT.is_absolute():
    REPO_ROOT = _DEFAULT_ROOT / REPO_ROOT
REPO_ROOT = REPO_ROOT.resolve()


def env_path(key: str, default: Path) -> Path:
    """Path from the environment variable ``key`` (relative paths are anchored to the repository root)."""
    value = os.environ.get(key, '')
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else REPO_ROOT / path


MODELS_DIR = env_path('MODELS_DIR', REPO_ROOT / 'models')
DATASETS_DIR = env_path('DATASETS_DIR', REPO_ROOT / 'datasets')
OUTPUT_DIR = env_path('OUTPUT_DIR', REPO_ROOT / 'outputs')
RESULTS_DIR = env_path('RESULTS_DIR', REPO_ROOT / 'results')
CONFIG_DIR = env_path('CONFIG_DIR', REPO_ROOT / 'configs')

# HF resolves its own variables; normalize local cache paths before any HF import.
for _key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE'):
    if os.environ.get(_key):
        os.environ[_key] = str(env_path(_key, REPO_ROOT / '.cache' / 'huggingface'))


def model_path(name: str) -> str:
    """Resolve a local backbone, otherwise retain the Hugging Face model id."""
    local = MODELS_DIR / name.split('/')[-1]
    return str(local) if local.exists() else name


def load_yaml(name: str) -> dict:
    """Load a registry from ``CONFIG_DIR``."""
    import yaml

    with open(CONFIG_DIR / f'{name}.yaml', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def set_parts(name: str) -> list[str]:
    """Generated datasets making up a logical test set."""
    return list(load_yaml('benchmark').get('test_union', {}).get(name, [name]))


def generated(names: list[str]) -> list[str]:
    """Deduplicate generated datasets in protocol order."""
    return list(dict.fromkeys(p for n in names for p in set_parts(n)))


def scores_dir(model: str) -> Path:
    """Per-response scores of the test sets."""
    return run_dir(model) / 'scores' / 'test'


def model_config(name: str) -> dict:
    models = load_yaml('models')
    if name not in models:
        raise KeyError(f'unknown model {name!r}; choose from {sorted(models)}')
    return models[name]


def run_dir(model: str) -> Path:
    """Artifacts for a model or named experiment condition."""
    return OUTPUT_DIR / model


def main() -> None:
    parser = argparse.ArgumentParser(description='Export the shared environment for project shell scripts.')
    parser.add_argument('--shell', action='store_true', required=True)
    parser.parse_args()
    # This output is evaluated by scripts/environment.sh, never logged. Secrets are
    # limited to recognized service credentials, rather than dumping the environment.
    paths = {key: str(globals()[key]) for key in (
        'MODELS_DIR', 'DATASETS_DIR', 'OUTPUT_DIR', 'RESULTS_DIR', 'CONFIG_DIR')}
    keys = ('HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'HF_TOKEN', 'ANTHROPIC_API_KEY',
            'PYTHON', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'SLURM_ACCOUNT', 'SLURM_PARTITION')
    values = {**{key: os.environ[key] for key in keys if key in os.environ}, **paths}
    for key, value in values.items():
        print(f'export {key}={shlex.quote(value)}')


if __name__ == '__main__':
    main()
