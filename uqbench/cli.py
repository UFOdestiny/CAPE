"""Discoverable command line entry point for the benchmark."""
from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    'pipeline': ('data', 'generate', 'replay', 'energy', 'followup', 'reward', 'judge', 'evaluate', 'nli'),
    'studies': ('ablation', 'abstention', 'attribution', 'audit', 'controls', 'decision', 'dependence', 'kscale',
                'refinement', 'robustness', 'within_question'),
    'labeling': ('extract', 'regrade'),
    'reporting': ('report',),
    'blackbox': ('batch',),
}
STAGES = {name: f'uqbench.{group}.{name}' for group, names in COMMANDS.items() for name in names}
STAGES['blackbox-evaluate'] = 'uqbench.blackbox.evaluate'


def main() -> None:
    description = 'Response-level uncertainty benchmark. Available stages:\n' + '\n'.join(
        f'  {group}: {", ".join(names)}' for group, names in COMMANDS.items())
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=sorted(STAGES), help='use STAGE --help for stage options')
    args = parser.parse_args(sys.argv[1:2])
    remainder = sys.argv[2:]
    # Stage parsers own all options after the stage name, including --help.
    sys.argv = [f'uqbench {args.stage}', *remainder]
    importlib.import_module(STAGES[args.stage]).main()
