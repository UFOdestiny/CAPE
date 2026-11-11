"""Discoverable command line entry point of the CAPE pipeline (``python -m uqbench <stage>``)."""
from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    'pipeline': ('data', 'generate', 'replay', 'evaluate'),
    'labeling': ('regrade', 'extract'),
    'blackbox': ('batch',),
}
STAGES = {name: f'uqbench.{group}.{name}' for group, names in COMMANDS.items() for name in names}
STAGES['blackbox-evaluate'] = 'uqbench.blackbox.evaluate'


def main() -> None:
    description = 'CAPE pipeline. Available stages:\n' + '\n'.join(
        f'  {group}: {", ".join(names)}' for group, names in COMMANDS.items())
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=sorted(STAGES), help='use STAGE --help for stage options')
    args = parser.parse_args(sys.argv[1:2])
    remainder = sys.argv[2:]
    # Stage parsers own all options after the stage name, including --help.
    sys.argv = [f'uqbench {args.stage}', *remainder]
    importlib.import_module(STAGES[args.stage]).main()
