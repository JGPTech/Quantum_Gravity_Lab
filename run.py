"""Friendly entry points for the original applications; run from any directory."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "simulation": "Legacy_Modules.unified_simulation",
    "bell": "Legacy_Modules.unified_bell_state",
    "darkmatter": "Legacy_Modules.unified_darkmatter",
    "calibrate": "Legacy_Modules.calibrate",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch the original Quantum Gravity Lab applications.")
    parser.add_argument("application", choices=COMMANDS, help="calibrate runs the full grid scan; viewers open a desktop window")
    args, remaining = parser.parse_known_args(argv)
    # Existing CSV lookups/writes are relative to the working directory.
    os.chdir(ROOT)
    sys.argv = [COMMANDS[args.application]] + remaining
    runpy.run_module(COMMANDS[args.application], run_name="__main__")


if __name__ == "__main__":
    main()
