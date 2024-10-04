"""A python module to convert QPC data to NWB format.
Provides a command line interface to run scripts.
Run each script with the -h flag to see the available options.
"""

import argparse
import importlib
from pathlib import Path
from .conversion.google_sheet import GOOGLE_SHEET as GS

SCRIPT_FOLDER = Path(__file__).parent / "scripts"

SCRIPT_FILES = [script for script in SCRIPT_FOLDER.glob("*.py")]

SCRIPTS = [script.stem for script in SCRIPT_FILES]


def main():
    cmd, args = parse_args()
    script = importlib.import_module(f".scripts.{cmd.name}", package="qpc")
    script.main(args)
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description=generate_description(),
        add_help=False,
        prog="python -m /path/to/qpc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update the Google Sheet.",
    )

    parser.add_argument(
        "name",
        choices=SCRIPTS,
        metavar="<script>",
        nargs="?",
    )

    cmd, args = parser.parse_known_args()

    if cmd.update:
        GS.update()

    if cmd.name is None:
        parser.print_help()
        exit()
    return cmd, args


def generate_description():
    script_descs = [script_description(script) for script in SCRIPT_FILES]
    opts = "\n".join(script_descs)
    desc = f"{__doc__}\nAvailable scripts:\n{opts}"
    return desc


def script_description(script):
    with script.open() as f:
        line = f.readline()
    line = line.strip("\"'# \n\t")
    return f"  {script.stem}: {line}"


if __name__ == "__main__":
    main()
