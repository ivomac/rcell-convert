"""A script to view the input data files.

By default, uses `libreoffice`.
Input and output folders are set in the .env file.
Run the script with a list of job_ids as arguments.
"""

import argparse
import subprocess as sp

from ..conversion.experiment import Experiment

DEF_TOOL = ["libreoffice", "--calc"]
DEF_TOOL_STR = " ".join(DEF_TOOL)


def main(argin=None):
    """View the input data files."""
    args = parse_args(argin)

    view(*args.job_ids, tool=args.tool)
    return


def view(*job_ids, tool=DEF_TOOL):
    """View the input data files."""
    for job_id in job_ids:
        exp = Experiment(job_id)

        tool_cmd = [
            *tool,
            str(exp.meta.file),
            str(exp.raw.file),
        ]
        sp.run(tool_cmd, text=True, capture_output=True)
    return


def parse_args(args):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tool",
        default=DEF_TOOL_STR,
        help=f"Tool to use for visualization. Default is {DEF_TOOL_STR}.",
    )
    parser.add_argument(
        "job_ids",
        nargs="*",
        type=int,
    )

    args = parser.parse_args(args)

    args.tool = args.tool.split()

    return args
