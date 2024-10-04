"""View the output rcells. By default, uses the `hdfview` tool.

Input and output folders are set in the .env file.

The script can be run with a list of job_ids as arguments
or without arguments to run on all experiments in the Google Sheet.
"""

import argparse
import subprocess as sp

from ..conversion.experiment import Experiment
from ..conversion.google_sheet import GOOGLE_SHEET as GS


def main(argin=None):
    """View the output rcells."""
    args = parse_args(argin)
    view(*args.job_ids, tool=args.tool)
    return


def view(*job_ids, tool=None):
    """View the output rcells."""
    if not tool:
        tool = ["hdfview"]
    """View the output rcells."""
    if not job_ids:
        job_ids = GS.job_ids

    paths = [str(Experiment(job_id).out) for job_id in job_ids]

    tool_cmd = tool + paths

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
        help="Tool to use. Default is 'hdfview'.",
        default="hdfview",
    )
    parser.add_argument(
        "job_ids",
        nargs="*",
        type=int,
    )

    args = parser.parse_args(args)

    if isinstance(args.tool, str):
        args.tool = args.tool.split()

    return args
