"""Create rCell files from QPC data.

Input and output folders are set in the .env file.

The script can be run with a list of job_ids as arguments or
without arguments to run on all experiments in the Google Sheet.
"""

import argparse

from ..conversion.google_sheet import GOOGLE_SHEET as GS
from ..conversion.rcell import RCell


def main(argin=None):
    """Create rCell files from QPC data."""
    args = parse_args(argin)
    create(
        *args.job_ids,
        overwrite=args.overwrite,
        drop=not args.incomplete,
        keep_reps=args.reps,
        print_report=args.print,
    )
    return


def create(*job_ids, print_report=False, **kwargs):
    """Create rCell files from QPC data."""
    if not job_ids:
        job_ids = GS.job_ids

    for job_id in job_ids:
        cell = RCell(job_id)
        if not print_report:
            print(f"\rCreating rCell file for job {job_id} ", end="")
        report = cell.create(**kwargs)
        if print_report:
            print(report)
    return


def parse_args(args):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Replace existing rCell files",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print",
        help="Print the report for each job_id",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--incomplete",
        help="Do not filter out incomplete runs of protocols",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--reps",
        help="Keep up to this number of repetitions per protocol",
        type=int,
        default=0,
    )
    parser.add_argument(
        "job_ids",
        nargs="*",
        type=int,
    )

    return parser.parse_args(args)
