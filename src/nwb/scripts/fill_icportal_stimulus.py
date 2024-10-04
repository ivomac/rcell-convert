"""A script to plot stimulus."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ..src.stimulus import StimCsv

STIMCSV = StimCsv()


def main():
    """Create plots for all stimuli."""
    args = parse_args()

    for stim_id, stim in STIMCSV:
        print(stim_id)

        plot_filepath = args.saving_path / f"stim_{stim_id:03d}.png"

        if args.overwrite or not plot_filepath.exists():
            fig = stim.plot().fig
            fig.savefig(plot_filepath, dpi=100)
            plt.close(fig)

        json_filepath = plot_filepath.with_suffix(".json")

        if args.overwrite or not json_filepath.exists():
            with open(json_filepath, "w") as fp:
                json.dump(stim.to_dict(), fp)

    return


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--overwrite",
        help="Replace existing plots",
        action="store_true",
    )

    parser.add_argument(
        "saving_path",
        type=Path,
    )

    return parser.parse_args()
