"""A script to plot stimulus."""

import argparse

import matplotlib.pyplot as plt

from ..src.stimulus import StimCsv

STIMCSV = StimCsv()


def main():
    """Create a plot for a stimulus."""
    args = parse_args()

    fig = STIMCSV.get(args.stimulus_id).plot().fig

    filename = f"stim_{args.stimulus_id:03d}.png"
    fig.savefig(filename, dpi=100)

    plt.close(fig)

    return


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stimulus_id",
        type=int,
    )
    return parser.parse_args()
