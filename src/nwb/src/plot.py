"""Plotting defaults for the project.

To add new styles, create a new .mplstyle file in the styles folder
and activate it with the style function.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

StyleFolder = Path(__file__).parent / "styles"


def style(style: str):
    """Apply a custom matplotlib style.

    Args:
        style (str): The name of the style file to use.

    Raises:
        FileNotFoundError: If the style file is not found.

    """
    if style.endswith(".mplstyle"):
        path = StyleFolder / style
    else:
        path = StyleFolder / f"{style}.mplstyle"

    if not path.is_file():
        raise FileNotFoundError(f"Style file not found: {path}")

    plt.style.use(path)

    return


class PlotObject:
    """A class that wraps a Matplotlib figure and axis, simulating a tuple (fig, ax).

    Attributes:
        fig (Figure): The Matplotlib figure object.
        ax (Axes): The Matplotlib axes object.

    """

    def __init__(self, fig: Figure, ax: Axes):
        """Initialize the PlotObject with a figure and axis and apply tight layout.

        Args:
            fig (Figure): The Matplotlib figure object.
            ax (Axes): The Matplotlib axes object.

        """
        self.fig: Figure = fig
        self.ax: Axes = ax
        fig.tight_layout()

    def __iter__(self):
        """Allow iteration over the figure and axis."""
        return iter((self.fig, self.ax))

    def __getitem__(self, key):
        """Allow indexed access to the figure and axis.

        Args:
            key (int): The index to access, 0 for figure and 1 for axis.

        Returns:
            Figure | Axes: The requested object (figure or axis).

        """
        return (self.fig, self.ax)[key]

    def __repr__(self):
        """Provide a string representation."""
        return f"PlotObject(fig={self.fig}, ax={self.ax})"


# Apply default styles
style("classic")
style("monospace")
