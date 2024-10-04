"""Interface to list, find, and load rCells.

The purpose of this module is to provide a way to access the stimulus and
acquisition data at different levels of the hierarchy. The CellDB is the
starting point to access stimulus and acquisition classes, which are designed
to be used as iterators to traverse the tree of the acquisition data.

We start with a CellDB singleton instance (from the rcell module) and initialize RCell
objects, then navigate down until we reach a segment of a sweep:

CellDB -> RCell -> Protocol -> Repetition -> Sweep (-> Segment -> SubSegment)

From Repetition onwards, the data is stored in a pandas DataFrame/Series with the time
as the index.

Segments and SubSegments are specific to the Pulse stimulus type: They hold the data
on a step of the pulse stimulus.

Usage:

    from db import CellDB

    cellDB = CellDB()

    # No need to specify the full path, just the ID
    rcell = cellDB.load(rcell_name)

    # Plot all data of the first repetition of Activation
    # and the stimulus
    prt = rcell.protocol("Activation")
    rep1 = prt.repetition(1)
    rep1.plot()
    prt.stimulus.plot()

    # Calculate the leak per sweep
    input_resistance = []
    for sweep in rep1:
        data = sweep.segment(0).data
        input_resistance.append(data.mean())

    # Silly example to plot the first and last
    # 10% of data of all segments (assuming Pulse stimulus)
    for protocol in rcell:
        for repetition in protocol:
            for sweep in repetition:
                for segment in sweep:
                    for percent in [0.1, -0.1]:
                        sub = segment.percent(percent)
                        sub.plot()

"""

from collections import Counter
from functools import cache
from os import environ
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .rcell import RCell
from .utils import singleton


@singleton
class CellDB:
    """A singleton class to interface with rCells (list, find, load).

    Attributes:
        path (dict[str, Path]): Path to rCell NWB files.
            Keys are machine names, values are Path objects.

    Methods:
        list: List rCell IDs from a specific machine or all machines.
        validate: Check for duplicate IDs and missing files.
        load: Load an rCell by ID.
        get_path: Get the path to an rCell by ID.
        qpc_path: Get the path to a QPC rCell by ID.
        igor_path: Get the path to an Igor rCell by ID.
        syncropatch_path: Get the path to a Syncropatch rCell by ID.
        meta_df: Create a pandas DataFrame with metadata and stimulus information.

    """

    def __init__(self):
        """Initialize the CellDB object."""
        self.path = {
            "qpc": Path(environ["QPC_NWB_PATH"]),
            "igor": Path(environ["IGOR_NWB_PATH"]),
            "syncropatch": Path(environ["SYNCROPATCH_NWB_PATH"]),
        }
        return

    @cache
    def _list(self, machine: str) -> list[str]:
        return [path.stem for path in self.path[machine].rglob("*.nwb")]

    def list(self, machine: str = "all") -> list[str]:
        """List rCell IDs from a specific machine or all machines.

        Args:
            machine (str, optional): Machine to list rCells from.
                Can be "qpc", "igor", "syncropatch", or "all".
                Defaults to "all".

        Returns:
            list[str]: List of rCell IDs.

        Raises:
            ValueError: If an unknown machine is specified.

        """
        if machine == "all":
            out = []
            for machine in self.path:
                out += self._list(machine)
        elif machine in self.path:
            out = self._list(machine)
        else:
            raise ValueError(f"Unknown machine {machine}.")
        return out

    def validate(self):
        """Validate the existence of all rCells.

        Raises:
            ValueError: If there are duplicate rCell IDs.
            FileNotFoundError: If an expected rCell file is not found.

        """
        all_cells = self.list("all")
        counts = Counter(all_cells)
        dups = {k: v for k, v in counts.items() if v > 1}
        if dups:
            raise ValueError("Duplicate rCell IDs found:", dups)

        for id in all_cells:
            cell_path = self.get_path(id)
            if not cell_path.is_file():
                raise FileNotFoundError(f"Expected rCell in path {cell_path}.")
        return

    def load(self, id: str) -> RCell:
        """Load an rCell by its ID.

        Args:
            id (str): The rCell ID.

        Returns:
            RCell: The loaded rCell object.

        Raises:
            FileNotFoundError: If the rCell file is not found.

        """
        path = self.get_path(id)

        if not path.is_file():
            raise FileNotFoundError(f"Expected rCell in path {path}.")

        return RCell(path)

    def get_path(self, id: str) -> Path:
        """Get the file path for an rCell ID.

        Args:
            id (str): The rCell ID.

        Returns:
            Path: The file path of the rCell.

        """
        if id.startswith("qpc"):
            return self.qpc_path(id)
        if id.startswith("HA"):
            return self.igor_path(id)
        return self.syncropatch_path(id)

    def qpc_path(self, id: str) -> Path:
        """Get the path for a 'qpc' rCell ID.

        Args:
            id (str): The rCell ID.

        Returns:
            Path: The file path of the rCell.

        """
        return (self.path["qpc"] / id).with_suffix(".nwb")

    def igor_path(self, id: str) -> Path:
        """Get the path for an 'igor' rCell ID.

        Args:
            id (str): The rCell ID.

        Returns:
            Path: The file path of the rCell.

        """
        folder = str(int(id.split("_")[-1]) // 1000)
        return (self.path["igor"] / folder / id).with_suffix(".nwb")

    def syncropatch_path(self, id: str) -> Path:
        """Get the path for a 'syncropatch' rCell ID.

        Args:
            id (str): The rCell ID.

        Returns:
            Path: The file path of the rCell.

        """
        date, num, cl, _ = id.split("_")
        return (
            self.path["syncropatch"]
            / "rcell"
            / date
            / f"{date}_{num}"
            / f"{date}_{num}_{cl}"
            / id
        ).with_suffix(".nwb")

    def meta_df(self, machine: str, progress: bool = False) -> pd.DataFrame:
        """Create a pandas DataFrame with metadata and stimulus information.

        Args:
            machine (str): Machine to load metadata from.
                Can be "qpc", "igor", or "syncropatch".
            progress (bool, optional): Whether to display a progress bar.
                Defaults to False.

        Returns:
            pd.DataFrame: Metadata DataFrame.

        Raises:
            ValueError: If no rCells are found for the specified machine.

        """
        rows = [
            self.load(rcell).meta_row
            for rcell in tqdm(
                self.list(machine),
                disable=not progress,
                desc=f"Loading {machine} metadata",
                colour="blue",
                dynamic_ncols=True,
            )
        ]

        if not rows:
            raise ValueError(f"No rCells found for machine {machine}.")

        df = pd.DataFrame(rows).fillna("")

        df.columns = pd.MultiIndex.from_tuples(df.columns)

        df = df[sorted(df.columns)].set_index(("id", "")).sort_index()
        df.index.name = None
        return df
