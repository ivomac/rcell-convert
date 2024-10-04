"""Classes for the metadata and acquisition data in the HDF5 file.

See db.py for the main interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dict import flatten, pad_keytuples
from .io import get, keys, load
from .plot import Axes, Figure, PlotObject
from .stimulus import PulseStimulus, StimCsv, StimType

STIMCSV = StimCsv()


class RCell:
    """Interface to load data from an existing rCell.

    The class provides methods to load data from an rCell file.

    Attributes:
        path (Path): The path to the rCell.
        id (str): The id of the rCell.
        metadata (dict): The metadata ("general" field) of the rCell.
        stimulus (dict): The stimulus information on the rCell.
        meta_row (pd.Series): The metadata and stimulus information in a pandas Series.

    Methods:
        get: Get a dataset from the rCell.
        load: Load rCell group as a nested dictionary.
        keys: Get all keys in a group of the rCell.
        protocol | prt: Get the protocol object

    """

    def __init__(self, path: Path):
        """Initialize the rCell object.

        Args:
            path (Path): The path to the rCell.

        """
        self.path: Path = path
        self.id: str = self.path.stem

        self.prt = self.protocol
        self.parent = None
        return

    def __iter__(self):
        """Iterate over the protocols in the rCell."""

        def iter(self):
            for prt in self.keys("/acquisition/timeseries"):
                yield self.protocol(prt)
            return

        return iter(self)

    def __repr__(self) -> str:
        """Return the id of the rCell."""
        return f"{self.id}"

    @cached_property
    def metadata(self) -> dict:
        """The metadata ("general" field) of the rCell."""
        meta = self.load("/general")
        meta["experiment"].pop("nanioncsv_log", None)
        return meta

    @cached_property
    def stimulus(self) -> dict:
        """The stimulus dictionary on the rCell."""
        return self.load("/stimulus/presentation")

    @property
    def meta_row(self) -> pd.Series:
        """Get the metadata and stimulus information in a pandas Series.

        This method is appropriate to use when creating a database of rCells.

        Returns:
            pd.Series: The metadata and stimulus information.

        """
        flat_row = flatten(self.metadata)
        flat_row[("id", "")] = self.id

        for protocol, dic in self.stimulus.items():
            flat_row[("stim_id", protocol)] = dic["stim_id"]

        flat_row = pad_keytuples(flat_row, 2)
        return pd.Series(flat_row)

    def get(self, key: str | Path):
        """Get a dataset from the rCell.

        Args:
            key (str | Path): The key path to get.

        Returns:
            Any: The data loaded from file.

        """
        return get(self.path, key)

    def load(self, root: str | Path = ""):
        """Load rCell as a nested dictionary.

        Args:
            root (str, optional): root group to load.

        Returns:
            dict | Any: nested dictionary or the data loaded from file.

        """
        return load(self.path, root=root)

    def keys(self, root: str | Path = ""):
        """Get all keys in a group of the rCell.

        Args:
            root (str | Path, optional): root group to inspect.

        Returns:
            list[str]: list of keys in the group.

        """
        return keys(self.path, root=root)

    def protocol(self, protocol: str):
        """Get the protocol object.

        Args:
            protocol (str): The protocol to get.

        Returns:
            Protocol: The protocol object.

        Raises:
            KeyError: If the protocol is not found.

        """
        if protocol not in self.stimulus:
            raise KeyError(f"Protocol '{protocol}' not found in rCell.")

        return Protocol(protocol, self)


class AcquisitionPart(ABC):
    """Base class for all parts of the acquisition data.

    Attributes:
        id (str | int): The id of the object.
        parent (AcquisitionPart | RCell): The parent object.
        name (str): The name of the class.
        path (Path): The path to the object.

    Methods:
        iter: Iterate over the children of the object.
        get: Get a field from the rCell file relative to the current level.
        load: Load the object from the rcell file as a dictionary.

    """

    def __init__(self, id, parent):
        """Initialize an AcquisitionPart object.

        Args:
            id (str | int): The id of the object.
            parent (AcquisitionPart | RCell): The parent object.

        """
        self.id: str | int = id
        self.parent = parent
        self.name = self.__class__.__name__
        self.path: Path = self.parent.path

    def __iter__(self) -> Iterator[AcquisitionPart]:
        """Iterate over the children of the object."""
        return self.iter()

    def __repr__(self):
        """Return a string representation of the object."""
        parent = "" if self.parent is None else f"{self.parent} | "
        return f"{parent}{self.name}:{self.id}"

    @cached_property
    def id_path(self) -> tuple:
        """A tuple of the ids of the object and its parents."""
        ids = [self.id]
        while self.parent is not None:
            self = self.parent
            ids.append(self.id)
        return tuple(reversed(ids))

    @cached_property
    def keys(self):
        """A list of ids of the children of the object."""
        return []

    @cached_property
    def rcell(self) -> RCell:
        """The root rCell object."""
        obj = self
        while not isinstance(obj, RCell):
            obj = obj.parent
        return obj

    @abstractmethod
    def iter(self) -> Iterator[AcquisitionPart]:
        """Iterate over the children of the object."""
        pass

    def get(self, key):
        """Get a field from the rCell file relative to the current level.

        Args:
            key (str): The field to get.

        Returns:
            object: The field value.

        """
        return self.rcell.get(self.path / key)

    def load(self):
        """Load the object from the rcell file as a dictionary.

        Returns:
            dict: The object data.

        """
        return self.rcell.load(self.path)


class Protocol(AcquisitionPart):
    """Class for a protocol.

    The object can be iterated over to get the repetitions of the protocol.

    Attributes:
        id (str): The id of the protocol.
        name (str): A short name for the object.
        parent | RCell (RCell): The parent rCell object.
        keys (list[int]): The repetition numbers for the protocol.
        stim_id (int): The stimulus id for the protocol.
        stimulus (Stimulus): The stimulus object for the protocol.
        id_path (tuple): The ids of the object and its parents.

    Methods:
        repetition: Get the repetition data for a given repetition number.
        get: Get a field from the protocol group.
        load: Load the protocol group as a dictionary.

    """

    def __init__(self, id: str, parent: RCell):
        """Initialize a Protocol object.

        Args:
            id (str): The id of the protocol.
            parent (RCell): The parent rCell object.

        """
        super().__init__(id, parent)
        self.path = Path(f"/acquisition/timeseries/{self.id}/repetitions")
        self.name = "Prt"
        self.rep = self.repetition

    @cached_property
    def keys(self):
        """The repetition numbers for the protocol."""
        return [int(rep[10:]) for rep in self.parent.keys(self.path)]

    @cached_property
    def stim_id(self) -> int:
        """The stimulus id for the protocol."""
        return self.rcell.get(f"/stimulus/presentation/{self.id}/stim_id")  # type: ignore

    @cached_property
    def stimulus(self) -> StimType:
        """The Stimulus object for the protocol."""
        return STIMCSV.get(self.stim_id)

    def iter(self) -> Iterator[Repetition]:
        """Iterate over the repetitions of the protocol."""
        for rep in self.keys:
            yield self.repetition(rep)

    def repetition(self, rep: int):
        """Get the repetition data for a given repetition number.

        Args:
            rep (int): The repetition number.

        Returns:
            Repetition: The repetition object.

        Raises:
            ValueError: If the repetition number is not in the protocol.

        """
        if rep not in self.keys:
            raise ValueError(f"Protocol does not have repetition {rep}")
        return Repetition(rep, self)


class SubProtocol(AcquisitionPart, ABC):
    """Subclass for acquisition parts below protocol."""

    def __init__(self, id, parent):
        """Initialize a SubProtocol object.

        Args:
            id (str | int): The id of the object.
            parent: The parent object.

        """
        super().__init__(id, parent)

        self.data: (
            property
            | pd.DataFrame
            | pd.Series
            | cached_property[pd.DataFrame]
            | cached_property[pd.Series]
        )

    @cached_property
    def protocol(self) -> Protocol:
        """The parent protocol object."""
        obj = self
        while not isinstance(obj, Protocol):
            obj = obj.parent
        return obj

    def plot(self, ax: Axes | None = None) -> PlotObject:
        """Plot the data on the acquisition object.

        Args:
            ax (Axes, optional): The axes object to plot on.

        Returns:
            PlotObject: The figure and axis objects.

        Raises:
            ValueError: If the axes object is invalid.

        """
        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(str(self))
        else:
            fig = ax.figure
            if not isinstance(fig, Figure):
                raise ValueError("Invalid axes object")

        # scale index to ms
        ax.plot(self.data.index / 1000, self.data)  # type: ignore
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (nA)")

        return PlotObject(fig, ax)


class Repetition(SubProtocol):
    """Class for a repetition.

    The object can be iterated over to get the sweeps of the repetition.

    Attributes:
        id (int): The id of the repetition.
        name (str): A short name for the object.
        parent (Protocol): The parent protocol object.
        data (pd.DataFrame): The data for the repetition.
        keys (list): The sweep numbers for the repetition.
        parent_protocol (Protocol): The parent protocol object.
        RCell (RCell): The root rCell object.

    Methods:
        sweep: Get the sweep data for a given sweep number.
        get: Get a field from the repetition group.
        load: Load the repetition group as a dictionary.
        plot: Plot the data on the repetition.

    """

    def __init__(self, id: int, parent: Protocol):
        """Initialize a Repetition object.

        Args:
            id (int): The id of the repetition.
            parent (Protocol): The parent protocol object.

        """
        super().__init__(id, parent)
        self.path = self.parent.path / f"repetition{self.id}"
        self.name = "Rep"
        self.swp = self.sweep

    @cached_property
    def keys(self):
        """The sweep numbers for the repetition."""
        n_sweeps = self.parent.stimulus.sweep_count
        return list(range(n_sweeps))

    @cached_property
    def data(self) -> pd.DataFrame:
        """Get the data of the repetition.

        Returns:
            pd.DataFrame: The data for this repetition with the time as index.

        """
        data: np.ndarray = self.get("data")  # type: ignore
        time: np.ndarray = self.get("time")  # type: ignore

        return pd.DataFrame(
            data,
            columns=pd.Index(range(data.shape[1]), name="Sweep"),
            index=pd.Index(time, name="Time (us)"),
        )

    def iter(self) -> Iterator[Sweep]:
        """Iterate over the sweeps of the repetition."""
        for swp in self.keys:
            yield self.sweep(swp)

    def sweep(self, sweep: int) -> Sweep:
        """Get the sweep object for a given sweep number.

        Args:
            sweep (int): The sweep number.

        Returns:
            Sweep: The sweep object.

        Raises:
            ValueError: If the sweep number is not in the repetition.

        """
        if sweep not in self.keys:
            raise ValueError(f"Repetition does not have sweep {sweep}")
        return Sweep(sweep, self)


class Sweep(SubProtocol):
    """Class for a sweep.

    Can be iterated over to get the segments of the sweep (if protocol is Pulse).

    Attributes:
        id (int): The id of the sweep.
        name (str): A short name for the object.
        parent (Repetition): The parent repetition object.
        data (pd.Series): The data for the sweep.
        keys (list): The segment numbers for the sweep.
        stimulus (pd.Series): The stimulus data for the sweep.

    Methods:
        get: Get a field from the repetition appropriate to this sweep.
        segment: Get a segment of the sweep.

    """

    def __init__(self, id: int, parent: Repetition):
        """Initialize a Sweep object.

        Args:
            id (int): The id of the sweep.
            parent (Repetition): The parent repetition object.

        """
        super().__init__(id, parent)
        self.name = "Swp"
        self.seg = self.segment

    @property
    def data(self) -> pd.Series:
        """The data of the sweep as a pandas Series."""
        return self.parent.data.loc[:, self.id]

    @cached_property
    def keys(self):
        """The segment numbers for the sweep.

        Raises:
            ValueError: If the stimulus is not a PulseStimulus.

        """
        stim = self.protocol.stimulus
        if not isinstance(stim, PulseStimulus):
            raise ValueError("Stimulus is not a PulseStimulus")
        n_segments = stim.t_pairs.shape[1]
        return list(range(n_segments))

    @cached_property
    def stimulus(self) -> pd.Series:
        """The stimulus data for the sweep as a pandas Series."""
        sweep_ind = self.id
        stim = self.protocol.stimulus

        if stim.sweep_count == 1:
            sweep_ind = 0

        t = stim.t_matrix[sweep_ind]
        v = stim.v_matrix[sweep_ind]

        return pd.Series(v, index=pd.Index(t))

    def iter(self) -> Iterator[PulseSegment]:
        """Iterate over the segments of the sweep."""
        for seg in self.keys:
            yield self.segment(seg)

    def get(self, key: str):
        """Get a field from the repetition appropriate to this sweep."""
        obj = super().get(key)
        if hasattr(obj, "__getitem__"):
            obj = obj[self.id].item()  # type: ignore
        return obj

    def segment(self, seg: int) -> PulseSegment:
        """Get a segment of the sweep.

        Args:
            seg (int): The segment number.

        Returns:
            PulseSegment: The segment object.

        Raises:
            ValueError: If the sweep does not have the segment.

        """
        if seg not in self.keys:
            raise ValueError(f"Sweep does not have segment {seg}")
        return PulseSegment(seg, self)


class PulseSegment(SubProtocol):
    """Class for a pulse segment.

    The object can be iterated over to get subsegments defined in the subs attribute.
        By default, the subsegments are defined as 5% increments from 5% to 95%
        and from -95% to -5%.

    Attributes:
        id (int): The id of the segment.
        name (str): A short name for the object.
        parent (Sweep): The parent sweep object.
        voltage (list): The voltage limits of the segment.
        data (pd.Series): The data for the segment.
        subs (list): The subsegment percentages to iterate on.

    Methods:
        subsegment | sub: Get a subsegment of the pulse segment.

    """

    def __init__(self, id: int, parent: Sweep):
        """Initialize a PulseSegment object.

        Args:
            id (int): The id of the segment.
            parent (Sweep): The parent sweep object.

        """
        super().__init__(id, parent)

        self.name = "Seg"
        self.sub = self.subsegment
        self.subs: list[str] = [
            f"{pct:+d}%" for pct in chain(range(5, 100, 5), range(-5, -100, -5))
        ]

    @property
    def stimulus(self) -> PulseStimulus:
        """The stimulus object.

        Raises:
            ValueError: If the stimulus is not a PulseStimulus.

        """
        stim = self.protocol.stimulus
        if not isinstance(stim, PulseStimulus):
            raise ValueError("Stimulus is not a PulseStimulus")
        return stim

    @cached_property
    def voltage(self) -> list:
        """The voltage limits of the segment as a list."""
        par_id: int = self.parent.id
        id: int = cast(int, self.id)
        return self.stimulus.v_pairs[par_id, id].tolist()

    @cached_property
    def time(self) -> list:
        """The time limits of the segment as a list."""
        par_id: int = self.parent.id
        id: int = cast(int, self.id)
        return self.stimulus.t_pairs[par_id, id].tolist()

    @property
    def data(self) -> pd.Series:
        """The data of the segment as a pandas Series with the time as index."""
        start, end = self.time
        return self.parent.data.loc[start:end]

    def iter(self) -> Iterator[PulseSubSegment]:
        """Iterate over the subsegments of the pulse segment."""
        for pct in self.subs:
            yield self.subsegment(pct)

    def subsegment(self, pct) -> PulseSubSegment:
        """Get a subsegment of the pulse segment.

        Args:
            pct (float): The percentage of the subsegment.

        Returns:
            PulseSubSegment: The subsegment object.

        """
        return PulseSubSegment(pct, self)


class PulseSubSegment(SubProtocol):
    """Class for a pulse subsegment.

    The subsegment is defined by a percentage of the pulse duration.
    Positive/Negative values are the percentage of the pulse duration
    from the start/end of the pulse.

    Attributes:
        id (str): The percentage of the subsegment as a string.
        name (str): A short name for the object.
        parent (PulseSegment): The parent segment object.
        voltage (list): The voltage limits of the subsegment.
        data (pd.Series): The data for the subsegment.
        pct (float): The percentage of the subsegment as a float.

    """

    def __init__(self, id, parent):
        """Initialize a PulseSubSegment object.

        Args:
            id (str | float): The percentage of the subsegment.
            parent (PulseSegment): The parent segment object.

        Raises:
            ValueError: If the id is not a float or string
                or if the percentage is not between -1 and 1.

        """
        if isinstance(id, str):
            id = float(id[:-1]) / 100
        elif not isinstance(id, float | int):
            raise ValueError(f"Invalid id type: {type(id)}")

        if not -1 <= id <= 1:
            raise ValueError(f"Invalid percentage: {id}")

        str_id = f"{id * 100:+.0f}%"
        super().__init__(str_id, parent)
        self.name = "Sub"
        self.pct = id

    @cached_property
    def voltage(self) -> tuple[int, int]:
        """The voltage limits of the subsegment in milliVolts."""
        return percent_scale(self.pct, *self.parent.voltage)

    @cached_property
    def time(self) -> tuple[int, int]:
        """The time limits of the subsegment in microseconds."""
        return percent_scale(self.pct, *self.parent.time)

    @property
    def data(self) -> pd.Series:
        """The data of the subsegment as a pandas Series with the time as index."""
        start, end = self.time
        return self.parent.data.loc[start:end]

    def iter(self):
        """Not implemented for PulseSubSegment."""
        raise NotImplementedError("SubSegments have no children.")


def percent_scale(pct, start, end) -> tuple[int, int]:
    """Scale the start and end values by the percentage.

    Args:
        pct: Percentage to scale by.
        start: Start value.
        end: End value.

    Returns:
        tuple[int, int]: Scaled start and end values.

    """
    if pct > 0:
        end = start + pct * (end - start)
        # ceil() the end
        int_end = int(end)
        end = int_end if end == int_end else int_end + 1
    elif pct < 0:
        # floor() the start
        start = int(end + pct * (end - start))
    return start, end
