"""A module to add stimulus information to the rCell dictionary."""

from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import cache, cached_property
from itertools import accumulate, pairwise
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.signal import find_peaks

from .plot import PlotObject
from .utils import singleton

load_dotenv()

global StimType

STIM_FOLDER = Path(environ["STIMULUS_PATH"])


@dataclass
class BaseStimulus:
    """A dataclass for stimulus objects.

    Attributes:
        stim_id (int): The stimulus ID.
        name (str): The stimulus name.
        type (str): The stimulus type.
        sweep_interval (int): The interval between sweeps.
        sweep_count (int): The number of sweeps.
        command (str): The stimulus command string.
        repetition_count (int): The number of times the protocol is repeated in a row.

    """

    stim_id: int
    name: str
    type: str
    sweep_interval: int
    sweep_count: int
    command: str

    @property
    def info(self):
        """Return stimulus information as a dictionary."""
        return asdict(self)

    @cached_property
    def repetition_count(self) -> int:
        """The number of times the protocol is repeated in a row.

        Some protocols consist of a command that defines a single sweep
        which is then repeated a number of times. When this is the case,
        the repetition_count information is stored in the sweep_count field.
        When the command defines multiple sweeps, this value should be 1.
        """
        return self.sweep_count

    @cached_property
    def t_matrix(self):
        """Matrix of time values per sweep. Ideal for plotting."""
        raise NotImplementedError

    @cached_property
    def v_matrix(self):
        """Matrix of voltage values per sweep. Ideal for plotting."""
        raise NotImplementedError

    @cached_property
    def duration(self) -> int:
        """The duration of the whole stimulus in microseconds."""
        raise NotImplementedError

    @cached_property
    def v_min(self):
        """The minimum voltage value in the stimulus."""
        return self.v_matrix.min()

    @cached_property
    def v_max(self):
        """The maximum voltage value in the stimulus."""
        return self.v_matrix.max()

    def validate(self):
        """Validate the command string and consistency with the name and sweep_count."""
        raise NotImplementedError

    def plot(self):
        """Plot the stimulus command.

        The plot shows the voltage against time for each sweep.

        Returns:
            plt.Figure: The plot figure.

        """
        fig, ax = plt.subplots()
        ax.plot(self.t_matrix.T / 1000, self.v_matrix.T)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title(f"{self.name} #{self.stim_id}")
        return PlotObject(fig, ax)

    def to_dict(self):
        """Convert the stimulus information to a dictionary."""
        return {
            "id": self.stim_id,
            "name": self.name,
            "source_type": self.type,
            "sweep_interval": self.sweep_interval,
            "sweep_count": self.sweep_count,
            "command": self.command,
            "duration": self.duration,
            "step_time": self.t_matrix.T.tolist(),
            "step_data": self.v_matrix.T.tolist(),
            "minV": self.v_min,
            "maxV": self.v_max,
            "x_interval": 0.1,
            "unit_time": "us",
            "unit_data": "mV",
        }


class VRestStimulus(BaseStimulus):
    """A class to represent a resting (zero flat) potential stimulus."""

    @cached_property
    def duration(self) -> int:
        """The duration of the resting stimulus in microseconds."""
        return self.t_matrix[0, -1]

    @cached_property
    def t_matrix(self):
        """Matrix of time values for the resting stimulus."""
        return np.array([[0, 2500000]])

    @cached_property
    def v_matrix(self):
        """Matrix of voltage values for the resting stimulus."""
        return np.array([[0, 0]])


class APStimulus(BaseStimulus):
    """A class to represent an action potential stimulus.

    The stimulus is stored in a .dat file with the same name as the command.
    These are text files with two columns: time and voltage.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the APStimulus object and load the .dat file."""
        super().__init__(*args, **kwargs)
        self.file_path = STIM_FOLDER / f"{self.command}.dat"
        self.df = pd.read_csv(self.file_path, sep="  ", header=None)
        self.df.columns = ["time", "voltage"]
        # convert time from ms to us
        self.df.time = (self.df.time * 1000).map(round)

    @cached_property
    def duration(self) -> int:
        """The duration of the action potential stimulus in microseconds."""
        return self.df.time.iat[-1]

    @cached_property
    def t_matrix(self) -> np.ndarray:
        """Matrix of time values for the action potential stimulus."""
        return self.df.time.values.reshape(1, -1)

    @cached_property
    def v_matrix(self) -> np.ndarray:
        """Matrix of voltage values for the action potential stimulus."""
        return self.df.voltage.values.reshape(1, -1)

    @cached_property
    def peaks(self) -> np.ndarray:
        """Identify middle points among maxima of the stimulus.

        Returns:
            np.ndarray: Array of middle points between peaks.

        """
        peak_indices, _ = find_peaks(self.df.voltage.values, height=0, prominence=0.01)
        return np.round((peak_indices[1:] + peak_indices[:-1]) / 2).astype(int)


class BaseSegment:
    """A class to represent a segment of a Pulse stimulus command string.

    Attributes:
        min (int): The minimum value of the segment.
        step (int): The step value of the segment.
        max (int): The maximum value of the segment.
        is_ramp (bool): True if the segment is a ramp.
        n_sweeps (int): The minimum number of sweeps for this segment.

    """

    def __init__(
        self,
        min: int,
        step: int = 0,
        max: int | None = None,
    ):
        """Initialize the BaseSegment class.

        Used to represent (part of) a segment of a pulse stimulus.

        Args:
            min (int): The minimum value of the segment.
            step (int, optional): The step value of the segment.
                Defaults to 0.
            max (int, optional): The maximum value of the segment.
                Defaults to None.

        Raises:
            IndexError: If the iterator is exhausted.

        """
        self.min = min
        self.step = step
        self.max = max if max is not None else self.min

        self.is_ramp = False
        if self.step == 0:
            self.is_ramp = self.min != self.max
            self.n_sweeps = 1
        else:
            self.n_sweeps = 1 + (self.max - self.min) // self.step

    def __iter__(self):
        """Iterate over the values in the segment."""
        if self.step:
            v = self.min
            while self.step > 0 and v <= self.max or self.step < 0 and v >= self.max:
                yield v, v
                v += self.step
        else:
            while True:
                yield self.min, self.max

    def __repr__(self):
        """Return a string representation of the segment."""
        return ":".join(str(v) for v in (self.min, self.step, self.max))


class PulseSegment:
    """A class to represent a segment of a Pulse stimulus command string.

    Used to represent a full segment (time and voltage) of a pulse stimulus.

    A segment of a Pulse stimulus command string is formatted as:
        ...;min_voltage:step_voltage:max_voltage:duration;...
    or
        ...;min_voltage:step_voltage:max_voltage:min_time:step_time:max_time;...

    This class parses the segment and stores the time and voltage values.
    Also provides an iterator over the sequences defined by the segment.

    Example:
        >>> seg = PulseSegment("0:10:20:1000")
        >>> seg.v
        0:10:20
        >>> seg.t
        1000:0:1000
        >>> seg.n_sweeps
        3
        >>> it = iter(seg.v)
        >>> [next(it) for _ in range(3)]
        [(0, 0), (10, 10), (20, 20)]

    """

    def __init__(self, pulse: str):
        """Initialize the PulseSegment object.

        Args:
            pulse (str): A segment of the pulse stimulus command string.

        """
        nums = [Decimal(d) for d in pulse.split(":")]

        if len(nums) not in [4, 6]:
            raise ValueError(f"Invalid pulse segment length: {pulse}")

        voltage, time = nums[:3], nums[3:]

        self.t = BaseSegment(*(int(1000 * t) for t in time))
        self.v = BaseSegment(*(int(v) for v in voltage))

        self.is_ramp = self.v.is_ramp

        self.n_sweeps = max(self.v.n_sweeps, self.t.n_sweeps)

    def __repr__(self):
        """Return a string representation of the PulseSegment."""
        return f"{self.v}:{self.t}"


class PulseStimulus(BaseStimulus):
    """A class to represent a pulse stimulus command string.

    Attributes:
        repetition_count (int): The number of times the protocol is repeated in a row.
        segments (list[PulseSegment]): Parsed segments of the command string.
        t_pairs (np.ndarray): Matrix of time pairs (init, fin) per sweep & segment.
        v_pairs (np.ndarray): Matrix of voltage pairs (init, fin) per sweep & segment.
        t_matrix (np.ndarray): Matrix of time values per segment/sweep, for plotting.
        v_matrix (np.ndarray): Matrix of voltage values per sweep, for plotting.
        duration (int): The duration of the entire PulseStimulus in microseconds.

    Methods:
        validate: Validate the PulseStimulus command string.
        plot: Plot the stimulus command.
        to_dict: Convert the stimulus information to a dictionary.

    """

    @cached_property
    def repetition_count(self) -> int:
        """The number of times the protocol is repeated in a row."""
        max_sweeps = max(seg.n_sweeps for seg in self.segments)
        if max_sweeps > 1:
            return 1
        return self.sweep_count

    def validate(self):
        """Validate the PulseStimulus command string."""
        if not self.command.endswith(";"):
            raise ValidationError(f"Expected semicolon at end: {self.command}")

        for seg in self.segments:
            if not (
                seg.v.n_sweeps == seg.t.n_sweeps
                or seg.v.n_sweeps == 1
                or seg.t.n_sweeps == 1
            ):
                raise ValidationError(
                    f"Inconsistent sweep counts: {seg.v.n_sweeps} != {seg.t.n_sweeps}"
                )

            if seg.t.is_ramp:
                raise ValidationError(f"Expected equal min and max in time: {seg.t}")

        seg_sweeps = [seg.n_sweeps for seg in self.segments]

        sweep_vals = set(seg_sweeps)

        if len(sweep_vals) not in [1, 2]:
            raise ValidationError(
                f"Inconsistent sweep counts in segments: {seg_sweeps}"
            )

        max_sweeps = max(sweep_vals)

        if max_sweeps > 1 and self.sweep_count != max_sweeps:
            raise ValidationError(
                "Mismatched sweep_count value and command:"
                + f" {self.sweep_count} != {max_sweeps} (from command)"
            )

        if self.name == "Ramp" and not any(seg.is_ramp for seg in self.segments):
            raise ValidationError(
                f"Expected at least one ramp segment in command: {self.command}"
            )

    @cached_property
    def duration(self) -> int:
        """The duration of the entire PulseStimulus in microseconds."""
        return sum(seg.t.max for seg in self.segments)

    @cached_property
    def segments(self) -> list[PulseSegment]:
        """Parsed segments of the command string."""
        return [PulseSegment(pulse) for pulse in self.command.split(";")[:-1]]

    @cached_property
    def t_pairs(self) -> np.ndarray:
        """Matrix of time pairs (initial, final) per sweep & segment.

        Returns:
            np.ndarray: A 3D array with shape (sweep_count, n_segments, 2).

        """
        matrix = []
        t_gens = [iter(seg.t) for seg in self.segments]

        for _ in range(self.sweep_count):
            nxt = [next(t_gen)[0] for t_gen in t_gens]
            lst = list(pairwise(accumulate(nxt, initial=0)))
            lst[-1] = (lst[-1][0], self.duration)
            matrix.append(lst)

        return np.array(matrix)

    @cached_property
    def v_pairs(self) -> np.ndarray:
        """Matrix of voltage pairs (initial, final) per sweep & segment.

        Returns:
            np.ndarray: A 3D array with shape (sweep_count, n_segments, 2).

        """
        matrix = []
        v_gens = [iter(seg.v) for seg in self.segments]

        for _ in range(self.sweep_count):
            nxt = [next(v_gen) for v_gen in v_gens]
            matrix.append(nxt)

        return np.array(matrix)

    @cached_property
    def t_matrix(self) -> np.ndarray:
        """Matrix of time values per segment/sweep. Ideal for plotting.

        This is the same data as t_pairs, but the array is reshaped.

        Returns:
            np.ndarray: A 2D array with shape (sweep_count, 2*n_segments).

        """
        return self.t_pairs.reshape(self.sweep_count, -1)

    @cached_property
    def v_matrix(self) -> np.ndarray:
        """Matrix of voltage values per sweep. Ideal for plotting.

        This is the same data as v_pairs, but the array is reshaped.

        Returns:
            np.ndarray: A 2D array with shape (sweep_count, 2*n_segments).

        """
        return self.v_pairs.reshape(self.sweep_count, -1)


StimType = BaseStimulus | VRestStimulus | PulseStimulus | APStimulus


@singleton
class StimCsv:
    """A singleton class to load and access the stimulus data from a CSV file.

    Attributes:
        path (Path): The path to the stimulus CSV file.
        data (pd.DataFrame): The stimulus data lazy-loaded from the CSV file.

    """

    def __init__(self, path: Path = STIM_FOLDER / "stimulus.csv"):
        """Initialize the StimCsv object.

        Args:
            path (Path, optional): Path to the stimulus CSV file.
                By default, the path is read from the STIMULUS_PATH env variable.

        """
        self.path = path

    def __iter__(self):
        """Iterate over the stimulus IDs in the CSV file."""
        self._iter_ids = iter(self.data.index)
        return self

    def __next__(self) -> tuple[int, StimType]:
        """Get the next stimulus ID and its corresponding stimulus data."""
        stim_id = next(self._iter_ids)
        return stim_id, self.get(stim_id)

    @cached_property
    def data(self) -> pd.DataFrame:
        """Lazy load the stimulus data."""
        # rename columns
        rename_map = {
            "ID": "stim_id",
            "Name": "name",
            "Type": "type",
            "SweepInterVal": "sweep_interval",
            "SweepCount": "sweep_count",
            "Command": "command",
        }

        return (
            pd.read_csv(self.path)
            .rename(columns=rename_map)
            .set_index("stim_id", drop=False)
        )

    @cache
    def get(self, stim_id: int) -> StimType:
        """Get the stimulus data for a given stimulus ID.

        Args:
            stim_id (int): The stimulus ID.

        Returns:
            StimType: The stimulus data as an instance of a stimulus class.

        """
        data = self.data.loc[stim_id].to_dict()

        if data["type"] == "Pulse":
            return PulseStimulus(**data)
        if data["name"] == "VRest":
            return VRestStimulus(**data)
        if data["name"] == "AP":
            return APStimulus(**data)

        raise ValueError(f"Unknown stimulus: {data}")

    def info(self, stim_ids: list[int]) -> dict:
        """Get the stimulus data for a list of stimulus IDs.

        This is the appropriate method to insert stimulus data on rCells.

        Args:
            stim_ids (list[int]): A list of stimulus IDs.

        Returns:
            dict: A dictionary of stimulus data with the stimulus names as keys.

        """
        out = {}
        for stim_id in stim_ids:
            info = self.get(stim_id).info
            name = info.pop("name")
            out[name] = info
        return out


class ValidationError(Exception):
    """An exception to raise when stimulus validation fails."""

    pass
