"""A module for checking the structure of the rCell dictionary before conversion to nwb.

A single DictValidator class instance is created and used to validate the structure
of the rCell dictionary.
"""

import re
from datetime import datetime
from functools import cached_property
from typing import Any

import numpy as np

from . import unit
from .stimulus import StimCsv
from .utils import singleton

STIMCSV = StimCsv()


@singleton
class Validator:
    """A class for validating the structure of an rCell dictionary.

    There are no public attributes in the class, only methods, the only one of
    interest being the 'validate' method.

    This class checks the structure of an rCell and inserts default values
    where necessary. The structure of the rCell dictionary is defined in the
    'structure' method of the class. The checking/structure follows certain rules:

    - the keys of 'timeseries' and 'presentation' are variable but must be
        exactly the same and must match the protocol types

    - no missing repetitions (every 1-to-n repetitions must be present)

    - special keys that are not keys of the rCell dictionary are prefixed with '#':
        * '#optional' indicates the key is not required
        * '#type' defines the type of the value
        * '#default' defines the default value if the key is missing
        * '#pattern' is a regex pattern that string types must match
        * '#date_pattern' is a datetime pattern that string types must match

    - np.ndarray types can have additional keys:
        * '#dims' defines the number of dimensions of the array
        * '#shape' is a tuple of callables to compare with the shape of the array
        * '#element' defines the type of the elements of the array
            should be a numpy dtype

    Possible improvements:
        Add a '#deprecated' key containing a string which is printed as a warning about
        the usage of that key and with instructions on what to use instead.

    """

    def __init__(self):
        """Initialize the Validator class."""
        self._prot_types: list[str] | None = None

        self.protocol: str
        self.repetition: str
        self.root: dict

        return

    def validate(self, root: dict):
        """Validate the structure of the rCell dictionary.

        This function first navigates the reference structure recursively to
        check for missing keys and insert default values where necessary.

        It then checks the stimulus and acquisition keys to ensure they are the
        same and match the protocol types.

        It then navigates the input dictionary to check their contents against
        the expected types/shapes and other constraints.

        Args:
            root: The rCell dictionary to validate.

        """
        self.root = root

        for key in self.structure:
            self.navigate_ref(self.root, self.structure, key)

        self.check_protocol_types()

        for key in self.root:
            self.navigate(self.root, self.structure, key)

        return

    def check_protocol_types(self):
        """Check match between 'stimulus/presentation' and 'acquisition/timeseries'.

        Raises:
            ValidationError: If the keys are not the same.

        """
        stim_protocols = set(self.root["stimulus"]["presentation"].keys())
        acq_protocols = set(self.root["acquisition"]["timeseries"].keys())

        if stim_protocols != acq_protocols:
            raise ValidationError(
                "The keys of 'stimulus/presentation' and 'acquisition/timeseries'"
                + f" must be exactly the same. Got {stim_protocols=}, {acq_protocols=}."
            )
        return

    def navigate_ref(self, node: dict, ref: dict, ref_key: str):
        """Navigate the reference structure recursively.

        Checks for missing keys and inserts default values where necessary.

        Args:
            node: The input dictionary.
            ref: The reference dictionary.
            ref_key: The key to navigate.

        Raises:
            ValidationError: If a key is missing and not optional.

        """
        # Key in reference dict but not in input dict
        if ref_key not in node:
            # Fill in default value if it exists
            if "#default" in ref[ref_key]:
                node[ref_key] = ref[ref_key]["#default"]

            # Raise error if key is not optional
            elif not ref[ref_key].get("#optional", False):
                raise ValidationError(f"Key {ref_key} missing from dictionary.")

        # Key is in both reference and input dict
        # Any keys not starting with '#' should be navigated further
        else:
            node = node[ref_key]
            ref = ref[ref_key]

            for key in ref:
                # Special keys are prefixed with '#'
                if not key.startswith("#"):
                    # recurse
                    self.navigate_ref(node, ref, key)

        return

    def navigate(self, node: dict, ref: dict, key: str):
        """Navigate the input dictionary.

        Checks their contents against the expected types/shapes and other constraints.
        Also checks that all 1-to-n repetitions are present.

        While navigating, the repetition and protocol keys are stored to be used
        later to check the n_points and sweep_count.

        Args:
            node: The input dictionary.
            ref: The reference dictionary.
            key: The key to navigate.

        Raises:
            ValidationError: If a key is not valid or has the wrong type.

        """
        if key not in ref:
            raise ValidationError(f"Key {key} is not valid.")

        node = node[key]
        ref = ref[key]

        if key == "repetitions":
            self.check_repetitions(node)

        if "#type" in ref:
            self.check_type(node, ref, key)
        else:
            for rkey in node:
                # Store the repetition and protocol keys
                # to be used later to check the n_points and sweep_count
                if key == "repetitions":
                    self.repetition = rkey
                elif key == "timeseries":
                    self.protocol = rkey

                self.navigate(node, ref, rkey)

        return

    def check_repetitions(self, node: dict):
        """Check that all 1-to-n repetitions are present.

        Args:
            node: The input dictionary.

        """
        rep_ids = sorted(int(rep[10:]) for rep in node)

        if any(i != ri for i, ri in enumerate(rep_ids, start=1)):
            raise ValidationError(
                "Invalid repetition IDs. Every 1-to-n repetitions must be present."
                + f" Got {rep_ids=}."
            )

        return

    def check_type(self, node: Any, ref: Any, key: str):
        """Check the type of the value against the reference.

        Args:
            node: The value to check.
            ref: The reference dictionary.
            key: The key of the value.

        Raises:
            ValidationError: If the value has the wrong type
                or does not match the pattern
                or does not match the date pattern.

        """
        if not isinstance(node, ref["#type"]):
            raise ValidationError(
                f"Key {key} has the wrong type."
                + f" Expected {ref['#type']}, got {type(node)}."
            )

        if isinstance(node, np.ndarray):
            self.check_array(node, ref, key)

        else:
            if "#pattern" in ref and not re.match(ref["#pattern"], node):
                raise ValidationError(
                    f"Value of {key} ({node}) does not match"
                    + f" the pattern {ref['#pattern']}."
                )

            if "#date_pattern" in ref:
                date_pattern = ref["#date_pattern"]
                try:
                    datetime.strptime(node, date_pattern)
                except ValueError as err:
                    raise ValidationError(
                        f"Value of {key} ({node}) does not match"
                        + f" the date pattern {date_pattern}."
                    ) from err

        return

    def check_array(self, node: np.ndarray, ref: dict, key: str):
        """Check the type, shape, and element type of the array against the reference.

        Args:
            node: The array to check.
            ref: The reference dictionary.
            key: The key of the array.

        Raises:
            ValidationError: If the array has the wrong number of dimensions
                or the wrong shape or the wrong element type
                or the elements do not match the pattern
                or the elements do not match the date pattern.

        """
        if "#dims" in ref and len(node.shape) != ref["#dims"]:
            raise ValidationError(
                f"Key {key} has the wrong number of dimensions."
                + f" Expected {ref['#dims']}, got {len(node.shape)}."
            )

        if "#shape" in ref:
            # Note: ref["#shape"] is a tuple of callables
            # since the shape of the array is not known in advance
            # We need to call the functions to get the expected shape
            shape = tuple(sh() for sh in ref["#shape"])
            if shape != node.shape:
                raise ValidationError(
                    f"Key {key} has the wrong shape."
                    + f" Expected {shape}, got {node.shape}."
                )

        if "#element" in ref and not np.issubdtype(node.dtype, ref["#element"]):
            raise ValidationError(
                f"Array {key} has the wrong element type."
                + f" Expected {ref['#element']}, got {node.dtype}."
            )

        if "#pattern" in ref and not all(
            re.match(ref["#pattern"], val) for val in node.flat
        ):
            raise ValidationError(
                f"Array {key} has elements that do not match"
                + f" the pattern {ref['#pattern']}."
            )

        if "#date_pattern" in ref:
            date_pattern = ref["#date_pattern"]
            for val in node.flat:
                try:
                    datetime.strptime(val, date_pattern)
                except ValueError as err:
                    raise ValidationError(
                        f"Array {key} has element ({val}) that does not match"
                        + f" the date pattern {date_pattern}."
                    ) from err

        return

    def n_points(self) -> int:
        """Get the number of points for the current repetition and protocol."""
        return self.root["acquisition"]["timeseries"][self.protocol]["repetitions"][
            self.repetition
        ]["n_points"][0]

    def sweep_count(self) -> int:
        """Get the number of sweeps for the current protocol."""
        return self.root["stimulus"]["presentation"][self.protocol]["sweep_count"]

    @cached_property
    def structure(self) -> dict:
        """Initialization of the reference structure."""
        sweep_count = self.sweep_count
        n_points = self.n_points

        protocol_types = STIMCSV.data["name"].unique().tolist()

        repetition = {
            "#optional": True,
            "amp": {
                "#optional": True,
                "#type": dict,
            },
            "pharmacology": {
                "#optional": True,
                "#type": dict,
            },
            "capacitance_fast": {
                "#optional": True,
                "#type": np.ndarray,
                "#element": np.floating,
                "#unit": unit.picoFarad,
                "#dims": 1,
                "#shape": [sweep_count],
            },
            "capacitance_slow": {
                "#optional": True,
                "#type": np.ndarray,
                "#element": np.floating,
                "#unit": unit.picoFarad,
                "#dims": 1,
                "#shape": [sweep_count],
            },
            "data": {
                "#type": np.ndarray,
                "#element": np.floating,
                "#unit": unit.nanoAmpere,
                "#dims": 2,
                "#shape": [n_points, sweep_count],
            },
            "head_temp": {
                "#default": np.nan,
                "#type": float,
            },
            "n_points": {
                "#type": np.ndarray,
                "#element": np.integer,
                "#dims": 1,
                "#shape": [sweep_count],
            },
            "r_membrane": {
                "#optional": True,
                "#type": np.ndarray,
                "#element": np.floating,
                "#unit": unit.MegaOhm,
                "#dims": 1,
                "#shape": [sweep_count],
            },
            "r_series": {
                "#optional": True,
                "#type": np.ndarray,
                "#element": np.floating,
                "#unit": unit.MegaOhm,
                "#dims": 1,
                "#shape": [sweep_count],
            },
            "seal": {
                "#optional": True,
                "#type": np.ndarray,
                "#element": np.floating,
                "#unit": unit.MegaOhm,
                "#dims": 1,
            },
            "time": {
                "#optional": True,
                "#description": "Time axis of the data for each sweep in microseconds",
                "#type": np.ndarray,
                "#element": np.integer,
                "#unit": unit.microsecond,
                "#dims": 1,
                "#shape": [n_points],
            },
            "trace_times": {
                "#optional": True,
                "#description": "Starting time of each sweep from the beginning"
                + "of the experiment in microseconds",
                "#type": np.ndarray,
                "#element": np.integer,
                "#unit": unit.microsecond,
                "#dims": 1,
                "#shape": [sweep_count],
            },
            "v_offset": {
                "#type": float,
                "#default": np.nan,
            },
            "x_interval": {
                "#description": "time between each point in microseconds",
                "#type": int,
            },
            "x_start": {
                "#optional": True,
                "#description": "Start time of each sweep in microseconds",
                "#type": np.ndarray,
                "#element": np.integer,
                "#unit": unit.microsecond,
                "#shape": [sweep_count],
            },
        }

        reps = {
            "repetitions": {f"repetition{i}": repetition for i in range(1, 1000)},
            "#optional": True,
        }

        timeseries = {k: reps for k in protocol_types}

        acquisition = {
            "images": {
                "#optional": True,
                "#type": dict,
            },
            "timeseries": timeseries,
        }

        general = {
            "code_info": {
                "#optional": True,
                "#type": dict,
            },
            "heka": {
                "#optional": True,
                "#type": dict,
            },
            "nanion": {
                "#optional": True,
                "#type": dict,
            },
            "cell_id": {
                "#type": int,
                "#default": 0,
            },
            "cell_info": {
                "cell_countpml": {
                    "#type": int | str,
                    "#default": "0k",
                },
                "chip_cols": {
                    "#optional": True,
                    "#type": str,
                    "#default": "",
                },
                "culture_medium": {
                    "#optional": True,
                    "#type": str,
                    "#default": "",
                },
                "cell_image": {
                    "#type": str,
                    "#default": "",
                },
                "cell_stock_id": {
                    "#type": str,
                    "#default": "",
                },
                "cell_suspension_medium": {
                    "#type": str,
                    "#default": "",
                },
                "host_cell": {
                    "#type": str,
                    "#default": "",
                },
                "passage": {
                    "#type": str,
                    "#default": "",
                },
                "species": {
                    "#type": str,
                },
            },
            "channel_info": {
                "host_cell": {
                    "#type": str,
                    "#default": "",
                },
                "ion_channel": {
                    "#type": str,
                    "#default": "",
                },
                "species": {
                    "#type": str,
                    "#default": "",
                },
            },
            "data_quality_notes": {
                "#type": str,
                "#default": "",
            },
            "drn": {
                "#date_pattern": "%Y.%m.%d",
                "#description": "Date of recording in yyyy.mm.dd format",
                "#type": str,
            },
            "experiment": {
                "comment": {
                    "#default": "",
                    "#type": str,
                },
                "date": {
                    "#date_pattern": "%Y.%m.%d",
                    "#description": "Date of experiment in yyyy.mm.dd format",
                    "#type": str,
                },
                "doxycycline_conc": {
                    "#type": str,
                    "#default": "",
                },
                "ec_id": {
                    "#type": str,
                    "#default": "",
                },
                "ec_solution": {
                    "#type": str,
                    "#default": "",
                },
                "ic_id": {
                    "#type": str,
                    "#default": "",
                },
                "ic_solution": {
                    "#type": str,
                    "#default": "",
                },
                "induction": {
                    "#type": int | str,
                    "#default": 24,
                },
                "induction_medium": {
                    "#type": str,
                    "#default": "",
                },
                "manufacturer": {
                    "#default": "",
                    "#type": str,
                },
                "model_name": {
                    "#default": "",
                    "#type": str,
                },
                "nanioncsv_log": {
                    "#type": str,
                    "#default": "",
                },
                "project_id": {
                    "#type": str,
                    "#default": "P0013",
                },
                "project_name": {
                    "#type": str,
                    "#default": "Channelome",
                },
                "se_id": {
                    "#type": str,
                    "#default": "",
                },
                "se_solution": {
                    "#type": str,
                    "#default": "",
                },
                "temp": {
                    "#type": str,
                    "#default": "rt",
                },
                "time": {
                    "#date_pattern": "%H:%M:%S",
                    "#description": "Time of experiment in HH:MM:SS format",
                    "#type": str,
                },
                "total_cells": {
                    "#type": float,
                    "#default": np.nan,
                },
                "trypsin_concentration": {
                    "#type": str,
                    "#default": "",
                },
                "trypsinization_time": {
                    "#type": int,
                    "#default": 60,
                },
            },
            "experimenter": {
                "experimenter": {
                    "#type": str,
                    "#default": "",
                },
                "user_email": {
                    "#type": str,
                    "#default": "",
                },
                "user_initials": {
                    "#type": str,
                    "#default": "",
                },
            },
            "institution": {
                "#type": str,
                "#default": "Ecole polytechnique federale de Lausanne (EPFL)",
            },
            "lab": {
                "#type": str,
                "#default": "Blue Brain Project (BBP)",
            },
            "session_id": {
                "#type": int | str,
                "#default": 0,
            },
        }

        stim_dict = {
            "#optional": True,
            "command": {
                "#type": str,
                "#default": "",
            },
            "stim_id": {
                "#type": int,
            },
            "sweep_count": {
                "#type": int,
            },
            "sweep_interval": {
                "#type": int,
                "#default": 0,
            },
            "type": {
                "#type": str,
                "#default": "",
            },
        }

        stimulus = {"presentation": {k: stim_dict for k in protocol_types}}

        return {
            "analysis": {
                "#optional": True,
                "#type": dict,
            },
            "epochs": {
                "#optional": True,
                "#type": dict,
            },
            "data_release": {
                "#date_pattern": "%Y.%m",
                "#description": "The date of data creation in yyyy.mm format",
                "#type": str,
            },
            "file_create_date": {
                "#date_pattern": "%d-%b-%Y %H:%M:%S",
                "#description": "The time of rCell creation in dd-mm-yyyy HH:MM:SS",
                "#type": str,
            },
            "identifier": {
                "#description": "Name of the experiment usually",
                "#type": str,
                "#default": "",
            },
            "session_description": {
                "#optional": True,
                "#type": str,
            },
            "acquisition": acquisition,
            "general": general,
            "stimulus": stimulus,
        }


class ValidationError(Exception):
    """An exception raised when a validation error occurs."""

    pass
