"""A module providing functions to save, load, and validate rCell data to nwb format.

Also provides a stimulus class to add stimulus information to the rCell dictionary.

The main classes are:
    - CellDB: A singleton class to find and load rCells.
    - StimCsv: A singleton class to read and parse stimulus information.
"""

from .src import dict, plot, unit
from .src.db import CellDB
from .src.io import keys, load, load_dataset, save
from .src.rcell import RCell
from .src.stimulus import StimCsv
from .src.validation import ValidationError, Validator

__all__ = [
    "dict",
    "unit",
    "plot",
    "CellDB",
    "keys",
    "load",
    "save",
    "load_dataset",
    "RCell",
    "StimCsv",
    "Validator",
    "ValidationError",
]
