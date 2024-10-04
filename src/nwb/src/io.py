"""I/O functions for transforming nested dictionaries to/from HDF5 files."""

from pathlib import Path

import h5py
import numpy as np

from .validation import Validator

VALIDATOR = Validator()


def save(
    file_path: str | Path,
    nest: dict,
    overwrite: bool = False,
    validate: bool = False,
):
    """Save nested dictionary to HDF5 file.

    Args:
        file_path (str | Path): Path to the file.
        nest (dict): Nested dictionary to save.
        overwrite (bool, optional): Overwrite the file if it exists.
            Defaults to False.
        validate (bool, optional): Validate the dictionary before saving.
            Defaults to False.

    """
    file_path = Path(file_path)

    if validate:
        VALIDATOR.validate(nest)

    if overwrite or not file_path.exists():
        with h5py.File(file_path, "w") as h5file:
            recurse_save(h5file, [""], nest)

    return


def recurse_save(h5file: h5py.File, path: list[str], nest: dict):
    """Recursively save the contents of a dictionary to an h5py file.

    Args:
        h5file (h5py.File): H5py file object.
        path (list[str]): List of keys to the current group.
        nest (dict): Dictionary to save.

    """
    for key, data in nest.items():
        path.append(key)
        st = "/".join(path)

        if isinstance(data, dict):
            h5file.create_group(st)
            recurse_save(h5file, path, data)
        else:
            # Strings are stored and read as bytes, need to be encoded
            # See https://github.com/h5py/h5py/issues/1769
            if isinstance(data, np.ndarray) and data.dtype.kind == "U":
                data = data.astype("S")

            kwargs = {"data": data}

            if key == "data":
                kwargs.update(
                    {
                        "chunks": data.shape,
                        "compression": "gzip",
                        "compression_opts": 6,
                    }
                )

            h5file.create_dataset(st, **kwargs)

        path.pop()


def load(file_path: str | Path, root: str | Path = "") -> dict:
    """Load nested dictionary from HDF5 file.

    Args:
        file_path (str | Path): Path to the file.
        root (str, optional): Root group to load.

    Returns:
        dict: Nested dictionary.

    Raises:
        ValueError: If the root is not a group.

    """
    file_path = Path(file_path)
    root = str(root)

    with h5py.File(file_path, "r") as h5file:
        if root:
            h5file = h5file[root]
        if not isinstance(h5file, h5py.Group):
            raise ValueError(f"{root} is {type(root)}. Use get() to load datasets.")

        return recurse_load([root], h5file)


def recurse_load(path: list[str], nest: h5py.Group) -> dict:
    """Recursively load the contents of an h5py file to a dictionary.

    Args:
        path (list[str]): List of keys to the current group.
        nest (h5py.Group): H5py group object.

    Returns:
        dict: Loaded dictionary.

    Raises:
        ValueError: If an unknown type is encountered.

    """
    data = {}

    for key in nest.keys():
        path.append(key)

        st = "/".join(path)

        val = nest[st]

        if isinstance(val, h5py.Group):
            data[key] = recurse_load(path, val)
        elif isinstance(val, h5py.Dataset):
            data[key] = load_dataset(val)
        else:
            raise ValueError("Unknown type:", type(val))

        path.pop()

    return data


def load_dataset(dataset: h5py.Dataset):
    """Convert a dataset correctly to pass validation.

    When loading data from an HDF5 file, the data types are not always
    equal to the original data types. This function converts the data
    to the correct types that passed validation.

    Examples:
        * Loaded dataset are usually numpy scalars or arrays, and strings
          are always in byte format.
        * Numpy scalars are converted to python scalars, and strings are
          decoded to utf-8.
        * Some old rCells have arrays with singleton dimensions, and some
          scalars were stored as 0D arrays. These are fixed here for
          compatibility with older datasets.

    Args:
        dataset (h5py.Dataset): Dataset from an HDF5 file.

    Returns:
        Converted data in its appropriate type.

    Raises:
        ValueError: If an unexpected dataset type is encountered.

    """
    val = dataset[()]

    # Strings are stored and read as bytes, need to be decoded
    # See https://github.com/h5py/h5py/issues/1769
    if isinstance(val, bytes):
        # Decode bytes
        val = val.decode("utf-8")

    # numpy scalar or array
    elif isinstance(val, np.generic | np.ndarray):
        if val.dtype.kind in ["S", "O"]:
            # Decode bytes
            val = val.astype(str)

        # Some old data has singleton dimensions for no reason, remove
        if len(val.shape) > 1 and 1 in val.shape:
            val = np.squeeze(val)

        # Convert numpy scalars to python scalars
        # Also, some scalars were stored as 0D arrays in old data
        if isinstance(val, np.generic) or val.size == 1:
            val = val.item()
    else:
        raise ValueError(
            f"Unexpected dataset type encountered while loading: {type(val)}"
        )

    return val


def get(path: str | Path, key: str | Path = ""):
    """Get a dataset from an HDF5 file.

    Args:
        path (str | Path): Path to the file.
        key (str | Path): Key to the dataset.

    Returns:
        Any: Data from the dataset.

    Raises:
        ValueError: If the key is not a dataset.

    """
    path = Path(path)
    key = str(key)

    with h5py.File(path, "r") as h5file:
        if key:
            h5file = h5file[key]
        if not isinstance(h5file, h5py.Dataset):
            raise ValueError(f"{key} is {type(key)}. Use load() to load groups.")

        return load_dataset(h5file)


def keys(path: str | Path, root: str | Path = "") -> list[str]:
    """Get all keys in a group of an HDF5 file.

    Args:
        path (str | Path): Path to the file.
        root (str | Path, optional): Root group to load.

    Returns:
        list[str]: List of keys in the group.

    Raises:
        ValueError: If the root is not a group.

    """
    path = Path(path)
    root = str(root)

    with h5py.File(path, "r") as h5file:
        if root:
            h5file = h5file[root]

        if not isinstance(h5file, h5py.Group):
            raise ValueError(f"{root} is not a group.")

        return list(h5file.keys())
