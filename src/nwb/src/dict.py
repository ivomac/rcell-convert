"""Dictionary utilities to transform the nested rcell dictionaries."""

from typing import Any


def flatten(nest: dict) -> dict:
    """Flatten a nested dictionary.

    Convert `{k1: {k2: {k3: v}}}` to `{(k1, k2, k3): v}`.

    Args:
        nest (dict): Nested dictionary to flatten.

    Returns:
        dict: Flattened dictionary (tuple of keys to leaf value).

    """
    out = {}

    def recurse_flatten(sub_nest: dict, *keys: str):
        for key, val in sub_nest.items():
            key_tuple = keys + (key,)

            if isinstance(val, dict):
                recurse_flatten(val, *key_tuple)
            else:
                out[key_tuple] = val

    recurse_flatten(nest)

    return out


def pad_keytuples(
    flat: dict[tuple, Any],
    depth: int,
    filler: str = "",
    left=False,
) -> dict:
    """Pad key tuples in a flat dictionary to a certain depth.

    Converts `{(k1, k2): v1, (k1, k2, k3): v2}` to
    `{(k1, k2, filler): v1, (k1, k2, k3): v2}` (if depth=3).

    If the key tuple is shorter than the desired depth, it will be padded
    with the filler string. Otherwise, it will be left as is.

    Args:
        flat (dict): Flat dictionary.
        depth (int): Desired depth of key tuples.
        filler (str, optional): Filler string to pad key tuples. Defaults to "".
        left (bool, optional): Whether to pad on the left or right. Defaults to False.

    Returns:
        dict[tuple, Any]: Flattened dictionary with padded keys.

    """
    out = {}

    for keys, val in flat.items():
        size = depth - len(keys)
        if size > 0:
            keys = (filler,) * size + keys if left else keys + (filler,) * size

        out[keys] = val

    return out


def nest(flat: dict) -> dict:
    """Nest a flattened dictionary.

    Converts `{(k1, k2, k3): v}` to `{k1: {k2: {k3: v}}}`.

    Args:
        flat (dict): Flattened dictionary to nest.

    Returns:
        dict: Nested dictionary.

    """
    out = {}

    for keys, val in flat.items():
        nest = out

        for key in keys[:-1]:
            nest = nest.setdefault(key, {})

        nest[keys[-1]] = val

    return out
