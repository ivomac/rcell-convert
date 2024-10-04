"""A module for handling units in RCell conversions.

The module provides two functions to set and convert units on RCell data.

It also provides variables that define the expected standard units:

Time: microsecond
Voltage: milliVolt
Current: nanoAmpere
Resistance: MegaOhm
Capacitance: picoFarad
"""

############################
### RCELL STANDARD UNITS ###
############################

# Standard units used in RCell
microsecond = "us"
milliVolt = "mV"
nanoAmpere = "nA"
MegaOhm = "MOhm"
picoFarad = "pF"

############################
#### PREFIX MULTIPLIERS ####
############################

PREFIX = {
    "T": 12,
    "G": 9,
    "M": 6,
    "k": 3,
    "h": 2,
    "": 0,
    "d": -1,
    "c": -2,
    "m": -3,
    "u": -6,
    "μ": -6,
    "n": -9,
    "p": -12,
    "f": -15,
}


def factor(to: str, from_unit: str) -> int | float:
    """Get unit multiplicative conversion factor.

    Args:
        to (str): The unit to convert to.
        from_unit (str): The unit that the data is currently in.

    Returns:
        int | float: The conversion factor.

    """
    to_pref, from_pref = remove_common_suffix(to, from_unit)

    exp = PREFIX[from_pref] - PREFIX[to_pref]

    return 10**exp


def remove_common_suffix(str1: str, str2: str) -> tuple:
    """Remove the longest common suffix of two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        tuple: A tuple containing the modified first and second strings
        without the common suffix.

    """
    pos = 0
    while pos < len(str1) and pos < len(str2) and str1[-pos - 1] == str2[-pos - 1]:
        pos += 1

    return str1[:-pos], str2[:-pos]
