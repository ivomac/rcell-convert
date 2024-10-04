"""Module to handle the Google Spreadsheet with the experiment summaries."""

from functools import cached_property

import pandas as pd
import requests

from .config import CONFIG

global GOOGLE_SHEET


class GoogleSheet:
    """Class to handle the Google Spreadsheet with the experiment summaries.

    To avoid the creation of multiple instances of the GoogleSheet class, an
    instance is created and stored in the global variable GOOGLE_SHEET. It will
    lazily load the data from the file when needed.

    Use the update method to download the latest version of the Spreadsheet.

    Attributes:
        url (str): The URL of the Google Spreadsheet.
        download_url (str): The URL to download the Google Spreadsheet.
        file (Path): The path to the local copy of the Google Spreadsheet.
        df (pd.DataFrame): The DataFrame with the Google Spreadsheet data.

    Usage:
        GOOGLE_SHEET["Experiment"] is the Experiment sheet as a DataFrame.
        GOOGLE_SHEET["Solution"] is the Solution sheet as a DataFrame.

    """

    def __init__(self):
        """Initialize the GoogleSheet class."""
        self.file = CONFIG.gs.file

        self.url = CONFIG.gs.url
        self.download_url = self.url + "/export?format=xlsx"

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Get the sheet as a DataFrame.

        The sheets are lazily loaded from the file.

        Args:
            key (str): The sheet name.

        Returns:
            pd.DataFrame: The sheet as a DataFrame.

        """
        return self.sheet[key]

    @property
    def sessions(self) -> list[str]:
        """List of session names."""
        return self["Experiment"]["session"].tolist()

    @property
    def job_ids(self) -> list[int]:
        """List of job IDs."""
        return self["Experiment"].index.tolist()

    @cached_property
    def sheet(self) -> dict[str, pd.DataFrame]:
        """Load the Google Spreadsheet as a dictionary of DataFrames."""
        out = {}

        sheets = [
            {
                "sheet_name": "Experiment",
                "index_col": 2,
                "na_filter": False,
            },
            {
                "sheet_name": "Solution",
                "index_col": [0, 1],
                "na_filter": False,
            },
        ]

        for sheet in sheets:
            name = sheet["sheet_name"]
            out[name] = pd.read_excel(
                self.file,
                **sheet,
            )
            out[name].sort_index(inplace=True)
            # verify index is unique
            if not out[name].index.is_unique:
                raise IndexError(f"GS {name} index is not unique")

        return out

    def update(self):
        """Download the Google Spreadsheet."""
        res = requests.get(self.download_url, timeout=10)
        # check if the request was successful
        res.raise_for_status()
        self.file.write_bytes(res.content)
        if hasattr(self, "sheet"):
            del self.sheet
        return


GOOGLE_SHEET = GoogleSheet()
