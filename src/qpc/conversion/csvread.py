"""A module that defines a class to read and load CSV files."""

from pathlib import Path

import pandas as pd


class CSVReader:
    """A class to read CSV files and load them as common python datastructures.

    Assumes the first row (and only the first) is the header row. Assumes that
    empty cells are represented as empty strings.

    Attributes:
        file (Path): The path to the CSV file.
        sep (str): The separator used in the CSV file.
        exp_folder (Path): The folder containing the CSV file.
        file_size (int): The size of the CSV file in bytes.
        file_size_human (str):
            The size of the CSV file in human-readable format.

    Methods:
        count_lines: Count the number of lines in the CSV file.
        read_last_line: Read the last line of the CSV file.
        read_column_sizes: Read the column sizes by reading the entire file.
        as_dataframe: Read the CSV file and return it as a DataFrame.

    """

    def __init__(self, file: Path, sep: str = "\t"):
        """Initialize the CSVReader object.

        Args:
            file (Path): The path to the CSV file.
            sep (str): The separator used in the CSV file. Default is tab.

        """
        self.file = file
        if not self.file.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.file}")
        self.sep = sep
        self.exp_folder = file.parent
        self.headers = self._read_headers()

    def count_lines(self) -> int:
        """Count the number of lines in the CSV file by reading the file.

        Returns:
            int: The number of lines in the CSV file.

        """
        with self.file.open("r") as file:
            return sum(1 for _ in file) - 1

    def read_last_line(self) -> list[str]:
        """Read the last line of the CSV file.

        Returns:
            str: The last line of the CSV file.

        """
        with self.file.open("rb") as f:
            f.seek(-2, 2)
            while f.read(1) != b"\n":
                f.seek(-2, 1)
            last_line = f.readline().decode()
        return last_line.strip("\n").split(self.sep)

    def read_column_sizes(self) -> list[int]:
        """Read the column sizes by reading the entire file.

        Returns:
            list: The number of non-empty cells in each column.

        """
        with self.file.open("r") as file:
            lengths = [0] * len(self.headers)
            for i, line in enumerate(file):
                line = line.rstrip("\n").split(self.sep)
                for j, val in enumerate(line):
                    if val:
                        lengths[j] = i
        return lengths

    def as_dataframe(self, **kwargs) -> pd.DataFrame:
        """Read the CSV file and return it as a DataFrame.

        Returns:
            pd.DataFrame: The CSV file as a DataFrame.

        """
        return pd.read_csv(self.file, sep=self.sep, **kwargs)

    def _read_headers(self) -> list[str]:
        with self.file.open("r") as file:
            return file.readline().strip().split(self.sep)
