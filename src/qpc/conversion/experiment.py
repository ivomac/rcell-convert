"""Module to handle experiment data files and output files."""

import re
from pathlib import Path

from . import csvread as csv
from . import xmlread as xml
from .config import CONFIG
from .google_sheet import GOOGLE_SHEET as GS


class Experiment:
    """Class to handle experiment data files and output files.

    Attributes:
        session (str): The session name.
        job_id (int): The job id.
        folder (Path): The experiment folder.
        raw (csv.CSVReader): The raw data reader.
        meta (csv.CSVReader): The metadata reader.
        xml (xml.XmlFolder): The XML metadata handler.
        out (Path): The nwb output filepath.

    """

    def __init__(
        self,
        session_or_job_id: str | int,
    ):
        """Initialize the Experiment class.

        Args:
            session_or_job_id (str | int): The session name or job id.

        """
        if isinstance(session_or_job_id, str):
            self.session = session_or_job_id
        elif isinstance(session_or_job_id, int):
            job_id = session_or_job_id
            self.session = GS["Experiment"].at[job_id, "session"]

        year, month, job_id = self._parse_session()
        self.folder = self._set_folder(year, month, job_id)
        self.job_id = int(job_id)

        self.out = (CONFIG.dir.output / self.session).with_suffix(".nwb")

        self.xml = xml.XmlFolder(self.folder / f"qpc_job{self.job_id}")

        raw_data_file = self.folder / f"Exp{self.job_id}.ogw"
        self.raw = csv.CSVReader(raw_data_file)

        meta_file = self.folder / f"Exp{self.job_id}_acquisition_metadata.ogw"
        self.meta = csv.CSVReader(meta_file)

    def _parse_session(self) -> tuple[str, str, str]:
        # session are in the format qpcYYMMDD_jobid
        regex = r"qpc(\d{2})(\d{2})\d{2}_(\d+)"
        match = re.match(regex, self.session)
        if not match:
            raise ValueError(f"Could not parse session name: {self.session}")
        year, month, job_id = match.groups()
        return year, month, job_id

    def _set_folder(self, year: str, month: str, job_id: str) -> Path:
        folder = CONFIG.dir.data / f"20{year}-{month}" / f"Exp{job_id}"
        if not folder.is_dir():
            raise FileNotFoundError(f"Experiment folder not found: {folder}")
        return folder
