"""Module to setup global variables."""

from dataclasses import dataclass
from os import environ as env
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConfigDirs:
    """Directory paths for conversion."""

    root: Path
    local: Path
    data: Path
    output: Path


@dataclass
class ConfigGoogleSheet:
    """Google Sheet configuration."""

    url: str
    file: Path


@dataclass
class Config:
    """Global configuration variables."""

    dir: ConfigDirs
    gs: ConfigGoogleSheet


def set_config():
    """Set the global configuration variables from the .env file in the root folder.

    Config structure:
        dirs: ConfigDirs
            root: Path to the channelome root folder
            local: Path to the qpc_conversion root folder
            data: Path to the raw data folder
            output: Path to the NWB output folder
                The NWB files in this folder are used as reference when
                testing the conversion process.
        gs: ConfigGoogleSheet
            url: str with the Google Sheet URL
            file: Path to the local copy of the Google Sheet

    Assumes local input/output if paths not set.
    Google Sheet file is saved in the data folder if not set.

    Returns:
        Config: The global configuration variables.

    Raises:
        FileNotFoundError: If the data folder is not found.
        ValueError: If the Google Sheet URL is not set

    """
    # ConfigDirs
    file = Path(__file__).resolve()
    root = file.parents[3]
    local = file.parents[1]
    data = env.get("QPC_RAW_DATA_PATH", root / "data" / "raw_data")
    output = env.get("QPC_NWB_PATH", root / "data" / "nwb")

    data = Path(data)
    output = Path(output)

    if not data.is_dir():
        raise FileNotFoundError(f"Data folder {data} not found.")

    output.mkdir(exist_ok=True, parents=True)

    dirs = ConfigDirs(root, local, data, output)

    # ConfigGoogleSheet
    url = env.get("QPC_GOOGLE_SHEET_URL")
    if not url:
        raise ValueError("QPC_GOOGLE_SHEET_URL is not set.")

    file = env.get("QPC_GOOGLE_SHEET_PATH", dirs.data / "Sophion_Experiment.xlsx")

    file = Path(file)

    gs = ConfigGoogleSheet(url, file)

    return Config(dirs, gs)


CONFIG = set_config()
