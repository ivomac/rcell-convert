# QPC to rCell Conversion

#### A python module to convert QPC data to NWB format

## How to use

The module can be run as a standalone script as:

`python -m /path/to/qpc_folder <script_name>`

After installation of the channelome package, it can also be run from anywhere in the (virtual) environment as:

`qpc <script_name>`

For example, we can create the rCell of experiment 272, and replace the existing one:

```bash
$ qpc rcell_create --overwrite 272
creating path/to/out/qpc240515_272.nwb
```

There are in 3 scripts available with this module.
Run `python -m qpc` or `qpc` to see the available scripts.

## Setup

#### Environment variables

Set the following environment variables in a .env file to configure the module:

* QPC_RAW_DATA_PATH:     path to the raw data files, containing dated folders.
* QPC_NWB_PATH:          path where the NWB files will be saved.
* QPC_GOOGLE_SHEET_PATH: path to the xlsx Google Sheet file.
* QPC_GOOGLE_SHEET_URL:  URL of the Google Sheet.

## Code structure

Most submodules simply define a single class to handle an aspect of the data processing.

Dependency tree:
```
rcell: Create rCell files for a given Experiment.
│
├──>experiment: Handle experiment data files and output files.
│   │
│   ├──>csvread: Read the CSV files with the experiment raw data/meta data.
│   │
│   ├──>xmlread: Read the XML files/folder with additional experiment metadata.
│   │
├───┼──>google_sheet: Handle the Google Spreadsheet with the experiment summaries.
│   │   │
└───┴───┴──>config: Handle all environment variables and configuration.
```

Note that a single `GoogleSheet` instance is created when `google_sheet` is imported to avoid loading the spreadsheet more than once.

