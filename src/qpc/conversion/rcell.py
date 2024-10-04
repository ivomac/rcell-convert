"""Module to create rCell dictionaries and nwb files from the data of an experiment."""

from datetime import datetime

import numpy as np
from tabulate import tabulate

import nwb

from .experiment import Experiment
from .google_sheet import GOOGLE_SHEET as GS

now = datetime.now()

# Sampling rate is 10 kHz = 10 ms^-1 = 1e-2 us^-1
# The time interval between samples is 1/sampling_rate = 100 us
X_INTERVAL = 100  # us

STIMCSV = nwb.StimCsv()


class RCell:
    """Class to create rCell dictionaries and nwb files from the data of an experiment.

    Attributes:
        Exp (Experiment): The Experiment object corresponding to the session or job_id.

    """

    def __init__(self, session_or_job_id: str | int):
        """Initialize the rCell object.

        Args:
            session_or_job_id (str | int): The session or job_id of the experiment.

        """
        self.Exp = Experiment(session_or_job_id)
        self.drop = True
        self.keep_reps = 0

        self.messages = []

        return

    def create(
        self,
        overwrite: bool = False,
        drop: bool = True,
        keep_reps: int = 0,
    ) -> str:
        """Create the rCell dictionary and nwb file.

        Args:
            overwrite (bool, optional): Replace the nwb file if it already exists.
                Defaults to False.
            drop (bool, optional): Drop all runs starting from the first incomplete run.
                Some experiments may have incomplete runs, that is, runs with less
                sweeps than expected. Defaults to False.
            keep_reps (int, optional): Keep this number of repetitions per protocol.

        Returns:
            str: A report on the rCell dictionary.

        """
        self.drop = drop
        self.keep_reps = keep_reps

        out_path = self.Exp.out

        report = ""
        if overwrite or not out_path.is_file():
            counts, rcell_dict = self.create_dict()
            nwb.save(
                out_path,
                rcell_dict,
                overwrite=True,
                validate=True,
            )
            report = self.dict_report(rcell_dict, counts)
        return report

    def create_dict(self) -> tuple[dict, dict]:
        """Create the rCell dictionary.

        Returns:
            dict: The rCell dictionary.

        """
        protocol_map = self.get_protocol_map()
        protocol_runs = self.get_protocol_runs(protocol_map)
        counts, acquisition = self.group_protocol_runs(protocol_runs, protocol_map)

        if not acquisition:
            raise ValueError(
                "No valid protocol runs found. Preventing creation of empty rCell."
            )

        prot_ids = [id for id, v in counts.items() if v["kept"]]
        stimulus = STIMCSV.info(prot_ids)

        general = self.get_general_dict()

        now = datetime.now()

        rcell_dict = {
            "data_release": now.strftime("%Y.%m"),
            "file_create_date": now.strftime("%d-%b-%Y %H:%M:%S"),
            "acquisition": {
                "timeseries": acquisition,
            },
            "general": general,
            "stimulus": {
                "presentation": stimulus,
            },
        }

        return counts, rcell_dict

    def dict_report(self, rcell_dict: dict, counts: dict) -> str:
        """Print a report on the rCell dictionary.

        Args:
            rcell_dict (dict): The rCell dictionary.
            counts (dict): The protocol counts (how many times seen/kept).

        """
        job_id = self.Exp.job_id
        date = rcell_dict["file_create_date"]

        title = f"Job ID {job_id:03d}          {date}"

        headers = ["Protocol", "ID", "sweeps", "runs seen", "#", "runs kept", "#"]
        rows = [
            [
                v["type"],
                f"{prot_id:03d}",
                v["sweeps"],
                " ".join(f"{a:02d}" for a in v["seen"]),
                len(v["seen"]),
                " ".join(f"{a:02d}" for a in v["kept"]),
                len(v["kept"]),
            ]
            for prot_id, v in counts.items()
        ]

        table = tabulate(rows, headers=headers, tablefmt="rst")

        report = f"\n{title}\n{table}\n"

        if self.messages:
            messages = "\n  ".join(self.messages)
            report += "\nLog:\n"
            report += f"  {messages}\n"

        return report

    def get_general_dict(self) -> dict:
        """Create the 'general' dictionary with experiment metadata.

        Returns:
            dict: The general dictionary.

        Structure of the general dictionary:

            cell_info:
                cell_stock_id            (string)
                cell_suspension_medium   (string)
                host_cell                (string)
                passage                  (string)
                species                  (string)

            channel_info:
                host_cell                (string)
                ion_channel              (string)
                species                  (string)

            drn                          (string)

            experiment:
                comment                  (string)
                date                     (string)
                ec_id                    (string)
                ic_id                    (string)
                induction                (int)
                manufacturer             (string)
                model_name               (string)
                se_id                    (string)
                temp                     (string)
                time                     (string)

            experimenter:
                user_initials            (string)

        """
        # XML METADATA

        def parse_datetime(dt: str) -> tuple[str, str]:
            dt_obj = datetime.strptime(
                dt,
                "%m/%d/%Y %H:%M:%S",
            )

            date = dt_obj.strftime("%Y.%m.%d")
            time = dt_obj.strftime("%H:%M:%S")

            return date, time

        xml_map = {
            "t_chipunit_applied": [
                ("CHIP_BARCODE", "amplifier_chip_id"),
            ],
            "t_chipunit_summary": [
                ("CSLOW_PICO_F", "amplifier_c_slow"),
                ("RSERIES_MEGA_OHM", "amplifier_r_series"),
                ("VOFFSET_MILLI_V", "amplifier_v_offset"),
            ],
            "t_job": [
                ("TIMEOFEXECUTION", "session_datetime", parse_datetime),
            ],
        }

        xml = {}
        for xml_file, it in xml_map.items():
            xml_file = self.Exp.xml.read(xml_file)

            for tag, key, *parser in it:
                parser = parser[0] if parser else None
                xml[key] = xml_file.tag("ROW", tag, parser=parser)

        date, time = xml["session_datetime"]

        gs_data = GS["Experiment"].loc[self.Exp.job_id, :]

        solution = {
            k: GS["Solution"].at[(k, gs_data[f"medium_{k}_batch"]), "solution"]
            for k in ["ec", "ic", "se"]
        }

        return {
            "cell_info": {
                "cell_stock_id": str(gs_data["vial_id"]),
                "cell_suspension_medium": gs_data["medium_cell_suspension"],
                "host_cell": gs_data["host_cell"],
                "passage": gs_data["passage"],
                "species": gs_data["species"],
            },
            "channel_info": {
                "host_cell": gs_data["host_cell"],
                "ion_channel": gs_data["ion_channel"],
                "species": gs_data["species"],
            },
            "drn": date,
            "experiment": {
                "comment": gs_data["comment"],
                "date": date,
                "ec_id": str(gs_data["medium_ec_batch"]),
                "ec_solution": solution["ec"],
                "ic_id": str(gs_data["medium_ic_batch"]),
                "ic_solution": solution["ic"],
                "induction": int(gs_data["induction_time"]),
                "manufacturer": "Sophion",
                "model_name": "QPatch Compact",
                "se_id": str(gs_data["medium_se_batch"]),
                "se_solution": solution["se"],
                "temp": gs_data["temperature"],
                "time": time,
            },
            "experimenter": {
                "user_initials": gs_data["experimenter"],
            },
        }

    def get_protocol_map(self) -> dict[int, int | None]:
        """Get the map from internal protocol IDs to protocol IDs.

        Returns:
            dict: The protocol map.

        This map is defined in the t_voltage_protocol.xml file.

        Each voltage protocol used should be found in this file.
        Whether the protocol was repeated or not, it may appear only once in this file.

        Each protocol is identified by a unique internal ID that is assigned by the
        machine. This ID is stored in T_VOLTAGE_PROTOCOL_ID.

        The experimenter adds a description to each protocol that includes the
        type and id assigned to that protocol. This description can be found in the
        DESCRIPTION or ID_NAME tags. Here we parse the DESCRIPTION tag to extract the
        protocol type and the assigned ID (different from the internal).

        """

        # the Description contains the protocol names and has several possible forms:
        # Original:#<word> <word/number>
        # Original:#<garbage1> <garbage2> ... <word> <word/number>
        # we extract the last number which should be the protocol id
        def parse_description(tag) -> int | None:
            prot_id = tag.removeprefix("Original:#").split(" ")[-1]

            # if the description does not end in integer (example, "holding potential")
            # we return None
            if not prot_id.isnumeric():
                return None

            return int(prot_id)

        xml_file = self.Exp.xml.read("t_voltage_protocol")

        internal_ids = xml_file.tags("ROW", "T_VOLTAGE_PROTOCOL_ID", parser=int)
        protocol_ids = xml_file.tags("ROW", "DESCRIPTION", parser=parse_description)

        if len(set(internal_ids)) != len(internal_ids):
            raise ValueError("Duplicate internal IDs")

        # map the internal ids to the parsed protocol id
        return {id: protocol_ids[i] for i, id in enumerate(internal_ids)}

    def get_protocol_runs(self, protocol_map: dict) -> list:
        """Get the protocol runs for the rcell dictionary.

        Returns:
            list: A list of protocol runs.

        Each run corresponds to the application of a protocol with a given number
        of sweeps. Runs corresponding to the same protocol will later be grouped
        together by another function under the same dictionary and named as
        repetition1, repetition2, etc.

        Each run is represented as a dictionary with the following keys:

        From raw_data:
            data     (matrix: time_steps X n_sweeps)
            n_points (scalar) # number of time steps

        From meta_data:
            capacitance_fast (vector: n_sweeps)
            capacitance_slow (vector: n_sweeps)
            r_membrane       (vector: n_sweeps)
            r_series         (vector: n_sweeps)
            head_temp        float

        Time column of raw_data:
            time         (vector: time_steps)
            x_interval   int

        """
        # INTEGRITY CHECK

        # Check if number of raw columns matches the number of meta rows
        # - 1: do not count the time column
        # // 2 : do not count the voltage columns
        raw_cols = (len(self.Exp.raw.headers) - 1) // 2
        meta_rows = self.Exp.meta.count_lines()

        if meta_rows != raw_cols:
            raise ValueError(f"""
            Mismatched Meta/Raw data sizes:
                Number of metadata rows ({meta_rows})
                Number of raw data columns ({raw_cols})
            """)

        # RAW DATA

        # Get the raw data as a dataframe
        # We do not need the entire raw data, in case there are long columns
        # like holding potential columns, so we get the max number of timesteps
        # used in the experiment and read only that number of rows.

        # Determine maximum time (in us) from the duration of each protocol
        max_time = max(
            STIMCSV.get(id).duration
            for id in protocol_map.values()
            if id is not None
        )  # us

        time = np.arange(
            0,
            max_time,
            X_INTERVAL,
            dtype=np.uint32,
        )

        raw_data = self.Exp.raw.as_dataframe(nrows=len(time))

        # METADATA

        # If temperature is not available, it appears as "Failed"
        meta_data = self.Exp.meta.as_dataframe(na_values=["Failed"])

        # The metadata columns are renamed to the desired names
        rename_map = {
            "R-membrane [MΩ]": "r_membrane",
            "R-series [MΩ]": "r_series",
            "C-fast [pF]": "capacitance_fast",
            "C-slow [pF]": "capacitance_slow",
            "Temperature [°C]": "head_temp",
        }
        cols_of_interest = list(rename_map.values())

        meta_data.rename(columns=rename_map, inplace=True)

        meta_groups = meta_data.groupby(
            ["r_membrane", "r_series", "capacitance_fast", "capacitance_slow"],
            sort=False,
        )

        protocol_runs = []

        start = 2
        for _, group in meta_groups:
            md = {}

            for k, series in group.loc[:, cols_of_interest].to_dict("series").items():
                if k == "head_temp":
                    md[k] = series.mean()
                else:
                    md[k] = series.to_numpy()

            length = 2 * len(group["Experiment"])

            rd = (
                raw_data.iloc[:, start : start + length : 2]
                .dropna()
                .to_numpy(dtype=float)
            )
            rd *= nwb.unit.factor(nwb.unit.nanoAmpere, "A")

            start += length

            vecs = {
                "n_points": np.full(rd.shape[1], rd.shape[0], dtype=int),
                "x_interval": X_INTERVAL,
                "time": time[: rd.shape[0]],
            }

            run = {
                "data": rd,
                **vecs,
                **md,
            }

            # # The metadata columns are already in the desired units
            # unit_map = {
            #     "r_membrane": nwb.unit.MegaOhm,
            #     "r_series": nwb.unit.MegaOhm,
            #     "capacitance_fast": nwb.unit.picoFarad,
            #     "capacitance_slow": nwb.unit.picoFarad,
            # }

            protocol_runs.append(run)

        return protocol_runs

    def group_protocol_runs(
        self, protocol_runs: list, protocol_map: dict
    ) -> tuple[dict, dict]:
        """Group repeated applications of the same protocol under the same dictionary.

        Args:
            protocol_runs (list): The list of protocol runs.
            protocol_map (dict): The protocol map.

        Returns:
            dict: A dictionary of protocol ids to the number of repetitions.
            dict: The grouped protocol runs dictionary.

        Runs with the same voltage protocol (same ramps, same number of sweeps...)
        are grouped together. Each run is named as repetition1, repetition2...

        The output is a dictionary with the following structure:
            {
                "ProtocolA": {
                    "repetitions": {
                        "repetition1": {
                            "amp": {
                                "step_voltage": vector,
                            },
                            "data": matrix,
                            "n_points": scalar,
                            "capacitance_slow": vector,
                            "capacitance_fast": vector,
                            {...}
                        },
                        "repetition2": {...},
                        ...
                    }
                },
                "ProtocolB": {...},
            }

        # t_assay_vp.xml
        The t_assay_vp.xml file contains the unique internal IDs of each run.
        With this, we can make the correspondence and determine the protocol
        name of each run. However, if a single protocol is used for all runs,
        this file is not present. In this case, t_voltage_protocol.xml will
        contain a single protocol, which is used to group all runs together.

        """
        if len(protocol_map) > 1:
            # More than one voltage protocol was used
            # The internal_id of each run is in t_assay_vp.xml
            xml_file = self.Exp.xml.read("t_assay_vp")
            xml_run_ids = xml_file.tags("ROW", "T_VOLTAGE_PROTOCOL_ID", parser=int)

            # get the protocol ids for each run, skipping holding potential
            prot_id_per_run = [
                protocol_map[id] for id in xml_run_ids if protocol_map[id] is not None
            ]
        else:
            # Only one voltage protocol found
            # We assume all runs used the same protocol
            prot_id_per_run = [protocol_map.popitem()[1]]

        single_protocol = all(id == prot_id_per_run[0] for id in prot_id_per_run)

        stim_ids = []

        discarded_protocols = set()
        counts = dict()

        # Finally, group the runs by protocol name
        acquisition = {}
        for i, run in enumerate(protocol_runs):
            # The t_assay_vp xml file can have less entries than the
            # number of runs, for example if the runs were not separated. In this case,
            # if a single protocol was identified, we will assume that the remaining
            # unlabeled runs used that same protocol. Otherwise, we will discard them.

            if i >= len(prot_id_per_run):
                if single_protocol:
                    id = prot_id_per_run[-1]
                else:
                    self.messages.append(
                        f"Discarded run {i:02d} with no attributed protocol"
                    )
                    continue
            else:
                id = prot_id_per_run[i]

            stim_info = STIMCSV.get(id).info

            if id not in counts:
                counts[id] = {
                    "type": stim_info["name"],
                    "sweeps": stim_info["sweep_count"],
                    "seen": [i],
                    "kept": [],
                }
            else:
                counts[id]["seen"].append(i)

            # Ramp 134 protocol was sometimes repeated in 12 sweeps
            # when in fact it should only have a single sweep.
            # We fix this here by discarding the extra sweeps
            if id == 134:
                n_sweeps = run["data"].shape[1]
                if n_sweeps > 1:
                    self.messages.append(
                        f"Discarded extra sweeps from run {i:02d} of protocol {id:03d}"
                    )
                    run = truncate_run(run, 1)

            # Discard runs with wrong number of sweeps
            if self.drop:
                sweep_count = stim_info["sweep_count"]
                data_sweeps = run["data"].shape[1]

                if sweep_count != data_sweeps:
                    discarded_protocols.add(id)
                    self.messages.append(
                        f"Discarded run {i:02d} of protocol {id:03d} with wrong"
                        + f" number of sweeps ({data_sweeps} != {sweep_count})"
                    )
                    continue

                if id in discarded_protocols:
                    self.messages.append(
                        f"Discarded run {i:02d} of protocol {id:03d} with wrong"
                        + " number of sweeps on previous run"
                    )
                    continue

            typ = stim_info["name"]

            if typ not in acquisition:
                acquisition[typ] = {"repetitions": dict()}
                stim_ids.append(id)

            reps = acquisition[typ]["repetitions"]

            rep_num = len(reps) + 1
            rep_name = f"repetition{rep_num}"

            if self.keep_reps and rep_num > self.keep_reps:
                self.messages.append(
                    f"Discarded run {i:02d} of protocol {id:03d} beyond"
                    + f" the repetition limit ({self.keep_reps})"
                )
                continue

            counts[id]["kept"].append(i)

            reps[rep_name] = run

        return counts, acquisition


def truncate_run(run: dict, n_sweeps: int) -> dict:
    """Truncate a run to a given number of sweeps.

    Args:
        run (dict): The run dictionary.
        n_sweeps (int): The number of sweeps to keep.

    Returns:
        dict: The truncated run dictionary.

    """
    truncated = {}
    for k, v in run.items():
        if k != "time" and isinstance(v, np.ndarray):
            truncated[k] = v[..., :n_sweeps]
        else:
            truncated[k] = v

    return truncated
