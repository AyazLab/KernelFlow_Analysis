import os
import snirf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from typing import Union
from behav_analysis import Data_Functions, Participant_Behav, load_results


def sort_dict(dictionary: dict, sort_by: str) -> dict:
    """
    Sort a dictionary by keys or values.

    Args:
        dictionary (dict): Dictionary to sort.
        sort_by (str): How to sort to dictionary: by "keys" or by "values".

    Raises:
        Exception: Invalid sort_by argument.

    Returns:
        dict: Sorted dictionary.
    """
    if "key" in sort_by.lower():
        return dict(sorted(dictionary.items(), key=lambda item: item[0]))
    elif "value" in sort_by.lower():
        return dict(sorted(dictionary.items(), key=lambda item: item[1]))
    else:
        raise Exception("Invalid 'sort_by' argument. Must be 'key' or 'value'.")


class Process_Flow:
    """
    This class contains functions for processing Kernel Flow data.
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialize by loading SNIRF file.

        Args:
            filepath (str): Path to SNIRF file.
        """
        self.snirf_file = self.load_snirf(filepath)

    def load_snirf(self, filepath: str) -> snirf.Snirf:
        """
        Load SNIRF file.

        Args:
            filepath (str): Path to SNIRF file.

        Returns:
            snirf.Snirf: SNIRF file object.
        """
        return snirf.Snirf(filepath, "r+", dynamic_loading=True)

    def get_time_origin(self, fmt: str = "datetime") -> Union[datetime.datetime, float]:
        """
        Get the time origin (start time) from the SNIRF file.

        Args:
            fmt (str, optional): Format to get the time origin in: "datetime" or "timestamp". Defaults to "datetime".

        Raises:
            Exception: Invalid fmt argument.

        Returns:
            Union[datetime.datetime, float]:
                datetime.datetime: Time origin datetime.
                -or-
                float: Time origin timestamp.
        """
        start_date = self.snirf_file.nirs[0].metaDataTags.MeasurementDate
        start_time = self.snirf_file.nirs[0].metaDataTags.MeasurementTime
        start_str = start_date + " " + start_time
        time_origin = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        if fmt.lower() == "datetime":
            return time_origin
        elif fmt.lower() == "timestamp":
            return datetime.datetime.timestamp(time_origin)
        else:
            raise Exception(
                "Invalid 'fmt' argument. Must be 'datetime' or 'timestamp'."
            )

    def get_subject_ID(self) -> str:
        """
        Get the subject ID from the SNIRF file.

        Returns:
            str: Subject ID.
        """
        return self.snirf_file.nirs[0].metaDataTags.SubjectID

    def get_time_rel(self) -> np.ndarray:
        """
        Get the relative time array from the SNIRF file.

        Returns:
            np.ndarray: Relative time array.
        """
        return self.snirf_file.nirs[0].data[0].time

    def get_time_abs(self, fmt: str = "datetime") -> np.ndarray:
        """
        Convert relative time array into an absolute time array.

        Args:
            fmt (str, optional): Format to get the time array in: "datetime" or "timestamp". Defaults to "datetime".

        Returns:
            np.ndarray: Absolute time array.
        """
        time_rel = self.get_time_rel()
        if fmt.lower() == "datetime":
            time_origin_dt = self.get_time_origin("datetime")
            return np.array(
                [
                    datetime.timedelta(seconds=time_rel[i]) + time_origin_dt
                    for i in range(len(time_rel))
                ]
            )
        elif fmt.lower() == "timestamp":
            time_origin_ts = self.get_time_origin("timestamp")
            return time_rel + time_origin_ts

    def get_data(self, cols: list[int | list | tuple]) -> np.ndarray:
        """
        Get timeseries data from the SNIRF file.

        Args:
            cols (list[int  |  list  |  tuple]): Data cols to select. Single col, list of cols, or slice of cols.

        Returns:
            np.ndarray: Timeseries data array.
        """
        if isinstance(cols, tuple):
            return self.snirf_file.nirs[0].data[0].dataTimeSeries[:, cols[0] : cols[1]]
        else:
            return self.snirf_file.nirs[0].data[0].dataTimeSeries[:, cols]

    def get_unique_data_types(self) -> list:
        """
        Get unique data types from the SNIRF file.

        Returns:
            list: Unique data types.
        """
        data_types = []
        for i in range(len(self.snirf_file.nirs[0].data[0].measurementList)):
            data_type = self.snirf_file.nirs[0].data[0].measurementList[i].dataType
            if data_type not in data_types:
                data_types.append(data_type)
        return data_types

    def get_unique_data_type_labels(self) -> list:
        """
        Get unique data type labels from the SNIRF file.

        Returns:
            list: Unique data type labels.
        """
        data_type_labels = []
        for i in range(len(self.snirf_file.nirs[0].data[0].measurementList)):
            data_type_label = (
                self.snirf_file.nirs[0].data[0].measurementList[i].dataTypeLabel
            )
            if data_type_label not in data_type_labels:
                data_type_labels.append(data_type_label)
        return data_type_labels

    def create_source_dict(self) -> dict:
        """
        Count the occurrences of each source index.

        Returns:
            dict: Counts for each source index.
        """
        source_dict = {}
        for i in range(len(self.snirf_file.nirs[0].data[0].measurementList)):
            source = self.snirf_file.nirs[0].data[0].measurementList[i].sourceIndex
            source_dict[source] = source_dict.get(source, 0) + 1
        source_dict = sort_dict(source_dict, "keys")
        return source_dict

    def create_detector_dict(self) -> dict:
        """
        Count the occurrences of each detector index.

        Returns:
            dict: Counts for each detector index.
        """
        detector_dict = {}
        for i in range(len(self.snirf_file.nirs[0].data[0].measurementList)):
            detector = self.snirf_file.nirs[0].data[0].measurementList[i].detectorIndex
            detector_dict[detector] = detector_dict.get(detector, 0) + 1
        detector_dict = sort_dict(detector_dict, "keys")
        return detector_dict


class Participant_Flow:
    """
    This class contains functions, data structures, and info necessary for
    processing Kernel Flow data from the experiments.
    """

    def __init__(self, par_num):
        self.data_fun = Data_Functions()
        self.par_behav = Participant_Behav(par_num)
        self.par_num, self.par_ID = self.data_fun.process_par(par_num)
        data_dir = r"C:\Kernel\participants"
        self.flow_data_dir = os.path.join(data_dir, self.par_ID, "flow_data")
        self.session_list = ["1001", "1002", "1003"]

    def load_flow_session(self, session_num: int) -> snirf.Snirf:
        """
        Load Kernel Flow data for an experiment session.

        Args:
            session_num (int): Experiment session number.

        Raises:
            Exception: Invalid session number argument.

        Returns:
            snirf.Snirf: SNIRF file object.
        """
        if isinstance(session_num, str):
            session_num = int(session_num)
        elif isinstance(session_num, int):
            pass
        else:
            raise Exception("Invalid session number.")
        session_num_str = f"session_{session_num}"
        session_dir = os.path.join(self.flow_data_dir, session_num_str)
        filename = os.listdir(session_dir)[0]
        filepath = os.path.join(session_dir, filename)
        return Process_Flow(filepath).snirf_file

    def load_flow_exp(self, exp_name: str) -> snirf.Snirf:
        # TODO: load kernel flow data from start to end timestamp of an experiment
        # include ts-adjusted time column as first column
        pass

    def create_flow_session_dict(self) -> dict:
        """
        Create a dictionary of Kernel Flow data for all experiment sessions.

        Returns:
            dict: Kernel Flow data for all experiment sessions.
                keys:
                    "session_1001", "session_1002", "session_1003"
                values:
                    SNIRF file object for each experiment session
        """
        flow_session_dict = {}
        for session in self.session_list:
            flow_session_dict[session] = self.load_flow_session(session)
        return flow_session_dict
