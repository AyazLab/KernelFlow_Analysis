import os
import bisect
import datetime
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union


class Data_Functions:
    """
    This class contains functions used for processing experiment files as well as
    participant, behavioral, and physiological data.
    """

    def process_par(self, par_num: list[str | int]) -> Tuple[int, str]:
        """
        Create the participant number and ID.

        Args:
            par_num (list[str | int]): Participant number.

        Raises:
            Exception: Invalid participant number.

        Returns:
            Tuple[int, str]:
                Participant number
                -and-
                Participant ID
        """
        if isinstance(par_num, str):
            par_num = int(par_num)
        elif isinstance(par_num, int):
            pass
        else:
            raise Exception("Invalid participant number.")
        par_num_str = "{:02d}".format(par_num)
        par_ID = f"participant_{par_num_str}"
        return par_num, par_ID

    def parse_log_file(self, par_dir: str, exp_name: str) -> dict:
        """
        Parses the experiment log file into start and end marker data.

        Args:
            par_dir (str): Path to specific participant directory
            exp_name (str): Name of the experiment

        Returns:
            dict: Start and end marker data
                keys:
                    'start_marker', 'end_marker'
                values:
                    start marker dictionary, end marker dictionary
                        keys:
                            'marker_ID', 'marker_value', 'marker_string', 'timestamp'
                        values:
                            'marker_ID', 'marker_value', 'marker_string', 'timestamp'
        """

        def _parse_udp(udp: str, sent_time: str) -> dict:
            """
            Parses UDP file lines into marker information.

            Args:
                udp (str): File line with UDP data
                sent_time (str): Relative time this marker was sent

            Returns:
                dict: Marker data
                    keys:
                        'sent_time', 'marker_ID', 'marker_value', 'marker_string', 'timestamp'
                    values:
                        'sent_time', 'marker_ID', 'marker_value', 'marker_string', 'timestamp'
            """
            marker_ID_info = udp[2].strip(",").split("=")
            marker_ID_str = marker_ID_info[0]
            marker_ID = marker_ID_info[1]

            marker_val_info = udp[3].strip(",").split("=")
            marker_val_str = marker_val_info[0]
            marker_val = marker_val_info[1]

            marker_string_info = udp[4].strip(",").split("=")
            marker_string_str = marker_string_info[0]
            marker_string = marker_string_info[1]

            marker_ts_info = udp[5].strip("\n").split("=")
            marker_ts_str = marker_ts_info[0]
            marker_ts = marker_ts_info[1]

            udp_dict = {
                "sent_time": sent_time,
                marker_ID_str: marker_ID,
                marker_val_str: marker_val,
                marker_string_str: marker_string,
                marker_ts_str: marker_ts,
            }

            return udp_dict

        log_dir = os.path.join(par_dir, exp_name, "data")
        for filename in os.listdir(log_dir):
            if ".log" in filename:
                log_filename = filename
        log_filepath = os.path.join(log_dir, log_filename)

        with open(log_filepath) as f:
            lines = f.readlines()

        udp_lines = []
        sent_time_list = []
        for line in lines:
            if "UDP" in line:  # only select lines with UDP info
                split_line = line.split("\t")
                udp_lines.append(split_line[-1])
                sent_time_list.append(split_line[0].strip())

        marker_data = {}
        try:
            start_udp = udp_lines[0].split(" ")
            marker_data["start_marker"] = _parse_udp(start_udp, sent_time_list[0])
        except:
            marker_data["start_marker"] = "_"
        try:
            end_udp = udp_lines[1].split(" ")
            marker_data["end_marker"] = _parse_udp(end_udp, sent_time_list[1])
        except:
            if (
                exp_name == "go_no_go"
            ):  # Go/No-go start marker did not write to log file
                marker_ID = int(marker_data["start_marker"]["marker_ID"]) + 1
                marker_val = 22
                marker_string = "go_no_go_end"
                end_ts = marker_data["start_marker"]["timestamp"]
                end_ts = int(
                    float(end_ts) + float(lines[-1].split("\t")[0]) * 1e9 - 0.4 * 1e9
                )
                marker_data["end_marker"] = {
                    "sent_time": float("NaN"),
                    "marker_ID": marker_ID,
                    "marker_value": marker_val,
                    "marker_string": marker_string,
                    "timestamp": end_ts,
                }
            else:
                marker_data["end_marker"] = "_"

        return marker_data

    def parse_narrative_log_file(self, par_dir: str, exp_name: str) -> float:
        """
        Parses the narrative log file to get the end timestamp for narrative experiments.

        Args:
            par_dir (str): Path to specific participant directory
            exp_name (str): Name of the experiment

        Returns:
            float: End timestamp of a narrative experiment
        """
        log_dir = os.path.join(par_dir, exp_name, "data")
        for filename in os.listdir(log_dir):
            if ".log" in filename:
                log_filename = filename
        log_filepath = os.path.join(log_dir, log_filename)

        with open(log_filepath) as f:
            lines = f.readlines()

        udp_lines = []
        for line in lines:
            if "UDP" in line:
                udp_lines.append(line.split("\t")[0])

        end_time = float(udp_lines[1]) * 1e9

        return end_time

    def parse_task_order_file(self, par_dir: str, exp_name: str) -> pd.DataFrame:
        """
        Parses the task order file to get the order of task CSV files for a given experiment.

        Args:
            par_dir (str): Path to specific participant directory
            exp_name (str): Name of the experiment

        Returns:
            pd.DataFrame: Column of task order CSV filenames
                column name: 'task_order'
        """

        exp_dir = os.path.join(par_dir, exp_name)
        for filename in os.listdir(exp_dir):
            if ".csv" in filename:
                task_order_filename = filename
        task_order_filepath = os.path.join(exp_dir, task_order_filename)
        task_order = pd.read_csv(task_order_filepath)

        return task_order

    def get_data_filepath(self, par_dir: str, exp_name: str) -> str:
        """
        Gets the file path to the data CSV file for a given experiment.

        Args:
            par_dir (str): Path to specific participant directory
            exp_name (str): Name of the experiment

        Returns:
            str: Path to experiment CSV data file
        """

        data_dir = os.path.join(par_dir, exp_name, "data")
        for filename in os.listdir(data_dir):
            if ".csv" in filename:
                data_filename = filename
        data_filepath = os.path.join(data_dir, data_filename)

        return data_filepath

    def load_marker_dict(self, filepath: str) -> dict:
        """
        Load the marker and experiment code CSV file into a dictionary.

        Args:
            filepath (str): Filepath to the marker CSV file.

        Returns:
            dict: Marker and experiment code dictionary.
        """
        df = pd.read_csv(filepath)
        marker_dict = {}
        for _, row in df.iterrows():
            marker_dict[row["marker_val"]] = row["marker_str"]
        return marker_dict

    def get_all_marker_timestamps(self, par_dir: str, exp_order: list) -> dict:
        """
        Organize the start and end timestamps for each experiment into a dictionary.

        Args:
            par_dir (str): Path to specific participant directory.
            exp_order (list): Experiment order.

        Returns:
            dict: Start and end timestamps for each experiment.
                keys:
                     'audio_narrative', 'go_no_go', 'king_devick', 'n_back', 'resting_state',
                     'tower_of_london', 'video_narrative_cmiyc', 'video_narrative_sherlock', 'vSAT'
                values:
                    [start timestamp, end timestamp]
        """
        all_marker_timestamps = {}
        for exp_name in exp_order:
            udp_dict = self.parse_log_file(par_dir=par_dir, exp_name=exp_name)
            start_marker, end_marker = (
                udp_dict["start_marker"],
                udp_dict["end_marker"],
            )
            if (
                "narrative" in exp_name
                and "participant_01" not in par_dir
                and "participant_02" not in par_dir
                and "participant_03" not in par_dir
            ):  # Narrative experiment timestamps changed for Participant 04+
                try:
                    start_ts = start_marker["timestamp"]
                except:
                    start_ts = "_"
                    print("Start marker not found for {exp_name}")
                try:
                    end_ts = float(start_ts) + self.parse_narrative_log_file(
                        par_dir, exp_name
                    )
                except:
                    print("End marker not found for {exp_name}")

            else:
                try:
                    start_ts = start_marker["timestamp"]
                except:
                    start_ts = "_"
                    print("Start marker not found for {exp_name}")

                try:
                    end_ts = end_marker["timestamp"]
                except:
                    end_ts = "_"
                    print("End marker not found for {exp_name}")
            all_marker_timestamps[exp_name] = [start_ts, end_ts]

        return all_marker_timestamps

    def adjust_all_marker_timestamps(
        self,
        all_marker_timestamps: dict,
        processed_data_dir: str,
        par_num: int,
    ) -> dict:
        """
        Adjust experiment end timestamp markers using the final stim end timestamp.

        Args:
            all_marker_timestamps (dict): Start and end timestamps for each experiment.
            processed_data_dir (str): Processed data directory.
            par_num (int): Participant number.

        Returns:
            dict: Start and adjusted end timestamp for each experiment.
                keys:
                     'audio_narrative', 'go_no_go', 'king_devick', 'n_back', 'resting_state',
                     'tower_of_london', 'video_narrative_cmiyc', 'video_narrative_sherlock', 'vSAT'
                values:
                    [start timestamp, adjusted end timestamp]
        """
        for exp_name in all_marker_timestamps.keys():
            orig_end_ts = float(all_marker_timestamps[exp_name][1])
            exp_df = load_results(processed_data_dir, exp_name, par_num)
            end_ts = float(exp_df.iloc[-1]["stim_end"])
            adj_end_ts = (end_ts + 3) * 1e9
            if (
                orig_end_ts < adj_end_ts
            ):  # only replace if adjusted end timestamp is more recent
                all_marker_timestamps[exp_name][1] = adj_end_ts
        return all_marker_timestamps

    def get_cols(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Get a selection of columns from a given DataFrame.

        Args:
            df (pd.DataFrame): Experiment data
            cols (list): List of columns to select from the DataFrame

        Returns:
            pd.DataFrame: DataFrame with selected columns only
        """
        return df[cols]

    def create_col(self, x, num_rows: int, dtype=object) -> pd.Series:
        """
        Create a Series column of a value.

        Args:
            x (Any): Value for each row in the column
            num_rows (int): Number of rows in the column
            dtype (object, optional): Type of x

        Returns:
            pd.Series: Column with num_rows rows of x
        """
        return pd.Series([x] * num_rows, dtype=dtype)

    def parse_df(
        self, df: pd.DataFrame, num_blocks: int, num_trials: int
    ) -> Tuple[dict, pd.DataFrame]:
        """
        Parses a DataFrame into a dictionary organized by block and a DataFrame with NaN rows removed.

        Args:
            df (pd.DataFrame): DataFrame to prase
            num_blocks (int): Number of blocks in the experiment
            num_trials (int): Number of trials in the experiment

        Returns:
            Tuple[dict, pd.DataFrame]:
                dict: dictionary organized by block
                    keys:
                        'block_1', 'block2', ... 'block_N'
                    values:
                        DataFrame of behavioral data for that block
                pd.DataFrame: DataFrame with no NaN rows
        """
        df_by_block = {}
        for i in range(num_blocks):
            block_name = f"block_{i+1}"
            if i == 0:
                temp_df = df.iloc[
                    i * num_trials : (i + 1) * num_trials
                ]  # select rows for this block
                df_no_nan = temp_df.copy()
            else:
                temp_df = df.iloc[
                    (i * num_trials) + i : ((i + 1) * num_trials) + i
                ]  # skip Nan line between blocks
                df_no_nan = pd.concat([df_no_nan, temp_df])
            df_by_block[block_name] = temp_df

        return df_by_block, df_no_nan

    def get_exp_ts(self, df: pd.DataFrame, exp_name: str) -> Tuple[int, int]:
        """
        Get the start and end timestamps from an experiment-organized DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with experiment-organized start and end marker timestamps
            exp_name (str): Name of the experiment

        Returns:
            Tuple[int, int]: start timestamp, end timestamp
        """
        df_temp = df[df["exp_name"] == exp_name]
        start_ts = df_temp["start_timestamp"].item()
        end_ts = df_temp["end_timestamp"].item()

        return start_ts, end_ts

    def get_exp_dt(
        self, df: pd.DataFrame, exp_name: str
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Get the start and end datetimes from an experiment-organized DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with experiment-organized start and end marker timestamps
            exp_name (str): Name of the experiment

        Returns:
            Tuple[datetime.datetime, datetime.datetime]: start datetime, end datetime
        """
        df_temp = df[df["exp_name"] == exp_name]
        start_dt = datetime.datetime.fromtimestamp(
            df_temp["start_timestamp"].item() / 1e9
        )
        end_dt = datetime.datetime.fromtimestamp(df_temp["end_timestamp"].item() / 1e9)

        return start_dt, end_dt

    def get_start_index_dt(
        self, array_like: list[pd.DataFrame | np.ndarray], start_dt: datetime.datetime
    ) -> Optional[int]:
        """
        Get the index of the start datetime of an experiment.

        Args:
            array_like list[pd.DataFrame, np.ndarray]: Experiment data DataFrame or Array
            start_dt (datetime.datetime): Start datetime of the experiment

        Returns:
            Optional[int]: Start index or None
        """
        try:
            if isinstance(array_like, pd.DataFrame):
                for loc, dt in enumerate(array_like["datetime"]):
                    if not dt < start_dt:
                        break
                if loc < array_like["datetime"].shape[0] - 1:
                    return loc
                else:
                    print("Start index datetime not found!")
                    return None
            elif isinstance(array_like, np.ndarray):
                for loc, dt in enumerate(array_like):
                    if not dt < start_dt:
                        break
                if loc < array_like.shape[0] - 1:
                    return loc
                else:
                    print("Start index datetime not found!")
                    return None
        except:
            print("Start index datetime not found!")
            return None

    def get_end_index_dt(
        self, array_like: list[pd.DataFrame | np.ndarray], end_dt: datetime.datetime
    ) -> Optional[int]:
        """
        Get the index of the end datetime of an experiment.

        Args:
            array_like list[pd.DataFrame, np.ndarray]: Experiment data DataFrame or Array
            end_dt (datetime.datetime): End datetime of the experiment

        Returns:
            Optional[int]: End index or None
        """
        try:
            if isinstance(array_like, pd.DataFrame):
                for loc, dt in enumerate(array_like["datetime"]):
                    if dt > end_dt:
                        break
                return loc
            elif isinstance(array_like, np.ndarray):
                for loc, dt in enumerate(array_like):
                    if dt > end_dt:
                        break
                return loc
        except:
            print("End index datetime not found!")
            return None

    def get_start_index_ts(self, df: pd.DataFrame, start_ts: float) -> Optional[int]:
        """
        Get the index of the start timestamp of an experiment

        Args:
            df (pd.DataFrame): Experiment data with timestamp column
            start_ts (float): Start timestamp of the experiment

        Returns:
            Optional[int]: Start index or None
        """
        try:
            for loc, ts in enumerate(df["timestamps"]):
                if not ts < start_ts:
                    break
            if loc < df["timestamps"].shape[0] - 1:
                return loc
            else:
                print("Start index timestamp not found!")
                return None
        except:
            print("Start index timestamp not found!")
            return None

    def get_end_index_ts(self, df: pd.DataFrame, end_ts: float) -> Optional[int]:
        """
        Get the index of the end timestamp of an experiment

        Args:
            df (pd.DataFrame): Experiment data with timestamp column
            end_ts (float): End timestamp of the experiment

        Returns:
            Optional[int]: End index or None
        """
        try:
            for loc, ts in enumerate(df["timestamps"]):
                if ts > end_ts:
                    break
            return loc
        except:
            print("End index timestamp not found!")
            return None

    def adjust_df_ts(
        self, df: pd.DataFrame, start_ts: int, cols: list, by_block: bool = False
    ) -> pd.DataFrame:
        """
        Offset experiment times by the initial timestamp of the experiment (relative to absolute timestamps).

        Args:
            df (pd.DataFrame): DataFrame to time-adjust
            start_ts (int): Start timestamp of the experiment
            cols (list): Columns to adjust the timestamps of
            by_block (bool, optional): Is the DataFrame organized by block? Defaults to False

        Returns:
            pd.DataFrame: Timestamp-adjusted DataFrame
        """
        df = df.copy()
        if by_block:
            for block, temp_df in df.items():
                temp_df = temp_df.copy()
                for col in cols:
                    temp_df[col] = (
                        temp_df[col].astype(float) + start_ts
                    )  # add start timestamp to relative timestamps
                df[block] = temp_df
        else:
            for col in cols:
                df[col] = (
                    df[col].astype(float) + start_ts
                )  # add start timestamp to relative timestamps
        return df

    def c_to_f(self, temp: float) -> float:
        """
        Convert celsius to fahrenheit.

        Args:
            temp (float): Temperature in fahrenheit

        Returns:
            float: Temperature in celsius
        """
        return round(temp * 9 / 5 + 32, 2)

    def flatten(self, input_list: List[list]) -> list:
        """
        Flatten a list of lists into a single list.

        Args:
            input_list (List[list]): List of lists

        Returns:
            list: Single list with all elements of the input list
        """
        return [x for xs in input_list for x in xs]

    def get_key_from_value(self, dictionary: dict, value: List[int | str]):
        """
        Get a dictionary key that contains a specified value.

        Args:
            dictionary (dict): Dictionary.
            value (List[int | str]): Value to get the key of.

        Returns:
            Any: Key corresponding to the specified value.
        """
        return [k for k, v in dictionary.items() if value in v][0]

    def format_exp_name(self, exp_name: str) -> str:
        """
        Format an experiment name into title case.

        Args:
            exp_name (str): Experiment name.

        Returns:
            str: Formatted experiment name.
        """
        if exp_name == "tower_of_london":
            return "Tower of London"
        elif exp_name == "video_narrative_cmiyc":
            return "Video Narrative CMIYC"
        elif exp_name == "vSAT":
            return exp_name
        else:
            return " ".join([word.capitalize() for word in exp_name.split("_")])

    def create_unique_stim_dict(self, df: pd.DataFrame, col_name: str = "stim") -> dict:
        """
        Create a dictionary of unique items in a DataFrame column.

        Args:
            df (pd.DataFrame): DataFrame.
            col_name (str, optional): Column name to get unique values from. Defaults to "stim".

        Returns:
            dict: Dictionary of unique column items.
                keys:
                    unique items
                values:
                    unique ID number (int) for each item
        """
        unique_stims = df[col_name].unique()
        return {value: i for i, value in enumerate(unique_stims)}

    def sort_dict(self, dictionary: dict, sort_by: str) -> dict:
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

    def insert_df_after_col(
        self, df_orig: pd.DataFrame, df_insert: pd.DataFrame, col_name: str
    ) -> pd.DataFrame:
        """
        Inserts a dataframe into an existing dataframe after the specified column.

        Arguments:
        df_orig (pd.DataFrame): Original DataFrame.
        df_insert (pd.DataFrame): DataFrame to insert.
        col_name (str): Name of the column after which to insert the DataFrame.

        Returns:
        pd.DataFrame: Original DataFrame with a DataFrame inserted after the specified column.
        """
        col_index = df_orig.columns.get_loc(col_name)
        new_df = pd.concat(
            [
                df_orig.iloc[:, : col_index + 1],
                df_insert,
                df_orig.iloc[:, col_index + 1 :],
            ],
            axis=1,
        )

        return new_df

    def find_closest_ts(
        self,
        given_ts: float,
        ts_list: list[list | np.ndarray],
        index_offset: int = None,
    ) -> Tuple[int, float]:
        """
        Find the closest timestamp to a given timestamp. The closet timestamp
        will not be more recent than the given timestamp.

        Args:
            given_ts (float): Timestamp to find the closet one to.
            ts_list (list[list | np.ndarray]): List or array of timestamps.
            index_offset (int): Offset the closest timestamp index by a specified number of points.
                                Defaults to None.

        Raises:
            Exception: No valid timestamp found.

        Returns:
            Tuple[int, float]: Index of the closet timestamp to the given timestamp
                               and the closest timestamp.
        """

        ts_arr = np.asarray(ts_list)
        valid_ts = ts_arr[ts_arr <= given_ts]
        if len(valid_ts) == 0:
            raise Exception("No valid timestamp found.")
        idx = bisect.bisect_left(valid_ts, given_ts)  # index of closest timestamp
        if idx == 0:
            closest_ts = valid_ts[0]
        elif idx == len(valid_ts):
            closest_ts = valid_ts[-1]
        else:
            diff1 = abs(valid_ts[idx - 1] - given_ts)
            diff2 = abs(valid_ts[idx] - given_ts)
            if diff1 < diff2:
                closest_ts = valid_ts[idx - 1]
            else:
                closest_ts = valid_ts[idx]
        closest_idx = np.where(ts_arr == closest_ts)[0][0]
        if index_offset:
            closest_idx = closest_idx + index_offset
        return closest_idx, closest_ts


def load_results(
    results_dir: str, exp_name: str = None, par_num: list[int | list | tuple] = None
) -> Union[pd.DataFrame, dict]:
    """
    Read the experiment behavioral or Kernel Flow results from CSV files into
    DataFrame or a dictionary of DataFrames.

    Args:
        results_dir (str): Path to the results directory
        exp_name (str, optional): Get results for a specific experiment. Defaults to None
        par_num (list[int | list | tuple], optional): Participants to select. Single participant, list of participants, or slice of participants.
                                                      Defaults to None (all participants).

    Returns:
        Union[pd.DataFrame, dict]:
            pd.DataFrame: Data for a specified experiment.
            -or-
            dict: Results dictionary
                keys:
                    'audio_narrative', 'go_no_go', 'king_devick', 'n_back', 'resting_state',
                    'tower_of_london', 'video_narrative_cmiyc', 'video_narrative_sherlock', 'vSAT'
                values:
                    DataFrame of results for each experiment
    """

    if exp_name:
        for results_csv in os.listdir(results_dir):
            if exp_name in results_csv:
                full_path = os.path.join(results_dir, results_csv)
                df = pd.read_csv(full_path)
                if isinstance(par_num, int):
                    return df[df["participant"] == par_num]
                elif isinstance(par_num, list):
                    return df[df["participant"].isin(par_num)]
                elif isinstance(par_num, tuple):
                    return df[
                        (df["participant"] >= par_num[0])
                        & (df["participant"] <= par_num[1])
                    ]
                else:
                    return df
        print(
            "Invalid experiment name."
        )  # only reached if invalid experiment name argument
    else:
        exp_dict = {}
        for results_csv in os.listdir(results_dir):
            end_idx = results_csv.rfind("_")
            exp_name = results_csv[0:end_idx]
            full_path = os.path.join(results_dir, results_csv)
            df = pd.read_csv(full_path)
            if isinstance(par_num, int):
                df = df[df["participant"] == par_num]
            elif isinstance(par_num, list):
                df = df[df["participant"].isin(par_num)]
            elif isinstance(par_num, tuple):
                df = df[
                    (df["participant"] >= par_num[0])
                    & (df["participant"] <= par_num[1])
                ]
            else:
                pass
            exp_dict[exp_name] = df
        return exp_dict


def exp_name_to_title(exp_name: str) -> str:
    """
    Convert experiment name into a title format.

    Args:
        exp_name (str): Experiment name.

    Returns:
        str: Experiment name in title format.
    """
    if exp_name == "go_no_go":
        exp_name_title = "Go/No-Go"
    elif exp_name == "n_back":
        exp_name_title = exp_name.replace("_", "-").title()
    elif exp_name == "tower_of_london":
        exp_name_title = "Tower of London"
    elif exp_name == "video_narrative_cmiyc":
        exp_name_title = "Video Narrative CMIYC"
    elif exp_name == "vSAT":
        exp_name_title = exp_name
    else:
        exp_name_title = exp_name.replace("_", " ").title()
    return exp_name_title
