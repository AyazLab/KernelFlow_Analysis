import snirf
import numpy as np
import datetime
from typing import Union


def load_snirf(filepath: str) -> snirf.Snirf:
    return snirf.Snirf(filepath, "r+", dynamic_loading=True)


def get_time_origin(snirf_file: snirf.Snirf, fmt: str="datetime") -> Union[datetime.datetime, float]:
    start_date = snirf_file.nirs[0].metaDataTags.MeasurementDate
    start_time = snirf_file.nirs[0].metaDataTags.MeasurementTime
    start_str = start_date + " " + start_time
    time_origin = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    if fmt.lower() == "datetime":
        return time_origin
    elif fmt.lower() == "timestamp":
        return datetime.datetime.timestamp(time_origin)
    else:
        raise Exception("Invalid 'fmt' argument. Must be 'datetime' or 'timestamp'.")


def get_subject_ID(snirf_file: snirf.Snirf) -> str:
    return snirf_file.nirs[0].metaDataTags.SubjectID


def get_time_rel(snirf_file: snirf.Snirf) -> np.ndarray:
        return snirf_file.nirs[0].data[0].time


def get_time_abs(snirf_file: snirf.Snirf, fmt: str="datetime") -> np.ndarray:
    time_rel = get_time_rel(snirf_file)
    if fmt.lower() == "datetime":
        time_origin_dt = get_time_origin(snirf_file, "datetime")
        return np.array([datetime.timedelta(seconds=time_rel[i]) + time_origin_dt for i in range(len(time_rel))])
    elif fmt.lower() == "timestamp":
        time_origin_ts = get_time_origin(snirf_file, "timestamp")
        return time_rel + time_origin_ts


def get_data(snirf_file: snirf.Snirf, cols: list[int | list | slice]) -> np.ndarray:
    if type(cols) == tuple:
        return snirf_file.nirs[0].data[0].dataTimeSeries[:, cols[0] : cols[1]]
    else:
        return snirf_file.nirs[0].data[0].dataTimeSeries[:, cols]


def get_unique_data_types(snirf_file: snirf.Snirf) -> list:
    data_types = []
    for i in range(len(snirf_file.nirs[0].data[0].measurementList)):
        data_type = snirf_file.nirs[0].data[0].measurementList[i].dataType
        if data_type not in data_types:
            data_types.append(data_type)
    return data_types


def get_unique_data_type_labels(snirf_file: snirf.Snirf) -> list:
    data_type_labels = []
    for i in range(len(snirf_file.nirs[0].data[0].measurementList)):
        data_type_label = snirf_file.nirs[0].data[0].measurementList[i].dataTypeLabel
        if data_type_label not in data_type_labels:
            data_type_labels.append(data_type_label)
    return data_type_labels


def sort_dict(dictionary: dict, sort_by: str) -> dict:
    if "key" in sort_by.lower():
        return dict(sorted(dictionary.items(), key=lambda item: item[0]))
    elif "value" in sort_by.lower():
        return dict(sorted(dictionary.items(), key=lambda item: item[1]))
    else:
        raise Exception("Invalid 'sort_by' argument. Must be 'key' or 'value'.")


def create_source_dict(snirf_file: snirf.Snirf) -> dict:
    source_dict = {}
    for i in range(len(snirf_file.nirs[0].data[0].measurementList)):
        source = snirf_file.nirs[0].data[0].measurementList[i].sourceIndex
        source_dict[source] = source_dict.get(source, 0) + 1
    source_dict = sort_dict(source_dict, "keys")
    return source_dict


def create_detector_dict(snirf_file: snirf.Snirf) -> dict:
    detector_dict = {}
    for i in range(len(snirf_file.nirs[0].data[0].measurementList)):
        detector = snirf_file.nirs[0].data[0].measurementList[i].detectorIndex
        detector_dict[detector] = detector_dict.get(detector, 0) + 1
    detector_dict = sort_dict(detector_dict, "keys")
    return detector_dict
