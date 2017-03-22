# coding: utf-8
from __future__ import division, print_function

import configparser

import glob

import matplotlib
import os
import pandas as pd

from definitions import PROJECT_ROOT
from external_modules.detect_peaks import detect_peaks
from raw_data_conversion.conversion import synchronize_sensors
import numpy as np

matplotlib.use("Agg")  # Choose non-interactive background
import matplotlib.pyplot as plt

SUBJECT_DATA_LOCATION = os.path.join(PROJECT_ROOT, 'private_data', 'annotated_data')
folder_prefix = ""


def find_peak_intervals(data_points, required_peaks=3, sampling_frequency=100, min_period=0.15, max_period=8.0):
    intervals = []
    start_peak = data_points[0]
    previous_peak = start_peak
    peak_count = 1
    max_data_points_between_peaks = max_period * sampling_frequency
    min_data_points_between_peaks = min_period * sampling_frequency

    for peak in data_points:
        time_interval = peak - previous_peak

        if min_data_points_between_peaks < time_interval <= max_data_points_between_peaks:
            peak_count += 1
        elif time_interval > max_data_points_between_peaks:
            start_peak = peak
            peak_count = 1
        else:
            continue

        previous_peak = peak

        if peak_count == required_peaks:
            interval = (start_peak, peak)
            intervals.append(interval)
            start_peak = peak
            peak_count = 1

    return intervals


def find_claps(data_points, sampling_frequency, mph=5.0, valley=True, required_claps=3):
    peaks = detect_peaks(data_points, mph=mph, valley=valley)
    clap_times = find_peak_intervals(peaks, required_peaks=required_claps, sampling_frequency=sampling_frequency)
    return clap_times


def combine_annotation_files(filenames):
    dfs = []
    rolling_time_offset = 0
    first = True
    for filename in filenames:
        new = pd.read_csv(filename, sep='\t')
        if first:
            dfs.append(new)
            rolling_time_offset = new.tail(1).iloc[0]['end']
            first = False
        else:
            new['start'] = new['start'] + rolling_time_offset
            new['end'] = new['end'] + rolling_time_offset
            rolling_time_offset = new.tail(1).iloc[0]['end']
            dfs.append(new)

    annotations = pd.concat(dfs, ignore_index=True)

    return annotations


def make_labels_for_timestamps(timestamps, labels_list, end_times_list):
    labels_for_timestamps = []

    next_timestamp_index = 0

    l = zip(labels_list, end_times_list)
    for label, end_time in l:
        while next_timestamp_index < len(timestamps) and timestamps[next_timestamp_index] <= end_time:
            labels_for_timestamps.append(label)
            next_timestamp_index += 1

    return labels_for_timestamps


def convert_string_labels_to_numbers(label_list):
    label_to_number_dict = {
        "none": 0,
        "walking": 1,
        "running": 2,
        "shuffling": 3,
        "stairs (ascending)": 4,
        "stairs (descending)": 5,
        "standing": 6,
        "sitting": 7,
        "lying": 8,
        "transition": 9,
        "bending": 10,
        "picking": 11,
        "undefined": 12,
        "cycling (sit)": 13,
        "cycling (stand)": 14,
        "heel drop": 15,
        "vigorous activity": 16,
        "non-vigorous activity": 17,
        "Car": 18,
        "Transport(sitting)": 18,
        "Commute(standing)": 19
    }

    return [label_to_number_dict[label] for label in label_list]


def remove_annotations_outside_heel_drops(annotations, starting_drops=3, ending_drops=0):
    if starting_drops:
        cutoff_index = starting_drops - 1

        heel_drops = annotations.loc[annotations['type'] == 'heel drop'][cutoff_index:]
        offset = heel_drops.iloc[0]['end']
        annotations = annotations.drop(annotations[annotations.end <= offset].index)

        annotations['start'] = annotations['start'] - offset
        annotations['end'] = annotations['end'] - offset
        print("Found first heel drops in annotations at", offset, "seconds")

    if ending_drops:
        heel_drops = annotations.loc[annotations['type'] == 'heel drop'][0:]
        last_heel_drop = heel_drops.iloc[0]['start']
        annotations = annotations.drop(annotations[annotations.end > last_heel_drop].index)

    end_ = annotations.tail(1)['end'].iloc[0]
    print("Length of annotated data after removing heel drops:", end_, "seconds")

    return annotations


def extract(subject_id, cwas, pre_conversion_fix=True, mph=5.0, clean_up=True,
            sensor_label_sync_index=None, shifts=None, starting_drops=3, sampling_frequency=100, max_hours_to_read=48):
    max_rows = 3600 * max_hours_to_read * sampling_frequency
    subject_folder = os.path.join(SUBJECT_DATA_LOCATION, subject_id)
    output_folder = os.path.join(subject_folder, "output")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    annotations_glob = os.path.join(subject_folder, subject_id + "*_Event_*.txt")
    annotation_files = sorted(glob.glob(annotations_glob))
    annotations = combine_annotation_files(annotation_files)
    annotations = remove_annotations_outside_heel_drops(annotations, starting_drops=starting_drops, ending_drops=0)

    sync_file = os.path.join(subject_folder, subject_id + "_synchronized.csv")
    magnitude_file = os.path.join(subject_folder, subject_id + "_magnitudes.csv")

    sensors_data_frame = synchronize_sensors(cwas, sync_file, nrows=max_rows, sync_fix=pre_conversion_fix)

    if sampling_frequency == 200:
        sensors_data_frame = sensors_data_frame[::2]
        print(len(sensors_data_frame))
        sensors_data_frame.index = range(len(sensors_data_frame))
        print("Original sampling frequency of 200 Hz reduced to 100")
        sampling_frequency = 100

    sensors_data_frame.rename(columns={0: "Time"}, inplace=True)

    number_of_columns = sensors_data_frame.shape[1]
    x_column_numbers = [i for i in range(1, number_of_columns, 3)]

    magnitudes = []

    if shifts is None:
        shifts = [0] * number_of_columns

    for i, s in zip(x_column_numbers, shifts):
        this_sensors_columns = [i + j for j in range(3)]
        if not s == 0:
            shifted = sensors_data_frame.shift(-s)
            for col in this_sensors_columns:
                sensors_data_frame[col] = shifted[col]
        magnitudes.append(np.linalg.norm(sensors_data_frame[this_sensors_columns], axis=1))

    magnitudes_data_frame = pd.DataFrame(np.vstack(magnitudes).transpose())

    if sensor_label_sync_index is None:
        sensor_label_sync_index = find_claps(magnitudes_data_frame[0], sampling_frequency, mph=mph, valley=False,
                                             required_claps=starting_drops)[0][1]
    print("Heel drops index:", sensor_label_sync_index)
    start_peaks_time = sensors_data_frame["Time"][sensor_label_sync_index]
    labeled_area_duration = annotations["end"].iloc[-1]
    labeled_area_endtime = start_peaks_time + pd.Timedelta(seconds=labeled_area_duration)

    indices_of_labeled_data_points = (start_peaks_time <= sensors_data_frame["Time"]) & (
        sensors_data_frame["Time"] < labeled_area_endtime)

    column_rename_dict = dict([(i, os.path.split(c)[1]) for i, c in zip(range(number_of_columns), cwas)])
    magnitudes_data_frame.rename(columns=column_rename_dict, inplace=True)
    plot_start = sensor_label_sync_index - 9 * sampling_frequency
    plot_end = sensor_label_sync_index + 1 * sampling_frequency
    ax = magnitudes_data_frame[plot_start:plot_end].plot(title=subject_id + " heel drops", figsize=(10, 4), fontsize=8)
    ax.axvline(sensor_label_sync_index, linestyle="dashed")
    ax.legend(prop={'size': 8})
    plt.savefig(os.path.join(output_folder, subject_id + "_heeldrops.png"))
    plt.close()

    sensors_data_frame = sensors_data_frame[indices_of_labeled_data_points]
    sensors_data_frame.reset_index(inplace=True, drop=True)

    absolute_ends = pd.to_timedelta(annotations["end"], unit="s") + start_peaks_time
    activity_numbers = convert_string_labels_to_numbers(annotations["type"])
    labels = make_labels_for_timestamps(sensors_data_frame["Time"], activity_numbers, absolute_ends)
    labels = pd.Series(labels)

    label_file = os.path.join(output_folder, subject_id + "_labels.csv")

    labels.to_csv(label_file, header=False, index=False)
    labels.index += sensor_label_sync_index

    magnitudes_data_frame = pd.concat([magnitudes_data_frame, labels], axis=1)
    magnitudes_data_frame.to_csv(magnitude_file, header=False, index=False)

    for i, cwa_file_path in zip(x_column_numbers, cwas):
        columns_to_save = [i + j for j in range(3)]
        sub_data_frame = sensors_data_frame[columns_to_save]
        cwa_file_name = os.path.split(cwa_file_path)[1]
        output_file_name = os.path.splitext(cwa_file_name)[0] + ".csv"
        output_file_path = os.path.join(output_folder, output_file_name)
        sub_data_frame.to_csv(output_file_path, header=False, index=False)

    if clean_up:
        os.remove(sync_file)
        os.remove(magnitude_file)


if __name__ == "__main__":
    print(SUBJECT_DATA_LOCATION)

    config = configparser.ConfigParser(allow_no_value=True)
    config.read_file(open(os.path.join(SUBJECT_DATA_LOCATION, "config.cfg")))

    subject_id_template = config["DEFAULT"].get("subject_id_template")
    try:
        non_existent_ids = [int(i) for i in config["DEFAULT"].get("non_existent_ids").split(",")]
    except AttributeError:
        non_existent_ids = []

    for i in [21]:
        if i in non_existent_ids:
            continue

        s_id = subject_id_template.format(i)
        print(s_id)

        subject_config = config[s_id]
        master_substring = subject_config.get("master_substring")

        subject_folder = os.path.join(SUBJECT_DATA_LOCATION, s_id)
        # Load files
        slave_cwas = []
        master_cwa = []

        for root, _, files in os.walk(subject_folder):
            for f in sorted(files):
                if f.endswith(".cwa"):
                    if master_substring in f:
                        master_cwa = [os.path.join(root, f)]
                    else:
                        slave_cwas.append(os.path.join(root, f))

        cwas = master_cwa + slave_cwas

        try:
            start_index = subject_config.getint("annotation_start_index")
        except TypeError:
            start_index = None

        try:
            shifts = [int(i) for i in subject_config.get("shifts").split(", ")]
        except AttributeError:
            shifts = None

        extract(s_id, cwas, pre_conversion_fix=subject_config.getboolean("pre_conversion_fix"),
                mph=subject_config.getfloat("mph"), clean_up=False, sensor_label_sync_index=start_index, shifts=shifts,
                starting_drops=subject_config.getint("start_drops"),
                sampling_frequency=subject_config.getint("sampling_frequency"),
                max_hours_to_read=subject_config.getint("max_hours_to_read"))
