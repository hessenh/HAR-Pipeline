# coding: utf-8
from __future__ import division, print_function

import glob
import json

import matplotlib
import os
import pandas as pd

from external_modules.detect_peaks import detect_peaks
from raw_data_conversion.conversion import create_synchronized_file_for_subject, \
    set_header_names_for_data_generated_by_omconvert_script
from tools.pandas_helpers import write_selected_columns_to_file

matplotlib.use("Agg")  # Choose non-interactive background
import matplotlib.pyplot as plt

SUBJECT_DATA_LOCATION = os.path.join('.', 'private_data', 'annotated_data')
folder_prefix = ""


class SubjectConfiguration:
    def __init__(self, path, subject_id, subject_folder):
        self.subject_folder = subject_folder
        self.id = subject_id
        self.amplitude = 5
        self.start_peaks = 3
        self.end_peaks = 3
        self.master_codeword = "THIGH"
        self.slave_codewords = ["BACK"]
        self.sampling_frequency = 100

        try:
            with open(path) as json_file:
                config_dict = json.load(json_file)
                self.amplitude = config_dict["heel_drop_amplitude"]
                self.start_peaks = config_dict["starting_heel_drops"]
                self.end_peaks = config_dict["ending_heel_drops"]
                self.slave_codewords = config_dict["slave_sensor_codewords"]
                self.sampling_frequency = config_dict["sampling_frequency"]
        except IOError:
            print("Could not find config file. Returning to default configurations")


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


def find_claps(data_points, sampling_frequency, mph=5, valley=True, required_claps=3):
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

    for label, end_time in zip(labels_list, end_times_list):
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

    for i, label in enumerate(label_list):
        if type(label) is not type("santheunsth"):
            print(i)
            quit()

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


def extract_back_and_thigh(subject_id, sync_fix=True, clean_up=True):
    subject_folder = os.path.join(SUBJECT_DATA_LOCATION, subject_id)
    config_file = os.path.join(subject_folder, subject_id + "_config.json")
    sc = SubjectConfiguration(config_file, subject_id, subject_folder)

    sampling_frequency = 100  # in hertz

    annotations_glob = os.path.join(subject_folder, sc.id + "_FreeLiving_Event_*.txt")

    annotation_files = sorted(glob.glob(annotations_glob))
    annotations = combine_annotation_files(annotation_files)

    annotations = remove_annotations_outside_heel_drops(annotations, sc.start_peaks, sc.end_peaks)

    for slave_sensor_codeword in sc.slave_codewords:
        print("\nReading sensor data ...")

        master_codeword = sc.master_codeword
        synchronized_csv = sc.id + "_" + master_codeword + '_' + slave_sensor_codeword + "_synchronized.csv"
        synchronized_complete_path = os.path.join(subject_folder, synchronized_csv)

        csv_output_folder = os.path.join(subject_folder, folder_prefix + slave_sensor_codeword)

        if not os.path.exists(csv_output_folder):
            os.makedirs(csv_output_folder)

        if not os.path.exists(synchronized_complete_path):
            print("Synchronized file not found. Creating synchronized data.")
            master_cwa = glob.glob(os.path.join(subject_folder, "*_" + master_codeword + "_*" + sc.id + ".cwa"))[0]
            slave_cwa = glob.glob(os.path.join(subject_folder, "*_" + slave_sensor_codeword + "_*" + sc.id + ".cwa"))[0]
            create_synchronized_file_for_subject(master_cwa, slave_cwa, synchronized_complete_path, sync_fix=sync_fix)
            print("Conversion finished.")

        print("Reading", synchronized_complete_path)
        sensor_readings = pd.read_csv(synchronized_complete_path, parse_dates=[0], header=None)

        heel_drop_column = 'Master-X'

        if sc.sampling_frequency == 200:
            sensor_readings = sensor_readings[::2].reindex()
            print("Original sampling frequency of 200 Hz reduced to 100")

        set_header_names_for_data_generated_by_omconvert_script(sensor_readings)

        print("Finding claps ...")

        labeled_sensor_readings = create_labeled_data_frame(sensor_readings, annotations, heel_drop_column, sc,
                                                            sampling_frequency, synchronized_csv)

        # Write the results to csv
        print("Writing results to CSVs")

        x_axes = labeled_sensor_readings[["Master-X", "Slave-X"]]
        last_peak = detect_peaks(x_axes["Master-X"], mph=2, valley=True)[-1]
        x_axes[last_peak - 900:last_peak + 100].plot(title=slave_sensor_codeword, figsize=(25, 4),
                                                     colormap=plt.get_cmap("bwr"))
        plt.savefig(os.path.join(csv_output_folder, slave_sensor_codeword + ".png"))
        plt.close()

        master_csv = os.path.join(csv_output_folder, sc.id + "_Axivity_" + master_codeword + "_Right.csv")
        slave_csv = os.path.join(csv_output_folder, sc.id + "_Axivity_" + "BACK" + "_Back.csv")
        label_csv = os.path.join(csv_output_folder, sc.id + "_GoPro_LAB_All.csv")

        master_columns = ["Master-X", "Master-Y", "Master-Z"]
        slave_columns = ["Slave-X", "Slave-Y", "Slave-Z"]
        label_columns = ["label"]

        csv_paths = [master_csv, slave_csv, label_csv]
        columns = [master_columns, slave_columns, label_columns]

        for p, col in zip(csv_paths, columns):
            write_selected_columns_to_file(labeled_sensor_readings, col, p)

        print("Wrote 'labeled_sensor_readings' to CSV files")
        if clean_up:
            os.remove(synchronized_complete_path)


def create_labeled_data_frame(sensor_readings, annotations, heel_drop_column, subject_configuration,
                              sampling_frequency=100, synced_filename="NO FILENAME_PROVIDED"):
    sensor_readings = remove_sensor_data_before_heel_drops(sensor_readings, heel_drop_column, subject_configuration,
                                                           sampling_frequency, synced_filename)
    labeled_sensor_readings = create_labeled_sensor_readings(sensor_readings, sampling_frequency, annotations)
    return labeled_sensor_readings


def create_labeled_sensor_readings(sensor_readings, sampling_frequency, annotations):
    print("Extracting timestamps to create labels for ...")

    time_column = sensor_readings['Time']
    annotations_duration = annotations.tail(1)['end']
    extra_sampling_factor = 1.05
    timestamps = extract_timestamps(time_column, annotations_duration, extra_sampling_factor, sampling_frequency)

    print("Creating label list for sensor data ...")
    annotation_texts = annotations.type
    annotation_end_times = annotations.end

    timestamp_labels = make_labels_for_timestamps(timestamps, annotation_texts, annotation_end_times)
    timestamp_labels = convert_string_labels_to_numbers(timestamp_labels)

    labeled_sensor_readings = sensor_readings[:len(timestamp_labels)].reindex()
    print("Length of label list:", len(timestamp_labels))

    print("Creating label column ...")
    labeled_sensor_readings['label'] = pd.Series(timestamp_labels, index=labeled_sensor_readings.index)

    return labeled_sensor_readings


def extract_timestamps(time_column, annotations_duration, extra_sampling_factor=1.05, sampling_frequency=100):
    start = time_column[0]
    end = int(annotations_duration * extra_sampling_factor * sampling_frequency)
    timestamps = [(current_time - start).total_seconds() for current_time in time_column[:end]]
    return timestamps


def remove_sensor_data_before_heel_drops(data_frame, most_affected_column, subject_configuration,
                                         sampling_frequency=100, synced_filename="NO FILENAME PROVIDED"):
    heel_drop_channel_data = data_frame[most_affected_column]
    master_claps = find_claps(heel_drop_channel_data, sampling_frequency, mph=subject_configuration.amplitude,
                              valley=True, required_claps=subject_configuration.start_peaks)
    end_of_first_drops = master_claps[0][1]

    offset = end_of_first_drops - 1000

    slave_claps = find_claps(data_frame["Slave-X"][offset:], sampling_frequency, mph=subject_configuration.amplitude,
                             valley=True, required_claps=subject_configuration.start_peaks)
    slave_end = slave_claps[0][1] + offset
    print("Found heel drops in sensor data")
    print("End of first heel drops at", end_of_first_drops / sampling_frequency, "seconds, index", end_of_first_drops)

    drops_diff = slave_end - end_of_first_drops
    if abs(drops_diff) > 1:
        shifted = data_frame.shift(-drops_diff)
        for col in ["Slave-X", "Slave-Y", "Slave-Z"]:
            data_frame[col] = shifted[col]
    print("DIFF BETWEEN SLAVE AND MASTER:", drops_diff)

    # Save a plot of the axes
    x_axes = data_frame[["Master-X", "Slave-X"]]
    plot_start = end_of_first_drops - 9 * sampling_frequency
    plot_end = end_of_first_drops + 1 * sampling_frequency
    x_axes[plot_start:plot_end].plot(title=synced_filename, figsize=(25, 4), colormap=plt.get_cmap("bwr"))
    plt.savefig(os.path.join(subject_configuration.subject_folder, folder_prefix + synced_filename.split("_")[2],
                             os.path.splitext(synced_filename)[0] + ".png"))
    plt.close()

    data_frame = data_frame[end_of_first_drops:]
    data_frame = data_frame.reset_index(drop=True)
    print("Removed sensor readings up to and including heel drops")
    return data_frame


if __name__ == "__main__":
    for i in [6]:
        if i == 7:
            continue
        s_id = "{0:0>3}".format(i)
        print(s_id)
        extract_back_and_thigh(s_id, sync_fix=True)
