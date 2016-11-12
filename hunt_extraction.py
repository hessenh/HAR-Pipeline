# coding: utf-8
from __future__ import division, print_function

import glob
import json
from time import time

import os
import pandas as pd

from external_modules.detect_peaks import detect_peaks
from raw_data_conversion.conversion import convert_subject_raw_file, create_synchronized_file_for_subject, \
    set_header_names_for_data_generated_by_omconvert_script
from tools.pandas_helpers import write_selected_columns_to_file

SUBJECT_DATA_LOCATION = 'private_data/annotated_data/'


def find_repeated_peaks(peak_array, required_peaks=3, sampling_frequency=100, min_interval=0.15, max_interval=8.0):
    starts_and_stops = []
    start_peak = peak_array[0]
    previous_peak = start_peak
    peak_count = 1
    max_data_points_between_peaks = max_interval * sampling_frequency
    min_data_points_between_peaks = min_interval * sampling_frequency

    for peak in peak_array:
        time_interval = peak - previous_peak

        if min_data_points_between_peaks < time_interval <= max_data_points_between_peaks:
            peak_count += 1
        elif time_interval > max_data_points_between_peaks:
            start_peak = peak
            peak_count = 1  # Added resetting peak count when time interval too large from last peak. -- Eirik
        else:
            # Interval is less than minimum interval; still part of an already registered peak.
            continue  # Skip to next loop iteration

        previous_peak = peak

        if peak_count == required_peaks:
            start_and_end_of_sequence = (start_peak, peak)
            starts_and_stops.append(start_and_end_of_sequence)
            start_peak = peak  # Reset
            peak_count = 1

    return starts_and_stops


def find_claps_from_sensor_data(channel_data, sampling_frequency, mph=5, valley=True,
                                required_claps=3):
    peaks = detect_peaks(channel_data, mph=mph, valley=valley)

    clap_times = find_repeated_peaks(peaks, required_peaks=required_claps, sampling_frequency=sampling_frequency)

    return clap_times


def combine_event_files_into_one_and_save(event_files_glob_expression, output_file):
    filenames = sorted(glob.glob(event_files_glob_expression))

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

    events_data_frame = pd.concat(dfs, ignore_index=True)

    events_data_frame = events_data_frame[['start', 'end', 'duration', 'type']]

    events_data_frame.to_csv(output_file)

    return events_data_frame


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
        "Car": 18  # TODO: Check with HAR group to see if this labeling of Car is all right.
    }

    return [label_to_number_dict[label] for label in label_list]


def extract_relevant_events(events_csv, starting_heel_drops, ending_heel_drops, event_files_glob_expression):
    print("Reading events ...")

    try:
        events = pd.read_csv(events_csv, sep=',')
    except IOError:
        print("Combined events file not found. Creating events file from separate files.")
        events = combine_event_files_into_one_and_save(event_files_glob_expression, events_csv)

    events = events[['start', 'end', 'duration', 'type']]

    print("\nFinding heel drops in the events file")
    if starting_heel_drops:
        last_starting_heel_drop_index = starting_heel_drops - 1

        heel_drop_events = events.loc[events['type'] == 'heel drop'][last_starting_heel_drop_index:]
        offset = heel_drop_events.iloc[0]['end']
        events = events.drop(events[events.end <= offset].index)

        events['start'] = events['start'] - offset
        events['end'] = events['end'] - offset
        print("First heel drops end at", offset, "seconds")

    if ending_heel_drops:
        heel_drop_events = events.loc[events['type'] == 'heel drop'][0:]
        last_heel_drop = heel_drop_events.iloc[0]['start']
        events = events.drop(events[events.end > last_heel_drop].index)

    end_ = events.tail(1)['end'].iloc[0]
    print("Length of annotated data after removing heel drops:", end_, "seconds")

    return events


def extract_back_and_thigh(subject_id='008'):
    master_sensor_codewords, slave_sensor_codeword = ["BACK"], "THIGH"
    starting_heel_drops, ending_heel_drops = 3, 3
    heel_drop_amplitude = 5

    subject_folder = SUBJECT_DATA_LOCATION + subject_id
    folder_and_subject_id = subject_folder + '/' + subject_id

    original_sampling_frequency = 100
    sampling_frequency = 100  # in hertz

    events_csv = folder_and_subject_id + '_events.csv'
    event_files_glob_expression = folder_and_subject_id + "_FreeLiving_Event_*.txt"

    try:
        with open(folder_and_subject_id + "_config.json") as json_file:
            config_dict = json.load(json_file)
            heel_drop_amplitude = config_dict["heel_drop_amplitude"]
            starting_heel_drops = config_dict["starting_heel_drops"]
            ending_heel_drops = config_dict["ending_heel_drops"]
            master_sensor_codewords = config_dict["master_sensor_codewords"]
            original_sampling_frequency = config_dict["sampling_frequency"]
    except IOError:
        print("Could not find config file. Returning to default configurations")

    events = extract_relevant_events(events_csv, starting_heel_drops, ending_heel_drops, event_files_glob_expression)

    synchronized_files = []

    for master_sensor_codeword in master_sensor_codewords:
        print("\nReading sensor data ...")

        sync_filename = subject_id + "_" + master_sensor_codeword + "_" + slave_sensor_codeword + "_synchronized.csv"
        sync_path = subject_folder + '/' + sync_filename

        if not os.path.isfile(sync_path):
            print("Synchronized sensor data file", sync_path, "not found. Creating synchronized data.")
            master_cwa = glob.glob(subject_folder + "/*_" + master_sensor_codeword + "_*" + subject_id + ".cwa")[0]
            slave_cwa = glob.glob(subject_folder + "/*_" + slave_sensor_codeword + "_*" + subject_id + ".cwa")[0]
            create_synchronized_file_for_subject(master_cwa, slave_cwa, sync_path)
            print("Conversion finished.")

        synchronized_files.append(sync_path)

    for sync_path, master_sensor_codeword in zip(synchronized_files, master_sensor_codewords):
        print("Reading", sync_path)
        a = time()
        sensor_readings = pd.read_csv(sync_path, parse_dates=[0], header=None)
        b = time()

        heel_drop_column = 'Slave-X'

        if original_sampling_frequency == 200:
            sensor_readings = sensor_readings[::2].reindex()
            print("Original sampling frequency of 200 Hz reduced to 100")

        print("Read in", b - a, "seconds")

        set_header_names_for_data_generated_by_omconvert_script(sensor_readings)

        print("Finding claps ...")
        labeled_sensor_readings = create_labeled_data_frame(sensor_readings, events, heel_drop_column,
                                                            heel_drop_amplitude, starting_heel_drops,
                                                            sampling_frequency)

        # Write the results to csv
        print("Writing results to CSVs")
        a = time()

        csv_output_folder = subject_folder + "/" + master_sensor_codeword

        master_csv = csv_output_folder + "/" + subject_id + "_Axivity_" + master_sensor_codeword + "_Back.csv"
        slave_csv = csv_output_folder + "/" + subject_id + "_Axivity_" + slave_sensor_codeword + "_Right.csv"
        label_csv = csv_output_folder + "/" + subject_id + "_GoPro_LAB_All.csv"

        master_columns = ["Master-X", "Master-Y", "Master-Z"]
        slave_columns = ["Slave-X", "Slave-Y", "Slave-Z"]
        label_columns = ["label"]

        csv_paths = [master_csv, slave_csv, label_csv]
        columns = [master_columns, slave_columns, label_columns]

        if not os.path.exists(csv_output_folder):
            os.makedirs(csv_output_folder)

        for p, col in zip(csv_paths, columns):
            write_selected_columns_to_file(labeled_sensor_readings, col, p)

        b = time()

        print("Wrote 'labeled_sensor_readings' to CSV files in", b - a)


def create_labeled_data_frame(sensor_readings, events, heel_drop_column, heel_drop_amplitude, starting_heel_drops=3,
                              sampling_frequency=100):
    sensor_readings = remove_sensor_data_before_heel_drops(sensor_readings, heel_drop_column, starting_heel_drops,
                                                           heel_drop_amplitude, sampling_frequency)
    labeled_sensor_readings = create_labeled_sensor_readings(sensor_readings, sampling_frequency, events)
    return labeled_sensor_readings


def create_labeled_sensor_readings(sensor_readings, sampling_frequency, events):
    print("Extracting timestamps to create labels for ...")

    time_column = sensor_readings['Time']
    events_duration = events.tail(1)['end']
    extra_sampling_factor = 1.05
    timestamps = extract_timestamps(time_column, events_duration, extra_sampling_factor, sampling_frequency)

    print("Creating label list for sensor data ...")
    events_labels = events.type
    events_end_times = events.end

    timestamp_labels = make_labels_for_timestamps(timestamps, events_labels, events_end_times)
    timestamp_labels = convert_string_labels_to_numbers(timestamp_labels)

    labeled_sensor_readings = sensor_readings[:len(timestamp_labels)].reindex()
    print("Length of label list:", len(timestamp_labels))

    print("Creating label column ...")
    labeled_sensor_readings['label'] = pd.Series(timestamp_labels, index=labeled_sensor_readings.index)

    return labeled_sensor_readings


def extract_timestamps(time_column, events_duration, extra_sampling_factor=1.05, sampling_frequency=100):
    start = time_column[0]
    end = int(events_duration * extra_sampling_factor * sampling_frequency)
    timestamps = [(current_time - start).total_seconds() for current_time in time_column[:end]]
    return timestamps


def remove_sensor_data_before_heel_drops(data_frame, most_affected_column, starting_heel_drops=3,
                                         heel_drop_amplitude=5, sampling_frequency=100):
    heel_drop_channel_data = data_frame[most_affected_column]
    claps = find_claps_from_sensor_data(heel_drop_channel_data,
                                        sampling_frequency,
                                        mph=heel_drop_amplitude,
                                        valley=True,
                                        required_claps=starting_heel_drops)
    first_clap_index = claps[0][1]
    print("Found heel drops in sensor data")
    print("End of first heel drops at", first_clap_index / sampling_frequency, "seconds, index", first_clap_index)
    data_frame = data_frame[first_clap_index:]
    data_frame = data_frame.reset_index(drop=True)
    print("Removed sensor readings up to and including heel drops")
    return data_frame


def extract_wrist(subject_id):
    sensor_codeword = "LEFTWRIST"

    starting_heel_drops, ending_heel_drops = 3, 0
    heel_drop_amplitude = 2

    subject_folder = SUBJECT_DATA_LOCATION + subject_id
    folder_and_subject_id = subject_folder + '/' + subject_id

    original_sampling_frequency = 100
    sampling_frequency = 100  # in hertz

    events_csv = folder_and_subject_id + '_events.csv'

    event_files_glob_expression = folder_and_subject_id + "_FreeLiving_Event_*.txt"

    events = extract_relevant_events(events_csv, starting_heel_drops, ending_heel_drops, event_files_glob_expression)

    csv_files = glob.glob(subject_folder + "/*_" + sensor_codeword + "_*" + subject_id + ".csv")

    if csv_files:
        wrist_csv = csv_files[0]
    else:
        print("Wrist CSV not found. Converting CWA file to CSV")
        wrist_cwa = glob.glob(subject_folder + "/*_" + sensor_codeword + "_*" + subject_id + ".cwa")[0]
        wrist_csv = os.path.splitext(wrist_cwa)[0] + ".csv"

        convert_subject_raw_file(wrist_cwa, csv_outfile=wrist_csv)

    sensor_readings = pd.read_csv(wrist_csv, parse_dates=[0])

    if original_sampling_frequency == 200:
        sensor_readings = sensor_readings[::2].reindex()
        print("Original sampling frequency of 200 Hz reduced to 100")

    heel_drop_column = " Accel-Y (g)"

    labeled_sensor_readings = create_labeled_data_frame(sensor_readings, events, heel_drop_column, heel_drop_amplitude,
                                                        starting_heel_drops, sampling_frequency)

    print("Writing results to CSVs")

    csv_output_folder = subject_folder + "/" + sensor_codeword
    labeled_csv = csv_output_folder + "/" + subject_id + "_Axivity_" + sensor_codeword + "_Labeled.csv"
    labeled_columns = ["Accel-X (g)", " Accel-Y (g)", " Accel-Z (g)", "label"]

    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)

    write_selected_columns_to_file(labeled_sensor_readings, labeled_columns, labeled_csv)

    print("Wrote CSV file")


if __name__ == "__main__":
    extract_back_and_thigh("002")
