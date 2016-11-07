# coding: utf-8
from __future__ import division, print_function

import glob
from time import time
import json
import os

import pandas as pd

from external_modules.detect_peaks import detect_peaks
from tools.pandas_helpers import write_selected_columns_to_file


def create_synchronized_file_for_subject(master_cwa, slave_cwa, output_csv, clean_up=True):
    from os.path import splitext
    import subprocess

    omconvert = "./private_data/conversion_scripts/omconvert/omconvert"
    timesync = "./private_data/conversion_scripts/timesync/timesync"

    master_wav = splitext(master_cwa)[0] + ".wav"
    slave_wav = splitext(slave_cwa)[0] + ".wav"

    # Create wav for master sensor
    subprocess.call([omconvert, master_cwa, "-out", master_wav])

    # Create wav for slave sensor
    subprocess.call([omconvert, slave_cwa, "-out", slave_wav])

    # Synchronize them and make them a CSV
    subprocess.call([timesync, master_wav, slave_wav, "-csv", output_csv])

    if clean_up:
        print("Deleting wav files")
        subprocess.call(["rm", master_wav])
        subprocess.call(["rm", slave_wav])


def find_claps_from_peaks(peak_array, required_claps=3, sampling_frequency=100, min_interval=0.15, max_interval=8.0):
    claps = []
    start_peak = peak_array[0]
    previous_peak = start_peak
    clap_count = 1
    max_data_points_between_peaks = max_interval * sampling_frequency
    min_data_points_between_peaks = min_interval * sampling_frequency

    for peak in peak_array:
        time_interval = peak - previous_peak

        if min_data_points_between_peaks < time_interval <= max_data_points_between_peaks:
            clap_count += 1
        elif time_interval > max_data_points_between_peaks:
            start_peak = peak
            clap_count = 1  # Added resetting peak count when time interval too large from last peak. -- Eirik
        else:
            # Interval is less than minimum interval; still part of an already registered peak.
            continue  # Skip to next loop iteration

        previous_peak = peak

        if clap_count == required_claps:
            start_and_end_of_claps = (start_peak, peak)
            claps.append(start_and_end_of_claps)
            start_peak = peak  # Reset
            clap_count = 1

    return claps


def set_header_names_for_data_generated_by_omconvert_script(data_frame):
    data_frame.rename(
        columns={0: 'Time', 1: 'Master-X', 2: 'Master-Y', 3: 'Master-Z', 4: 'Slave-X', 5: 'Slave-Y', 6: 'Slave-Z'},
        inplace=True)


def find_claps_from_sensor_data(channel_data, sampling_frequency, mph=5, valley=True,
                                required_claps=3):
    peaks = detect_peaks(channel_data, mph=mph, valley=valley)

    clap_times = find_claps_from_peaks(peaks, required_claps=required_claps, sampling_frequency=sampling_frequency)

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
        "Car": 18                           # TODO: Check with HAR group to see if this labeling of Car is all right.
    }

    return [label_to_number_dict[label] for label in label_list]


def extract_relevant_events(events_csv, starting_heel_drops, ending_heel_drops, event_files_glob_expression):
    print("Reading events ...")

    a = time()
    try:
        events = pd.read_csv(events_csv, sep=',')
    except IOError:
        print("Combined events file not found. Creating events file from separate files.")
        events = combine_event_files_into_one_and_save(event_files_glob_expression, events_csv)
    b = time()
    print("Read events in", b - a)

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


def main():
    subject_id = '008'

    master_sensor_codewords, slave_sensor_codeword = ["BACK"], "THIGH"
    starting_heel_drops, ending_heel_drops = 3, 3
    heel_drop_amplitude = 5

    subject_folder = 'private_data/conversion_scripts/annotated_data/' + subject_id
    folder_and_subject_id = subject_folder + '/' + subject_id

    sampling_frequency = 100  # Sampling frequency in hertz

    events_csv = folder_and_subject_id + '_events.csv'
    event_files_glob_expression = folder_and_subject_id + "_FreeLiving_Event_*.txt"

    try:
        with open(folder_and_subject_id + "_config.json") as json_file:
            config_dict = json.load(json_file)
            heel_drop_amplitude = config_dict["heel_drop_amplitude"]
            starting_heel_drops = config_dict["starting_heel_drops"]
            ending_heel_drops = config_dict["ending_heel_drops"]
            master_sensor_codewords = config_dict["master_sensor_codewords"]

    except IOError:
        print("Could not find config file. Returning to default configurations")

    events = extract_relevant_events(events_csv, starting_heel_drops, ending_heel_drops, event_files_glob_expression)

    for master_sensor_codeword in master_sensor_codewords:
        print("\nReading sensor data ...")

        a = time()
        sync_filename = subject_id + "_" + master_sensor_codeword + "_" + slave_sensor_codeword + "_synchronized.csv"
        sync_path = subject_folder + '/' + sync_filename

        try:
            sensor_readings = pd.read_csv(sync_path, parse_dates=[0], header=None)
        except IOError:
            print("Synchronized sensor data file", sync_path, "not found. Creating synchronized data.")
            master_cwa = glob.glob(subject_folder + "/*_" + master_sensor_codeword + "_*" + subject_id + ".cwa")[0]
            slave_cwa = glob.glob(subject_folder + "/*_" + slave_sensor_codeword + "_*" + subject_id + ".cwa")[0]
            create_synchronized_file_for_subject(master_cwa, slave_cwa, sync_path)
            print("Conversion finished. Reading converted data.")
            sensor_readings = pd.read_csv(sync_path, parse_dates=[0], header=None)

        b = time()
        print("Read sensor data in", b - a)

        set_header_names_for_data_generated_by_omconvert_script(sensor_readings)

        print("Finding claps ...")
        # First find the value range between the 3 starting and ending hand claps
        sx = sensor_readings['Slave-X']

        claps = find_claps_from_sensor_data(sx,
                                            sampling_frequency,
                                            mph=heel_drop_amplitude,
                                            valley=True,
                                            required_claps=starting_heel_drops)

        first_clap_index = claps[0][1]
        print("Found heel drops in sensor data")

        print("End of first heel drops at", first_clap_index / sampling_frequency, "seconds, index", first_clap_index)

        sensor_readings = sensor_readings[first_clap_index:]
        sensor_readings = sensor_readings.reset_index(drop=True)
        print("Removed sensor readings up to and including heel drops")

        start_time = sensor_readings.Time[0]

        print("Extracting sensor data to create labels for ...")
        a = time()

        events_duration = events.tail(1)['end']

        extra_sampling_from_sensor_readings = 1.05

        seconds_list = [(current_time - start_time).total_seconds() for current_time in
                        sensor_readings.Time[
                        :int(events_duration * extra_sampling_from_sensor_readings * sampling_frequency)]]
        b = time()
        print("Extracted data in", b - a, "seconds")

        print("Creating label list for sensor data ...")
        events_labels = events.type
        events_end_times = events.end

        a = time()
        labels_for_each_data_point_in_seconds_list = make_labels_for_timestamps(seconds_list, events_labels,
                                                                                events_end_times)
        labels_for_each_data_point_in_seconds_list = convert_string_labels_to_numbers(
            labels_for_each_data_point_in_seconds_list)

        b = time()
        print("Created label list in", b - a, "seconds")

        labeled_sensor_readings = sensor_readings[:len(labels_for_each_data_point_in_seconds_list)].reindex()
        print("Length of label list:", len(labels_for_each_data_point_in_seconds_list))

        print("Creating label column ...")
        a = time()
        labeled_sensor_readings['label'] = pd.Series(labels_for_each_data_point_in_seconds_list,
                                                     index=labeled_sensor_readings.index)
        b = time()
        print("Created 'label' column in", b - a, "seconds")

        # Write the results to csv
        print("Writing results to CSVs")
        a = time()

        path_to_csv_folder_using_this_master_sensor = subject_folder + "/" + master_sensor_codeword

        if not os.path.exists(path_to_csv_folder_using_this_master_sensor):
            os.makedirs(path_to_csv_folder_using_this_master_sensor)

        master_csv_file_path = path_to_csv_folder_using_this_master_sensor + "/" + subject_id + "_Axivity_" + master_sensor_codeword + "_Back.csv"
        slave_csv_file_path = path_to_csv_folder_using_this_master_sensor + "/" + subject_id + "_Axivity_" + slave_sensor_codeword + "_Right.csv"
        label_csv_file_path = path_to_csv_folder_using_this_master_sensor + "/" + subject_id + "_GoPro_LAB_All.csv"

        master_columns = ["Master-X", "Master-Y", "Master-Z"]
        slave_columns = ["Slave-X", "Slave-Y", "Slave-Z"]
        label_columns = ["label"]

        write_selected_columns_to_file(labeled_sensor_readings, master_columns, master_csv_file_path)
        write_selected_columns_to_file(labeled_sensor_readings, slave_columns, slave_csv_file_path)
        write_selected_columns_to_file(labeled_sensor_readings, label_columns, label_csv_file_path)

        b = time()

        print("Wrote 'labeled_sensor_readings' to CSV files in", b - a)


if __name__ == "__main__":
    main()
