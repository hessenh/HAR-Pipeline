# coding: utf-8
from __future__ import division, print_function

import glob
from time import time

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


def find_first_and_second_clap_times_from_sensor_channel(channel_data, mph=5, valley=True):
    peaks = detect_peaks(channel_data, mph=mph, valley=valley)

    claps = find_claps_from_peaks(peaks)
    end_of_first_claps = claps[0][1]
    start_of_second_claps = claps[1][0]

    return end_of_first_claps, start_of_second_claps


def combine_event_files_into_one_and_save(folder, s_id):
    path = folder + '/' + s_id + "_FreeLiving_Event_"

    filenames = sorted(glob.glob(path + "*.txt"))

    dfs = []
    rolling_offset = 0
    first = True
    for filename in filenames:
        new = pd.read_csv(filename, sep='\t')
        if first:
            dfs.append(new)
            rolling_offset = new.tail(1).iloc[0]['end']
            first = False
        else:
            new['start'] = new['start'] + rolling_offset
            new['end'] = new['end'] + rolling_offset
            rolling_offset = new.tail(1).iloc[0]['end']
            dfs.append(new)

    events_data_frame = pd.concat(dfs, ignore_index=True)

    events_data_frame = events_data_frame[['start', 'end', 'duration', 'type']]

    events_data_frame.to_csv(folder + '/' + s_id + "_events.csv")

    return events_data_frame


def make_labels_for_points_in_time(data_point_times, labels_list, end_times_list):
    labels_for_data_points = []

    index_of_data_point_time_to_be_examined = 0

    size_of_times = len(data_point_times)

    for label, end_time in zip(labels_list, end_times_list):
        while index_of_data_point_time_to_be_examined < size_of_times \
                and data_point_times[index_of_data_point_time_to_be_examined] <= end_time:
            labels_for_data_points.append(label)
            index_of_data_point_time_to_be_examined += 1

    return labels_for_data_points


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
        "non-vigorous activity": 17
    }

    return [label_to_number_dict[label] for label in label_list]


def main():
    subject_id = '009'
    master_sensor_codeword = "UPPERBACK"
    thigh_sensor_codeword = "THIGH"
    subject_folder = 'private_data/conversion_scripts/annotated_data/' + subject_id
    folder_and_subject_id = subject_folder + '/' + subject_id

    sampling_frequency = 100  # Sampling frequency in hertz

    print("Reading annotations ...")
    a = time()

    try:
        events = pd.read_csv(folder_and_subject_id + '_events.csv', sep=',')
    except IOError:
        print("Combined annotations file not found. Creating annotation file from separate files.")
        events = combine_event_files_into_one_and_save(subject_folder, subject_id)

    b = time()
    print("Read annotations in", b - a)

    events = events[['start', 'end', 'duration', 'type']]

    print("\nFinding heel drops in the data")
    starting_heel_drops = 3
    ending_heel_drop_index = starting_heel_drops - 1

    heel_drop_events = events.loc[events['type'] == 'heel drop'][ending_heel_drop_index:]
    offset = heel_drop_events.iloc[0]['end']
    events = events.drop(events[events.end <= offset].index)

    events['start'] = events['start'] - offset
    events['end'] = events['end'] - offset

    # remove the last heel drops
    heel_drop_events = events.loc[events['type'] == 'heel drop'][0:]
    last_heeldrop = heel_drop_events.iloc[0]['start']
    events = events.drop(events[events.end > last_heeldrop].index)

    print("First heel drops end at", offset, "seconds")
    print("Length of annotated data after removing heel drops:", last_heeldrop, "seconds")

    print("Reading sensor data ...")

    a = time()
    synchronized_csv = subject_folder + '/' + subject_id + "_synchronized.csv"
    try:
        sensor_readings = pd.read_csv(synchronized_csv, parse_dates=[0],
                                      header=None)
    except IOError:
        print("Synchronized sensor data not found. Creating synchronized data.")
        master_cwa = glob.glob(subject_folder + "/*_" + master_sensor_codeword + "_*" + subject_id + ".cwa")[0]
        slave_cwa = glob.glob(subject_folder + "/*_" + thigh_sensor_codeword + "_*" + subject_id + ".cwa")[0]
        create_synchronized_file_for_subject(master_cwa, slave_cwa, synchronized_csv)
        print("Conversion finished. Reading converted data.")
        sensor_readings = pd.read_csv(synchronized_csv, parse_dates=[0],
                                      header=None)

    b = time()
    print("Read sensor data in", b - a)

    set_header_names_for_data_generated_by_omconvert_script(sensor_readings)

    print("Finding claps ...")
    # First find the value range between the 3 starting and ending hand claps
    sx = sensor_readings['Slave-X']

    first_clap_index, ending_clap_index = find_first_and_second_clap_times_from_sensor_channel(sx, mph=5,
                                                                                               valley=True)
    print("Found claps in sensor data")

    session_length = (ending_clap_index - first_clap_index) / sampling_frequency
    print("Session length between start and end claps:", session_length, "seconds")

    #  create the subset
    labeled_sensor_readings = sensor_readings[first_clap_index:ending_clap_index]
    print("Created 'labeled_sensor_readings' from sensor data")

    # Adding a column that counts the seconds between the beginning of the time series and the current data point.
    # This is later used to find the matching label in the events file
    labeled_sensor_readings = labeled_sensor_readings.reset_index(drop=True)
    start_time = labeled_sensor_readings.Time[0]

    print("Making 'seconds_list' ...")
    a = time()
    seconds_list = [(row - start_time).total_seconds() for row in labeled_sensor_readings.Time]
    b = time()
    print("Made seconds list in", b - a, "seconds")

    print("Creating label list for each data point ...")
    events_labels = events.type
    events_end_times = events.end

    a = time()
    labels_for_each_data_point_in_seconds_list = make_labels_for_points_in_time(seconds_list, events_labels,
                                                                                events_end_times)
    labels_for_each_data_point_in_seconds_list = convert_string_labels_to_numbers(
        labels_for_each_data_point_in_seconds_list)

    b = time()
    print("Created label list in", b - a, "seconds")

    assert len(labels_for_each_data_point_in_seconds_list) == labeled_sensor_readings.shape[0]

    print("Creating label column ...")
    a = time()
    labeled_sensor_readings['label'] = pd.Series(labels_for_each_data_point_in_seconds_list,
                                                 index=labeled_sensor_readings.index)
    b = time()
    print("Created 'label' column in", b - a, "seconds")

    # Write the results to csv
    print("Writing results to CSVs")
    a = time()

    master_csv_file_path = folder_and_subject_id + "_Axivity_BACK_Back.csv"
    slave_csv_file_path = folder_and_subject_id + "_Axivity_THIGH_Right.csv"
    label_csv_file_path = folder_and_subject_id + "_GoPro_LAB_All.csv"

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
