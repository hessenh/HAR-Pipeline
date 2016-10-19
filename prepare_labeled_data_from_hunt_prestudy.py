# coding: utf-8
from __future__ import division, print_function

import glob

import pandas as pd
from time import time
from external_modules.detect_peaks import detect_peaks


def create_synchronized_back_and_thigh_file_for_subject(folder, s_id):
    import subprocess

    omconvert = "./private_data/conversion_scripts/omconvert/omconvert"
    timesync = "./private_data/conversion_scripts/timesync/timesync"

    back_cwa = glob.glob(folder + "/*_BACK_*" + s_id + ".cwa")[0]
    thigh_cwa = glob.glob(folder + "/*_THIGH_*" + s_id + ".cwa")[0]

    back_wav = folder + '/' + s_id + "_back.wav"
    thigh_wav = folder + '/' + s_id + "_thigh.wav"

    synchronized_csv = folder + '/' + s_id + "_thigh.resampled.csv"

    # Create wav and CSV for back sensor
    subprocess.call([omconvert, back_cwa, "-out", back_wav, "-csv-file", folder + '/' + s_id + "_back.csv"])

    # Create wav for thigh sensor
    subprocess.call([omconvert, thigh_cwa, "-out", thigh_wav])

    # Synchronize them and make them a CSV
    subprocess.call([timesync, back_wav, thigh_wav, "-csv", synchronized_csv])


def find_claps_from_peaks(peak_array, required_claps=3, sampling_frequency=100, min_interval=0.15, max_interval=8.0):
    claps = []
    start_peak = peak_array[0]
    previous_peak = start_peak
    peak_count = 0
    max_data_points_between_peaks = max_interval * sampling_frequency
    min_data_points_between_peaks = min_interval * sampling_frequency

    for peak in peak_array:
        time_interval = peak - previous_peak

        if min_data_points_between_peaks < time_interval <= max_data_points_between_peaks:
            peak_count += 1
        elif time_interval > max_data_points_between_peaks:
            start_peak = peak
            peak_count = 0  # Added resetting peak count when time interval too large from last peak. -- Eirik
            # TODO: Should peak_count be reset here? -- Eirik
        else:
            # Interval is less than minimum interval; still part of an already registered peak.
            continue  # Skip to next loop iteration

        previous_peak = peak

        if peak_count == required_claps - 1:
            start_and_end_of_claps = (start_peak, peak)
            claps.append(start_and_end_of_claps)
            start_peak = peak  # Reset
            peak_count = 0

    return claps


def add_to_column_in_data_frame(data_frame, column_name, added):
    data_frame[column_name] = data_frame.apply(lambda row: (row[column_name] + added), axis=1)


def subtract_from_column_in_data_frame(data_frame, column_name, subtracted):
    add_to_column_in_data_frame(data_frame, column_name, -subtracted)


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
            add_to_column_in_data_frame(new, 'start', rolling_offset)
            add_to_column_in_data_frame(new, 'end', rolling_offset)
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


def write_selected_columns_to_file(data_frame, columns, file_path, with_header=False, with_index=False):
    data_frame_with_selected_columns = data_frame[columns]
    data_frame_with_selected_columns.to_csv(file_path, header=with_header, index=with_index)


def main():
    subject_id = '004'
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

    subtract_from_column_in_data_frame(events, 'start', offset)
    subtract_from_column_in_data_frame(events, 'end', offset)

    # remove the last heel drops
    heel_drop_events = events.loc[events['type'] == 'heel drop'][0:]
    last_heeldrop = heel_drop_events.iloc[0]['start']
    events = events.drop(events[events.end > last_heeldrop].index)

    print("First heel drops end at", offset, "seconds")
    print("Length of annotated data after removing heel drops:", last_heeldrop, "seconds")

    print("Reading sensor data ...")

    a = time()
    synchronized_csv_suffix = '_thigh.resampled.csv'
    try:
        sensor_readings = pd.read_csv(folder_and_subject_id + synchronized_csv_suffix, parse_dates=[0],
                                      header=None)
    except IOError:
        print("Synchronized sensor data not found. Creating synchronized data.")
        create_synchronized_back_and_thigh_file_for_subject(subject_folder, subject_id)
        print("Conversion finished. Reading converted data.")
        sensor_readings = pd.read_csv(folder_and_subject_id + synchronized_csv_suffix, parse_dates=[0],
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

    # EIRIK: These lines are at
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
    labeled_sensor_readings = labeled_sensor_readings[
        ['Time', 'Master-X', 'Master-Y', 'Master-Z', 'Slave-X', 'Slave-Y', 'Slave-Z', 'label']]

    print("Writing results to CSV")
    a = time()
    labeled_sensor_readings.to_csv(folder_and_subject_id + '.raw-labeled.csv')

    back_file_path = folder_and_subject_id + "_Axivity_BACK_Back.csv"
    thigh_file_path = folder_and_subject_id + "_Axivity_THIGH_RIGHT.csv"
    label_file_path = folder_and_subject_id + "_GoPro_LAB_All.csv"

    back_columns = ["Master-X", "Master-Y", "Master-Z"]
    thigh_columns = ["Slave-X", "Slave-Y", "Slave-Z"]
    label_columns = ["label"]

    write_selected_columns_to_file(labeled_sensor_readings, back_columns, back_file_path)
    write_selected_columns_to_file(labeled_sensor_readings, thigh_columns, thigh_file_path)
    write_selected_columns_to_file(labeled_sensor_readings, label_columns, label_file_path)

    b = time()

    print("Wrote 'labeled_sensor_readings' to csv in", b - a)


if __name__ == "__main__":
    main()
