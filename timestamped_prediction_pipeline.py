from __future__ import print_function

import shutil
import os
import pandas as pd
import TRAINING_VARIABLES

from time import time
from raw_data_conversion.conversion import create_synchronized_file_for_subject, \
    set_header_names_for_data_generated_by_omconvert_script
from tools.pandas_helpers import write_selected_columns_to_file, average_columns_and_write_to_file
from predicting import predict
from scipy import signal

V = TRAINING_VARIABLES.VARS()


def predict_with_timestamps_for_subject(subject_id, all_subjects_folder):
    time_start = time()

    # Names of files which are to be written
    subject_folder = os.path.join(all_subjects_folder, subject_id)

    master_sensor = os.path.join(subject_folder, subject_id + "_L.cwa")
    slave_sensor = os.path.join(subject_folder, subject_id + "_R.cwa")
    synced_csv = os.path.join(subject_folder, subject_id + "_synced.csv")

    predictions_csv = os.path.join(subject_folder, subject_id + "_raw_predictions.csv")

    master_csv = os.path.join(subject_folder, subject_id + "_Axivity_THIGH_Right.csv")
    slave_csv = os.path.join(subject_folder, subject_id + "_Axivity_BACK_Back.csv")
    time_csv = os.path.join(subject_folder, subject_id + "_timestamps.csv")
    sensor_averages_csv = os.path.join(subject_folder, subject_id + "_sensor_averages.csv")

    files_that_should_be_removed_after_prediction = [master_csv, slave_csv, time_csv, synced_csv, predictions_csv,
                                                     sensor_averages_csv]

    # The only file which is to remain after prediction
    timestamped_predictions_csv = os.path.join(subject_folder, subject_id + "_timestamped_predictions.csv")

    if (not os.path.isfile(master_csv)) or (not os.path.isfile(slave_csv)) or (not os.path.isfile(time_csv)) or (
            not os.path.isfile(sensor_averages_csv)):
        if not os.path.exists(synced_csv):
            create_synchronized_file_for_subject(master_sensor, slave_sensor, synced_csv, clean_up=True,
                                                 with_dirty_fix=True)

        print("Reading synced CSV")
        synced = pd.read_csv(synced_csv, parse_dates=[0], header=None)
        set_header_names_for_data_generated_by_omconvert_script(synced)

        print("Writing sensor readings and time stamps to files")
        master_columns = ["Master-X", "Master-Y", "Master-Z"]
        slave_columns = ["Slave-X", "Slave-Y", "Slave-Z"]
        time_columns = ["Time"]

        sensor_keep_rate = 1  # Keep every sample
        time_stamp_keep_rate = 100  # Keep only each 100th time stamp, because only 1 second windows will be labeled

        csv_paths = [master_csv, slave_csv, time_csv]
        columns = [master_columns, slave_columns, time_columns]
        keep_rates = [sensor_keep_rate, sensor_keep_rate, time_stamp_keep_rate]

        for f, cols, k in zip(csv_paths, columns, keep_rates):
            if not os.path.isfile(f):
                write_selected_columns_to_file(synced, cols, f, keep_rate=k, keep_only_complete_windows=True)

        print("Filtering the sensor columns, suppressing high frequencies")
        b, a = signal.butter(4, 0.3)

        def _filter_series(s):
            return signal.filtfilt(b, a, s)

        for col in master_columns + slave_columns:
            print("\t", col)
            synced[col] = _filter_series(synced[col])

        print("Calculating sensor reading averages. This will take a couple of minutes.")
        average_columns_and_write_to_file(synced, master_columns + slave_columns, sensor_averages_csv, window_size=100,
                                          average_only_complete_windows=True, float_format="%.8f")

    print("Getting predictions")
    predict(subject_list=[subject_id], subjects_folder=all_subjects_folder, output_file=predictions_csv,
            with_viterbi=False)

    time_stop = time()

    print("Adding timestamps to predictions, saving to", timestamped_predictions_csv)
    timestamps = pd.read_csv(time_csv, parse_dates=[0], header=None)
    predictions = pd.read_csv(predictions_csv, header=None, converters={0: int}) + 1  # Add 1 to get real activity nos
    predictions = predictions.applymap(reverse_conversion)
    sensor_averages = pd.read_csv(sensor_averages_csv, header=None)
    timestamped_predictions = pd.concat([timestamps, sensor_averages, predictions], axis=1)
    timestamped_predictions.to_csv(timestamped_predictions_csv, header=False, index=False)

    print("Removing auxilary files")
    for f in files_that_should_be_removed_after_prediction:
        os.remove(f)

    shutil.rmtree(os.path.join(subject_folder, "WINDOW"))

    print("Used", time_stop - time_start, "seconds in total")


def get_all_predictable_subjects(root_folder=os.path.join(".", "DATA", "PREDICTING")):
    walk_from_root = os.walk(root_folder)
    i = 0
    subjects = []
    while True:
        try:
            folder, subfolders, files = walk_from_root.next()
            all_subjects_folder, subject_id = os.path.split(folder)
            if os.path.exists(os.path.join(folder, subject_id + "_L.cwa")) and os.path.exists(
                    os.path.join(folder, subject_id + "_R.cwa")):
                subjects.append((all_subjects_folder, subject_id))
                i += 1
        except StopIteration:
            print(i)
            break

    return subjects


def reverse_conversion(x):
    return V.REVERSE_CONVERSION[x]


if __name__ == "__main__":
    folders_and_subjects = get_all_predictable_subjects()
    for folder, subject in folders_and_subjects:
        predict_with_timestamps_for_subject(subject_id=subject, all_subjects_folder=folder)
