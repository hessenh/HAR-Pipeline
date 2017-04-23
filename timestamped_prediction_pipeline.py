from __future__ import print_function

import shutil
import os
import pandas as pd
import TRAINING_VARIABLES

from time import time
from raw_data_conversion.conversion import synchronize_sensors
from tools.pandas_helpers import write_selected_columns_to_file, average_columns_and_write_to_file
from predicting import predict
from scipy import signal

V = TRAINING_VARIABLES.VARS()

slave_ending = "_R.cwa"
master_ending = "_L.cwa"


def predict_with_timestamps_for_subject(subject_id, residing_folder, master_cwa=None, slave_cwa=None,
                                        output_csv=None, remove_auxiliary_files=True):
    """
    Creates a CSV with timestamped predictions for a subject given its ID and residing folder.

    :param subject_id: This subject's ID (the name of the sub-folder)
    :param residing_folder: The folder where the subject resides
    :param master_cwa: (optional) location of master CWA
    :param slave_cwa: (optional) location of slave CWA
    :param output_csv: (optional) location to write output
    :param remove_auxiliary_files: (optional) whether or not to remove auxiliary files
    :return:
    """
    time_start = time()

    # Names of files which are to be written
    subject_folder = os.path.join(residing_folder, subject_id)

    if master_cwa is None:
        master_cwa = os.path.join(subject_folder, subject_id + master_ending)
    if slave_cwa is None:
        slave_cwa = os.path.join(subject_folder, subject_id + slave_ending)

    synced_csv = os.path.join(subject_folder, subject_id + "_synced.csv")

    predictions_csv = os.path.join(subject_folder, subject_id + "_raw_predictions.csv")

    master_csv = os.path.join(subject_folder, subject_id + "_Axivity_THIGH_Right.csv")
    slave_csv = os.path.join(subject_folder, subject_id + "_Axivity_BACK_Back.csv")
    time_csv = os.path.join(subject_folder, subject_id + "_timestamps.csv")
    sensor_averages_csv = os.path.join(subject_folder, subject_id + "_sensor_averages.csv")

    files_that_should_be_removed_after_prediction = [master_csv, slave_csv, time_csv, synced_csv, predictions_csv,
                                                     sensor_averages_csv]

    # The only file which is to remain after prediction
    if output_csv is None:
        timestamped_predictions_csv = os.path.join(subject_folder, subject_id + "_timestamped_predictions.csv")

    if (not os.path.isfile(master_csv)) or (not os.path.isfile(slave_csv)) or (not os.path.isfile(time_csv)) or (
            not os.path.isfile(sensor_averages_csv)):
        print("Reading synced CSV")
        synced = synchronize_sensors([master_cwa, slave_cwa], synced_csv, clean_up=remove_auxiliary_files,
                                     sync_fix=True)

        print("Writing sensor readings and time stamps to files")
        master_columns = [1, 2, 3]
        slave_columns = [4, 5, 6]
        time_columns = [0]

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
    predict(subject_list=[subject_id], subjects_folder=residing_folder, output_file=predictions_csv,
            with_viterbi=False)

    time_stop = time()

    print("Adding timestamps to predictions and saving to", timestamped_predictions_csv)
    timestamps = pd.read_csv(time_csv, parse_dates=[0], header=None)
    predictions = pd.read_csv(predictions_csv, header=None, converters={0: int}) + 1  # Add 1 to get real activity nos
    predictions = predictions.applymap(reverse_conversion)
    sensor_averages = pd.read_csv(sensor_averages_csv, header=None)
    timestamped_predictions = pd.concat([timestamps, sensor_averages, predictions], axis=1)
    timestamped_predictions.to_csv(timestamped_predictions_csv, header=False, index=False)

    if remove_auxiliary_files:
        print("Removing auxiliary files")
        for f in files_that_should_be_removed_after_prediction:
            os.remove(f)

        shutil.rmtree(os.path.join(subject_folder, "WINDOW"))

    print("Used", time_stop - time_start, "seconds in total")


def get_all_predictable_subjects(root_folder=os.path.join(".", "DATA", "PREDICTING")):
    """
    Finds all subjects with the two required CWA files in the given folder and its sub-folders.

    :param root_folder: The folder at which to start the search.
    :return: A list of tuples [(folder, subject_id), ...]
    """
    subjects = []
    for f, _, _ in os.walk(root_folder):
        all_subjects_folder, subject_id = os.path.split(f)
        if os.path.exists(os.path.join(f, subject_id + master_ending)) and os.path.exists(
                os.path.join(f, subject_id + slave_ending)):
            subjects.append((all_subjects_folder, subject_id))
        else:
            print("Could not find two .cwa-files in", f)

    return subjects


def reverse_conversion(x):
    """
    Converts a number output by the CNN-Viterbi-pipeline back to the way it was annotated

    :param x: Output number
    :return: Label-style number
    """
    return V.REVERSE_CONVERSION[x]


if __name__ == "__main__":
    master_ending = "_femur.cwa"
    slave_ending = "_vertebra.cwa"
    folders_and_subjects = sorted(get_all_predictable_subjects())
    for folder, subject in folders_and_subjects:
        predict_with_timestamps_for_subject(subject_id=subject, residing_folder=folder)
