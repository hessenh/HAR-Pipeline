import glob

import os
import pandas as pd

from hunt_extraction import SUBJECT_DATA_LOCATION, SubjectConfiguration, combine_annotation_files, \
    remove_annotations_outside_heel_drops, create_labeled_data_frame
from raw_data_conversion.conversion import convert_subject_raw_file
from tools.pandas_helpers import write_selected_columns_to_file


def extract_wrist(subject_id):
    sensor_codeword = "LEFTWRIST"
    subject_folder = os.path.join(SUBJECT_DATA_LOCATION, subject_id)

    config_file = os.path.join(subject_folder, subject_id + "_config.json")
    sc = SubjectConfiguration(config_file, subject_id, subject_folder)
    sc.amplitude = 2  # Weaker amplitude for wrist

    original_sampling_frequency = 100
    sampling_frequency = 100  # in hertz

    annotations_glob = os.path.join(subject_folder, sc.id + "_FreeLiving_Event_*.txt")

    annotations_filenames = sorted(glob.glob(annotations_glob))
    annotations = combine_annotation_files(annotations_filenames)

    annotations = remove_annotations_outside_heel_drops(annotations, sc.start_peaks, sc.end_peaks)

    csv_files = glob.glob(os.path.join(subject_folder, "*_" + sensor_codeword + "_*" + sc.id + ".csv"))

    if csv_files:
        wrist_csv = csv_files[0]
    else:
        print("Wrist CSV not found. Converting CWA file to CSV")
        wrist_cwa = glob.glob(os.path.join(subject_folder, "*_" + sensor_codeword + "_*" + sc.id + ".cwa"))[0]
        wrist_csv = os.path.splitext(wrist_cwa)[0] + ".csv"

        convert_subject_raw_file(wrist_cwa, csv_outfile=wrist_csv)

    sensor_readings = pd.read_csv(wrist_csv, parse_dates=[0])

    if original_sampling_frequency == 200:
        sensor_readings = sensor_readings[::2].reindex()
        print("Original sampling frequency of 200 Hz reduced to 100")

    heel_drop_column = " Accel-Y (g)"

    labeled_sensor_readings = create_labeled_data_frame(sensor_readings, annotations, heel_drop_column, sc,
                                                        sampling_frequency)

    print("Writing results to CSVs")

    csv_output_folder = os.path.join(subject_folder, sensor_codeword)
    labeled_csv = os.path.join(csv_output_folder, sc.id + "_Axivity_" + sensor_codeword + "_Labeled.csv")
    labeled_columns = ["Accel-X (g)", " Accel-Y (g)", " Accel-Z (g)", "label"]

    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)

    write_selected_columns_to_file(labeled_sensor_readings, labeled_columns, labeled_csv)

    print("Wrote CSV file")