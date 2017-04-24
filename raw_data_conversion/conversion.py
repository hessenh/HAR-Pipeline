from __future__ import print_function

import sys

import os
import pandas as pd

from definitions import OMCONVERT_LOCATION, TIMESYNC_LOCATION, PROJECT_ROOT


def convert_subject_raw_file(input_file, csv_outfile=None, wav_outfile=None):
    import subprocess

    omconvert = OMCONVERT_LOCATION

    command = [omconvert, input_file]

    if csv_outfile is not None:
        command += ['-csv-file', csv_outfile]

    if wav_outfile is not None:
        command += ['-out', wav_outfile]

    subprocess.call(command)


def synchronize_sensors(cwas, output_csv, clean_up=True, sync_fix=True, nrows=None):
    import subprocess

    timesync = TIMESYNC_LOCATION

    output_folder = os.path.split(output_csv)[0]

    master_cwa = cwas[0]

    slave_cwas = cwas[1:]

    master_wav = os.path.splitext(master_cwa)[0] + ".wav"
    convert_subject_raw_file(master_cwa, wav_outfile=master_wav)
    master_start_time = subprocess.check_output(["grep", "-a", 'Time', master_wav]).strip()[-23:]

    wav_files = [master_wav]
    csv_files = []

    for i, s in enumerate(slave_cwas):
        s_prefix = os.path.splitext(s)[0]
        slave_wav = s_prefix + ".wav"
        wav_files.append(slave_wav)
        convert_subject_raw_file(s, wav_outfile=slave_wav)

        if sync_fix:
            slave_start_time = subprocess.check_output(["grep", "-a", 'Time', slave_wav]).strip()[-23:]
            subprocess.call(["sed", "-i", "s/" + slave_start_time + "/" + master_start_time + "/g", slave_wav])

        # Synchronize them and make them a CSV
        tmp_output_path = os.path.join(output_folder, s_prefix + ".csv")

        csv_files.append(tmp_output_path)
        subprocess.call([timesync, master_wav, slave_wav, "-csv", tmp_output_path])

    first_csv_file = csv_files[0]
    first_dataframe = pd.read_csv(first_csv_file, parse_dates=[0], header=None, nrows=nrows)

    if len(csv_files) == 1:
        os.rename(first_csv_file, output_csv)
        csv_files.remove(first_csv_file)
        output_dataframe = first_dataframe
    else:
        data_frames = [first_dataframe]
        for next_csv in csv_files[1:]:
            data_frames.append(pd.read_csv(next_csv, header=None, usecols=[4, 5, 6], nrows=nrows))

        output_dataframe = pd.concat(data_frames, axis=1, ignore_index=True)
        output_dataframe.to_csv(output_csv, header=False, index=False)

    if clean_up:
        print("Cleaning up files")
        for f in csv_files + wav_files:
            subprocess.call(["rm", f])

    return output_dataframe
