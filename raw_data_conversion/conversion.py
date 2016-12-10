OMCONVERT_LOCATION = "./raw_data_conversion/omconvert/omconvert"
TIMESYNC_LOCATION = "./raw_data_conversion/timesync/timesync"


def convert_subject_raw_file(input_file, csv_outfile=None, wav_outfile=None):
    import subprocess

    omconvert = OMCONVERT_LOCATION

    command = [omconvert, input_file]

    if csv_outfile is not None:
        command += ['-csv-file', csv_outfile]

    if wav_outfile is not None:
        command += ['-out', wav_outfile]

    subprocess.call(command)


def create_synchronized_file_for_subject(master_cwa, slave_cwa, output_csv, clean_up=True):
    from os.path import splitext
    import subprocess

    timesync = TIMESYNC_LOCATION

    master_wav = splitext(master_cwa)[0] + ".wav"
    slave_wav = splitext(slave_cwa)[0] + ".wav"

    convert_subject_raw_file(master_cwa, wav_outfile=master_wav)
    convert_subject_raw_file(slave_cwa, wav_outfile=slave_wav)

    # Synchronize them and make them a CSV
    subprocess.call([timesync, master_wav, slave_wav, "-csv", output_csv])

    if clean_up:
        print("Deleting wav files")
        subprocess.call(["rm", master_wav])
        subprocess.call(["rm", slave_wav])


def set_header_names_for_data_generated_by_omconvert_script(data_frame):
    data_frame.rename(
        columns={0: 'Time', 1: 'Master-X', 2: 'Master-Y', 3: 'Master-Z', 4: 'Slave-X', 5: 'Slave-Y', 6: 'Slave-Z'},
        inplace=True)