# coding: utf-8
from __future__ import division, print_function

import glob

import pandas as pd
import numpy as np
from time import time


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indices of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


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
    sensor_readings = pd.read_csv(folder_and_subject_id + '_thigh.resampled.csv', parse_dates=[0],
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
