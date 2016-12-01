import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
from collections import Counter
from numpy.lib.stride_tricks import as_strided as ast

import TRAINING_VARIABLES

V = TRAINING_VARIABLES.VARS()


class DataSet(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(data)

    def shuffle_data_set(self):
        perm = np.arange(len(self.data))
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.labels[start:end]


def main():
    get_data_set("testing", False, False, False)


def get_data_set(data_type, generate_new_windows, oversampling, viterbi, subjects_path):
    if generate_new_windows:
        generate_windows_for_subjects_in_folder(data_type, viterbi, subjects_path)

    df_sensor, df_label = load_windows(data_type, oversampling, subjects_path)
    data_set = DataSet(df_sensor, df_label)

    return data_set


def load_windows(data_type, oversampling, subjects_path):
    all_sensor_dataframes = None
    all_label_dataframes = None

    individual_sensor_dataframes = []
    individual_label_dataframes = []

    subject_list = get_folder_names(subjects_path)

    if len(subject_list) == 0:
        print "No subjects found!"

    for subject_id in subject_list:
        print subject_id
        subject_window_path = subjects_path + '/' + subject_id + '/WINDOW/'

        df_sensor_temp = load_dataframe(subject_window_path + 'SENSORS.csv').as_matrix()
        individual_sensor_dataframes.append(df_sensor_temp)

        if data_type != "predicting":
            df_label_temp = pd.read_csv(subject_window_path + 'LABEL.csv', header=None, sep=',').as_matrix()
            individual_label_dataframes.append(df_label_temp)

    # Concatenate all individual dataframes
    if individual_sensor_dataframes:
        all_sensor_dataframes = np.vstack(individual_sensor_dataframes)

    if individual_label_dataframes:
        all_label_dataframes = np.vstack(individual_label_dataframes)

    if data_type == "training" and oversampling:
        print 'OVERSAMPLING'
        data_sensor = all_sensor_dataframes
        data_label = all_label_dataframes

        # length of longest activity
        max_length = 0
        activities = V.ACTIVITIES
        for activity in activities:
            activity_length = sum(data_label[::, activity])
            if activity_length > max_length:
                max_length = activity_length

        data_sensor_new = np.zeros([max_length * len(activities), 600])
        data_label_new = np.zeros([max_length * len(activities), len(activities)])

        for i in range(0, len(activities)):
            activity_boolean = data_label[::, i] == 1.0
            activity_data = data_sensor[activity_boolean]
            activity_label = data_label[activity_boolean]

            activity_length = len(activity_data)
            fraction = int(max_length / activity_length) + 1

            new_activity_data = np.tile(activity_data, (fraction, 1))
            new_activity_label = np.tile(activity_label, (fraction, 1))
            new_activity_data = new_activity_data[0:max_length]
            new_activity_label = new_activity_label[0:max_length]
            data_sensor_new[i * max_length:i * max_length + max_length] = new_activity_data
            data_label_new[i * max_length:i * max_length + max_length] = new_activity_label

        all_sensor_dataframes = data_sensor_new
        all_label_dataframes = data_label_new

    return all_sensor_dataframes, all_label_dataframes


def generate_windows_for_subjects_in_folder(data_type, viterbi, subjects_path):
    all_subjects_in_folder = get_folder_names(subjects_path)

    for subject_name in all_subjects_in_folder:
        print subject_name
        subject_path = subjects_path + '/' + subject_name

        generate_windows_for_one_subject(subject_path, data_type, viterbi)


def generate_windows_for_one_subject(subject_path, data_type, viterbi):
    subject_files_dictionary = get_subject_files_from_path(subject_path)

    df_sensor_1 = load_dataframe(subject_path + '/' + subject_files_dictionary[V.SENSOR_1])
    df_sensor_2 = load_dataframe(subject_path + '/' + subject_files_dictionary[V.SENSOR_2])

    if data_type != "predicting":
        df_label = load_dataframe(subject_path + '/' + subject_files_dictionary[V.LABEL])

    # Remove activities
    if data_type == "training" and not viterbi:
        df_sensor_1, df_sensor_2, df_label = remove_activities(df_sensor_1, df_sensor_2, df_label,
                                                               V.REMOVE_ACTIVITIES)
    result_path = subject_path + '/WINDOW/'
    # Create windows
    if data_type == "testing" or viterbi:
        overlap = V.TESTING_OVERLAP
        create_window_sensors(df_sensor_1, df_sensor_2, result_path, V.WINDOW_LENGTH, overlap)
        create_window_label(df_label, result_path, V.WINDOW_LENGTH, overlap)
    elif data_type == "training":
        overlap = V.TRAINING_OVERLAP
        create_window_sensors(df_sensor_1, df_sensor_2, result_path, V.WINDOW_LENGTH, overlap)
        create_window_label(df_label, result_path, V.WINDOW_LENGTH, overlap)
    elif data_type == "predicting":
        overlap = V.PREDICTING_OVERLAP
        create_window_sensors(df_sensor_1, df_sensor_2, result_path, V.WINDOW_LENGTH, overlap)


def find_most_common_label(l):
    word_counts = Counter(l)
    most_common_label = word_counts.most_common(1)
    return most_common_label[0][0]


def convert_label(l):
    n = np.zeros(V.NUMBER_OF_ACTIVITIES)
    if l in V.CONVERTION:
        activity = V.CONVERTION[l]
        n[activity - 1] = 1.0
    else:
        activity = 1
        n[activity - 1] = -100
    return n


def create_window_label(df_label, folder, length, overlap):
    df_window = split_data_frame(df_label[0], length, overlap)
    window = df_window.as_matrix()

    new_window = np.zeros([len(df_window), V.NUMBER_OF_ACTIVITIES])
    for i in range(0, len(window)):
        new_window[i] = convert_label(find_most_common_label(window[i]))

    # df_window = df_window.apply(find_most_common_label,axis=1)
    # df_window = df_window.apply(convert_label)
    df_window = pd.DataFrame(new_window)

    save_data_frame(df_window, folder, V.WINDOW_NAME_LABEL)


def create_window_sensors(df_sensor_1, df_sensor_2, folder, length, overlap):
    # Sensor 1
    df_s_1_x = split_data_frame(df_sensor_1[0], length, overlap)
    df_s_1_y = split_data_frame(df_sensor_1[1], length, overlap)
    df_s_1_z = split_data_frame(df_sensor_1[2], length, overlap)

    # Sensor 2
    df_s_2_x = split_data_frame(df_sensor_2[0], length, overlap)
    df_s_2_y = split_data_frame(df_sensor_2[1], length, overlap)
    df_s_2_z = split_data_frame(df_sensor_2[2], length, overlap)

    df = pd.concat([df_s_1_x, df_s_1_y, df_s_1_z, df_s_2_x, df_s_2_y, df_s_2_z], axis=1)

    save_data_frame(df, folder, V.WINDOW_NAME_SENSOR)


def save_data_frame(df, folder, file_name):
    # Check if folder exists, if not, create one
    if not exists(folder):
        makedirs(folder)
    print folder + file_name
    df.to_csv(folder + file_name + '.csv', header=None, index=False)


def split_data_frame(df, length, overlap):
    windows = sliding_window(df, length, overlap)
    return pd.DataFrame(windows)


def remove_activities(df_back, df_thigh, df_label, remove_activity_list):
    for activity in remove_activity_list:
        keep_boolean = df_label[0] != activity
        df_back = df_back[keep_boolean]
        df_thigh = df_thigh[keep_boolean]
        df_label = df_label[keep_boolean]

    return df_back, df_thigh, df_label


def load_dataframe(path):
    return pd.read_csv(path, header=None)


def get_folder_names(path):
    return listdir(path)


def get_subject_files_from_path(subject_path):
    subject_files = [f for f in listdir(subject_path) if isfile(join(subject_path, f))]

    # Using a dictionary to connect sensor type with filename
    # 'LAB': '01A_GoPro_LAB_All.csv'
    subject_files_dictionary = {}
    for subject_file in subject_files:
        file_split = subject_file.split("_")
        # Using the third word as the key
        subject_files_dictionary[file_split[2]] = subject_file

    return subject_files_dictionary


''' Sliding window methods bellow are from http://www.johnvinyard.com/blog/?p=268'''


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return i,
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    """

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(
            'ws cannot be larger than a in any dimension.a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    new_shape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the stridden array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    new_shape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    new_strides = norm_shape(np.array(a.strides) * ss) + a.strides
    stridden = ast(a, shape=new_shape, strides=new_strides)
    if not flatten:
        return stridden

    # Collapse stridden so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    first_dim = (np.product(new_shape[:-meat]),) if ws.shape else ()
    dim = first_dim + (new_shape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i: i != 1, dim)
    return stridden.reshape(dim)


if __name__ == "__main__":
    main()
