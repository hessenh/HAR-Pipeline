from __future__ import print_function

import numpy as np
from collections import Counter


def generate_all_integer_combinations(stop_integer):
    combinations_of_certain_length = dict()
    combinations_of_certain_length[1] = {(j,) for j in range(stop_integer)}

    for i in range(2, stop_integer + 1):
        combinations_of_certain_length[i] = set()
        for t in combinations_of_certain_length[i - 1]:
            largest_element = t[-1]
            for j in range(largest_element + 1, stop_integer):
                l = list(t)
                l.append(j)
                new_combination = tuple(l)
                combinations_of_certain_length[i].add(new_combination)

    values = [sorted(list(combinations_of_certain_length[j])) for j in range(2, stop_integer + 1)]

    reduced_list = []
    for value in values:
        reduced_list += value

    return reduced_list


def peak_acceleration(array):
    return max(np.linalg.norm(array, axis=1))


def max_and_mins(array):
    return np.hstack(array.max())


def means_and_std_factory(absolute_values=False):
    def means_and_std(a):
        if absolute_values:
            a = abs(a)
        return np.hstack((np.mean(a, axis=0), np.std(a, axis=0)))

    return means_and_std


def most_frequent_value(array):
    if len(array.shape) > 1:
        most_common = []
        for column in array.T:
            counts = Counter(column)
            top = counts.most_common(1)[0][0]
            most_common.append(top)

        return np.array(most_common)

    counts = Counter(array)
    top = counts.most_common(1)[0][0]
    return np.array([top])


def column_product_factory(columns):
    def columns_product(array):
        transposed = np.transpose(array)[[columns]]  # Transpose for simpler logic

        product = np.ones(transposed.shape[1])
        for row in transposed:
            product *= row

        product = np.transpose(product)

        return np.array([product.mean(), product.std()])

    return columns_product


class DataLoader:
    functions = {'means_and_std': [means_and_std_factory(False)],
                 'abs_means_and_std': [means_and_std_factory(True)],
                 'peak_acceleration': [peak_acceleration],
                 'most_common': [most_frequent_value]}

    def __init__(self, sample_frequency=100, window_length=2.0, degree_of_overlap=0.0):
        self.sample_frequency = sample_frequency
        self.window_size = int(round(window_length * sample_frequency))
        self.step_size = int(round((1 - degree_of_overlap) * self.window_size))

    def read_data(self, file_path, func_keywords, abs_vals=False, dtype="float", relabel_dict=None):
        sensor_data = np.loadtxt(fname=file_path, delimiter=",", dtype=dtype)

        if relabel_dict:
            for k in relabel_dict:
                np.place(sensor_data, sensor_data == k, [relabel_dict[k]])

        if abs_vals:
            sensor_data = abs(sensor_data)

        if not func_keywords:
            return sensor_data

        functions = []

        if len(sensor_data.shape) > 1:
            column_combinations = generate_all_integer_combinations(sensor_data.shape[1])

            self.functions["column_products"] = [column_product_factory(t) for t in column_combinations]

        for name in func_keywords:
            functions += self.functions[name]

        all_features = []

        for window_start in range(0, len(sensor_data), self.step_size):
            window_end = window_start + self.window_size
            if window_end > len(sensor_data):
                break
            window = sensor_data[window_start:window_end]

            extracted_features = [func(window) for func in functions]
            all_features.append(np.hstack(extracted_features))

        return np.vstack(all_features)

    def read_sensor_data(self, file_path, abs_vals=False):
        if abs_vals:
            keywords = ["means_and_std", "peak_acceleration", "column_products"]
        else:
            keywords = ["means_and_std", "abs_means_and_std", "peak_acceleration", "column_products"]
        return self.read_data(file_path, keywords, abs_vals=abs_vals)

    def read_label_data(self, file_path, relabel_dict):
        return self.read_data(file_path, ["most_common"], dtype="int", relabel_dict=relabel_dict).ravel()
