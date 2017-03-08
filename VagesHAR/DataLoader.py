from __future__ import print_function

import os
import numpy as np
from collections import Counter


def generate_all_combinations(k):
    combinations = dict()
    for i in range(1, k + 1):
        if i == 1:
            combinations[i] = set([(j,) for j in range(k)])
        else:
            combinations[i] = set()
            for t in combinations[i - 1]:
                largest_element = t[-1]
                for j in range(largest_element + 1, k):
                    l = list(t)
                    l.append(j)
                    new_combination = tuple(l)
                    combinations[i].add(new_combination)

    vs = [sorted(list(combinations[j])) for j in range(2, k + 1)]
    return reduce(lambda x, y: x + y, vs)


def peak_acceleration(a):
    return max(np.linalg.norm(a, axis=1))


def max_and_mins(a):
    return np.hstack(a.max())


def means_and_std_factory(absolute_values=False):
    def means_and_std(a):
        if absolute_values:
            a = abs(a)
        return np.hstack((np.mean(a, axis=0), np.std(a, axis=0)))

    return means_and_std


def most_frequent_value(a):
    if len(a.shape) > 1:
        most_common = []
        for column in a.T:
            counts = Counter(column)
            top = counts.most_common(1)[0][0]
            most_common.append(top)

        return np.array(most_common)

    counts = Counter(a)
    top = counts.most_common(1)[0][0]
    return np.array([top])


def column_product_factory(columns):
    def columns_product(a):
        transposed = np.transpose(a)[[columns]]  # Transpose for simpler logic
        product = np.transpose(reduce(lambda x, y: x * y, transposed))
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

        fs = []

        if len(sensor_data.shape) > 1:
            column_combinations = generate_all_combinations(sensor_data.shape[1])

            self.functions["column_products"] = [column_product_factory(t) for t in column_combinations]

        for a in func_keywords:
            fs += self.functions[a]

        all_features = []

        for window_start in range(0, len(sensor_data), self.step_size):
            window_end = window_start + self.window_size
            if window_end > len(sensor_data):
                break
            window = sensor_data[window_start:window_end]

            extracted_features = [f(window) for f in fs]
            all_features.append(np.hstack(extracted_features))

        return np.vstack(all_features)

    def read_sensor_data(self, file_path, abs_vals=False):
        if abs_vals:
            kws = ["means_and_std", "peak_acceleration", "column_products"]
        else:
            kws = ["means_and_std", "abs_means_and_std", "peak_acceleration", "column_products"]
        return self.read_data(file_path, kws, abs_vals=abs_vals)

    def read_label_data(self, file_path, relabel_dict):
        return self.read_data(file_path, ["most_common"], dtype="int", relabel_dict=relabel_dict).ravel()


if __name__ == "__main__":
    thigh_sensor = os.path.join("..", "DATA", "inlab_dataset", "01A", "01A_Axivity_THIGH_Right.csv")
    dl = DataLoader(100, 2, 0)
    acceleration_data = dl.read_sensor_data(thigh_sensor)

    print(acceleration_data.shape)

    label_file = os.path.join("..", "DATA", "inlab_dataset", "01A", "01A_GoPro_LAB_All.csv")
    label_data = dl.read_label_data(label_file)

    print(label_data)
    print(label_data.shape)
