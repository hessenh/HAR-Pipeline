
from data import get_data_set
from lstm import LongShortTermMemory

import numpy as np


def load_data():
    data_type = "training"
    generate_new_windows = True
    oversampling = True
    viterbi = False

    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi, V.TRAINING_PATH)
    data_set.shuffle_data_set()
    data_set.data = np.reshape(data_set.data, (len(data_set.data), -1, 1))

    return data_set


def create_network(data_set):
    nn = LongShortTermMemory()
    nn.compile_model()
    nn.train_model(data_set.data, data_set.labels)

    return nn


def main():
    data_set = load_data()
    nn = create_network(data_set)


if __name__ == "__main__":
    main()
