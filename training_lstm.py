
from data import get_data_set
from lstm import build_model
import TRAINING_VARIABLES

import numpy as np


V = TRAINING_VARIABLES.VARS()


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
    nn = build_model()
    nn.set_data_set(data_set)
    nn.train_network()
    nn.save_model()

    return nn


def main():
    data_set = load_data()
    print(data_set.data[0][0])
    print(type(data_set.data[0][0]))
    print(len(data_set.data[0]))
    print(len(data_set.labels[0]))
    #nn = create_network(data_set, "cnn")


if __name__ == "__main__":
    main()
