
from data import get_data_set
from lstm import LongShortTermMemory
import LSTM_PARAMETERS
import numpy as np


V = LSTM_PARAMETERS.VARS()


def load_data():
    data_type = "testing"
    generate_new_windows = V.GENERATE_NEW_WINDOWS
    oversampling = V.OVERSAMPLING
    viterbi = False
    path = V.TESTING_PATH

    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi, path)
    data_set.shuffle_data_set()
    data_set.data = np.reshape(data_set.data, (len(data_set.data), -1, 1))

    n_windows = data_set.data.shape[0]
    window_size = data_set.data.shape[1]
    time_sequence_size = 6
    n_time_sequences = int(window_size / time_sequence_size)

    data_set.data = data_set.data.reshape(n_windows, time_sequence_size, n_time_sequences)
    data_set.data = np.transpose(data_set.data, (0, 2, 1))

    data_set.data = data_set.data[data_set.labels[:, 0] >= 0]
    data_set.labels = data_set.labels[data_set.labels[:, 0] >= 0]

    return data_set


def load_network():
    nn = LongShortTermMemory(model_name=V.LSTM_LOAD_MODEL_NAME,
                             input_dim=V.LSTM_INPUT_DIM,
                             hidden_dim=V.LSTM_HIDDEN_DIM,
                             output_dim=V.LSTM_OUTPUT_DIM,
                             dropout=V.LSTM_DROPOUT,
                             activation=V.LSTM_ACTIVATION,
                             objective=V.LSTM_LOSS,
                             lr=V.LSTM_LEARNING_RATE,
                             nepoch=V.LSTM_NEPOCH,
                             batch_size=V.LSTM_BATCH_SIZE,
                             val_split=V.LSTM_VALIDATION_SPLIT,
                             model_dir=V.LSTM_MODEL_DIR,
                             result_dir=V.LSTM_RESULT_DIR)
    nn.load()
    nn.compile()

    return nn


def classify_data(nn, data_set):
    prediction = nn.classify(data_set.data)
    nn.performance(prediction, data_set.labels)


def main():
    data_set = load_data()
    nn = load_network()
    classify_data(nn, data_set)


if __name__ == "__main__":
    main()
