
from data import get_data_set
from lstm import LongShortTermMemory
import TRAINING_VARIABLES
import numpy as np
from time import strftime


V = TRAINING_VARIABLES.VARS()


def load_data():
    data_type = "training"
    generate_new_windows = True
    oversampling = True
    viterbi = False
    path = V.TRAINING_PATH

    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi, path)
    data_set.shuffle_data_set()
    data_set.data = np.reshape(data_set.data, (len(data_set.data), -1, 1))

    return data_set


def create_network(data_set):
    time_date = strftime("%Y%m%d_%H%M_")
    model_name = time_date + V.LSTM_BUILD_MODEL_NAME
    nn = LongShortTermMemory(model_name=model_name,
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
                             model_dir=V.LSTM_MODEL_PATH)
    nn.build_model()
    nn.compile_model()
    nn.train_model(data_set.data, data_set.labels)

    return nn


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
                             model_dir=V.LSTM_MODEL_PATH)
    nn.load_model()
    nn.compile_model()

    return nn


def main():
    data_set = load_data()
    # nn = create_network(data_set)
    nn = load_network()
    score = nn.test_model(data_set.data, data_set.labels)


if __name__ == "__main__":
    main()
