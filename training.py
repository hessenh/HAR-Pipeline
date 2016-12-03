from data import get_data_set
from cnn import ConvolutionalNeuralNetwork
from viterbi import generate_transition_matrix
import numpy as np
import TRAINING_VARIABLES

V = TRAINING_VARIABLES.VARS()


def train(subject_list=None, normalize_sensor_data=False):
    # Load training data
    # Input: Data_type, generate new windows, oversampling, viterbi training
    data_type = "training"
    generate_new_windows = True
    oversampling = True
    viterbi = False
    print("Getting dataset for training")
    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi, V.TRAINING_PATH, subject_list=subject_list, normalize_sensor_data=normalize_sensor_data)
    data_set.shuffle_data_set()

    # Create network
    cnn = ConvolutionalNeuralNetwork()
    cnn.set_data_set(data_set)
    print("Training network")
    cnn.train_network()
    cnn.save_model()

    # Viterbi

    # Unshuffled data set
    # Input: Testing, generate new windows, oversampling, viterbi training
    data_type = "training"
    generate_new_windows = True
    oversampling = False
    viterbi = True
    print("Getting dataset for viterbi classification")
    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi, V.TRAINING_PATH, subject_list=subject_list, normalize_sensor_data=normalize_sensor_data)
    cnn.load_model()
    # Data set and number of samples
    print("Getting predictions for viterbi")
    actual, predictions = cnn.get_viterbi_data(data_set, V.VITERBI_LENGTH_OF_TRANSITION_DATA)

    np.savetxt(V.VITERBI_PREDICTION_PATH_TRAINING, predictions, delimiter=",")
    np.savetxt(V.VITERBI_ACTUAL_PATH_TRAINING, actual, delimiter=",")

    generate_transition_matrix("combination")
    cnn.close_session()
    del cnn

    print "Training finished"


if __name__ == "__main__":
    train()
