from data import get_data_set
from cnn import ConvolutionalNeuralNetwork
from viterbi import generate_transition_matrix
import numpy as np
import TRAINING_VARIABLES

V = TRAINING_VARIABLES.VARS()


def main():
    # Load training data
    # Input: Data_type, generate new windows, oversampling, viterbi training
    data_type = "training"
    generate_new_windows = True
    oversampling = True
    viterbi = False
    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi)
    data_set.shuffle_data_set()

    # Create network
    cnn = ConvolutionalNeuralNetwork()
    cnn.set_data_set(data_set)
    cnn.train_network()
    cnn.save_model()

    # Viterbi

    # Unshuffled data set
    # Input: Testing, generate new windows, oversampling, viterbi training
    data_type = "training"
    generate_new_windows = True
    oversampling = False
    viterbi = True
    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi)
    cnn.load_model()
    # Data set and number of samples
    actual, predictions = cnn.get_viterbi_data(data_set, V.VITERBI_LENGTH_OF_TRANSITION_DATA)

    np.savetxt(V.VITERBI_PREDICTION_PATH_TRAINING, predictions, delimiter=",")
    np.savetxt(V.VITERBI_ACTUAL_PATH_TRAINING, actual, delimiter=",")

    generate_transition_matrix("combination")

    print "Training finished"


if __name__ == "__main__":
    main()
