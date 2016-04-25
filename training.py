from data import get_data_set
from cnn import Convolutional_Neural_Network
from viterbi import generate_transition_matrix
import numpy as np
import TRAINING_VARIABLES



V = TRAINING_VARIABLES.VARS()

def main():


  ''' Load training data '''
  # Input: Testing, generate new windows, oversampling, viterbi training
  #data_set = get_data_set(False, True, True, False)
  #data_set.shuffle_data_set()

  ''' Create network '''
  #cnn = Convolutional_Neural_Network()
  #cnn.set_data_set(data_set)
  #cnn.train_network()
  #cnn.save_model()



  # Viterbi

  # Unshuffled data set
  # Input: Testing, generate new windows, oversampling, viterbi training
  #data_set = get_data_set(False, True, False, True)
  #cnn.load_model()
  # Data set and number of samples
  #actual, predictions = cnn.get_viterbi_data(data_set, 19676) #16473

  #np.savetxt(V.VITERBI_PREDICTION_PATH_TRAINING, predictions, delimiter=",")
  #np.savetxt(V.VITERBI_ACTUAL_PATH_TRAINING, actual, delimiter=",")

  generate_transition_matrix("BW")



	



		
if __name__ == "__main__":
    main()