from data import get_data_set
from cnn import Convolutional_Neural_Network
from viterbi import generate_transition_matrix
import numpy as np
import TRAINING_VARIABLES



V = TRAINING_VARIABLES.VARS()

def main():


  ''' Load training data '''
  # Input: Testing, generate new windows, oversampling
  #data_set = get_data_set(False, False, True)
  #data_set.shuffle_data_set()

  ''' Create network '''
  cnn = Convolutional_Neural_Network()
  #cnn.set_data_set(data_set)
  #cnn.train_network()
  #cnn.save_model()



  # Viterbi

  # Unshuffled data set
  #data_set = get_data_set(False, False, False)
  #cnn.load_model()
  # Data set and number of samples
  #actual, predictions = cnn.get_viterbi_data(data_set, 16473) #16473

  #np.savetxt(V.VITERBI_PREDICTION_PATH_TRAINING, predictions, delimiter=",")
  #np.savetxt(V.VITERBI_ACTUAL_PATH_TRAINING, actual, delimiter=",")

  generate_transition_matrix("BW")



	



		
if __name__ == "__main__":
    main()