from data import get_data_set
from cnn import Convolutional_Neural_Network
from viterbi import run_viterbi
import numpy as np
import TRAINING_VARIABLES

V = TRAINING_VARIABLES.VARS()


def main():
	

	''' Load test data '''
	# Input: Testing, generate new windows, oversampling
	TESTING = True
	GENERATE_NEW_WINDOWS = False
	OVERSAMPLING = False
	data_set = get_data_set(TESTING, GENERATE_NEW_WINDOWS, OVERSAMPLING)

	''' Create network '''
	cnn = Convolutional_Neural_Network()
	cnn.set_data_set(data_set)
 	cnn.load_model()
 	
 	''''''
	actual = data_set._labels
	cnn_result = cnn.get_predictions()
	viterbi_result = run_viterbi()


	''' Add results in array with actual label'''
	result = np.zeros((len(cnn_result), 3))
	for i in range(0,len(cnn_result)):
		a = np.argmax(actual[i])
		c = np.argmax(cnn_result[i])
		v = viterbi_result[i]-1
		result[i] = [a,c,v]

	cnn_score = 0.0
	viterbi_score = 0.0
	for i in range(0, len(result)):
		if result[i][0] == result[i][1]:
			cnn_score +=1.0
		if result[i][0] == result[i][2]:
			viterbi_score += 1.0
	print cnn_score/len(result), viterbi_score/len(result)

		
if __name__ == "__main__":
    main()