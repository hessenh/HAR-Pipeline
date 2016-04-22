from data import get_data_set
from cnn import Convolutional_Neural_Network
from viterbi import run_viterbi
import numpy as np
import TRAINING_VARIABLES
import pandas as pd
import matplotlib.pyplot as plt



V = TRAINING_VARIABLES.VARS()


def main():
	

	''' Load test data '''
	# Input: Testing, generate new windows, oversampling, viterbi training
	TESTING = True
	GENERATE_NEW_WINDOWS = False
	OVERSAMPLING = False
	VITERBI = False
	#data_set = get_data_set(TESTING, GENERATE_NEW_WINDOWS, OVERSAMPLING, VITERBI)

	''' Create network '''
	#cnn = Convolutional_Neural_Network()
	#cnn.set_data_set(data_set)
 	#cnn.load_model()
 	
 	''''''
	#actual = data_set._labels
	#cnn_result = cnn.get_predictions()
	#cnn_result = pd.read_csv(V.VITERBI_PREDICTION_PATH_TESTING, header=None, sep='\,',engine='python').as_matrix()

	#viterbi_result = run_viterbi()
	#viterbi_result = pd.read_csv(V.VITERBI_RESULT_TESTING, header=None, sep='\,',engine='python').as_matrix()
	

	''' Add results in array with actual label'''
	#result = np.zeros((len(cnn_result), 3))
	#for i in range(0,len(cnn_result)):
	#	a = np.argmax(actual[i])
	#	c = np.argmax(cnn_result[i])
	#	v = viterbi_result[i]-1
	#	result[i] = [a,c,v]


	#np.savetxt(V.PREDICTION_RESULT_TESTING, result, delimiter=",")
	result = pd.read_csv(V.PREDICTION_RESULT_TESTING, header=None, sep='\,',engine='python').as_matrix()


	#print get_score(result)


	visualize(result)

def get_score(result_matrix):
	activities = V.ACTIVITIES
	'''TP / (FP - TP)
	Correctly classified walking / Classified as walking
	'''
	TP = np.zeros(len(activities))
	TN = np.zeros(len(activities))

	FP_TP = np.zeros(len(activities))
	TP_FN = np.zeros(len(activities))
	FP_TN = np.zeros(len(activities))
	
	actual = result_matrix[:,0]
	predicted = result_matrix[:,2]



	for activity in activities:
		''' FP - TP'''
		FP_TP[activity-1] = np.sum(predicted == activity) #len(df[df[0]==activity])
		''' TP - FN '''
		TP_FN[activity-1] = np.sum(actual == activity) #len(df_actual[df_actual[0]==activity])
		''' FP - TN '''
		FP_TN[activity-1] = np.sum(actual != activity)#len(df_actual[df_actual[0] != activity])

	for i in range(0, len(predicted)):
		if predicted[i] == actual[i]:
			TP[actual[i]-1] += 1.0
		
		for activity in activities:
			if actual[i] != activity and predicted[i]  != activity:
				TN[activity-1] += 1.0
				

	accuracy = sum(TP) / sum(TP_FN)
	specificity = TN / FP_TN
	precision = TP / FP_TP
	recall = TP / TP_FN
	return [accuracy, specificity, precision, recall]
		


def visualize(result_matrix):
	start = 0
	stop = 100
	actual = result_matrix[:,0]
	cnn = result_matrix[:,1]
	viterbi = result_matrix[:,2]


	t = cnn != viterbi
	actual = actual[t]
	cnn = cnn[t]
	viterbi = viterbi[t]



	plt.figure(1)

	plt.subplot(311)
	axes = plt.gca()
	axes.set_ylim([0.9,13.4])
	plt.plot(actual)

	plt.subplot(312)
	axes = plt.gca()
	axes.set_ylim([0.9,13.4])
	plt.plot(cnn)


	plt.subplot(313)
	axes = plt.gca()
	axes.set_ylim([0.9,13.4])
	plt.plot(viterbi)
	plt.show()





if __name__ == "__main__":
    main()