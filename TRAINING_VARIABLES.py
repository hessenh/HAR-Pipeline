class VARS(object):
    # Used to format a list when creating cnn model name
    def format_list(list_input):
        string_list = ""
        for s in list_input:
            string_list += str(s) + '_'
        return string_list

    ''' Variables '''
    # Convertion of activities
    CONVERTION = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 8, 13: 9, 14: 10}
    REVERSE_CONVERSION = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 13, 10: 14}
    REMOVE_ACTIVITIES = [0, 3, 9, 11, 16, 12, 15, 17, 18, 19]  # This has been extended with 18 and 19, Eirik
    ACTIVITY_NAMES_CONVERTION = {1: 'WALKING',
                                 2: 'RUNNING',
                                 3: 'STAIRS (UP)',
                                 4: 'STAIRS (DOWN)',
                                 5: 'STANDING',
                                 6: 'SITTING',
                                 7: 'LYING',
                                 8: 'BENDING',
                                 9: 'CYCLING (SITTING)',
                                 10: 'CYCLING (STANDING)'}
    ACTIVITIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    NUMBER_OF_ACTIVITIES = 10

    # Amount of overlap used for training, testing and prediction phase. None equals zero overlap.
    # 20 equals 80% overlap. 40 = 60% overlap. Yeah, not intuitive, but...
    TESTING_OVERLAP = None
    TRAINING_OVERLAP = 20
    PREDICTING_OVERLAP = None
    # Length of data window. With sampling frequency as 100Hz, we chose 100, meaning 1.0 second windows.
    WINDOW_LENGTH = 100

    # Paths for training, testing and predicting data
    TRAINING_PATH = "DATA/TRAINING"
    TESTING_PATH = 'DATA/TESTING'
    PREDICTING_PATH = 'DATA/PREDICTING'

    # Name of sensors and label.
    # This is how the system differentiate between the different files located in each subject folder.
    # The back sensor file must have a word (e.g. "BACK") separated by two underscores ("_").
    # This word can be changed by altering the variables below (e.g. "CHEST").
    SENSOR_1 = 'BACK'
    SENSOR_2 = 'THIGH'
    SENSORS = [SENSOR_1, SENSOR_2]
    LABEL = 'LAB'
    WINDOW_NAME_SENSOR = 'SENSORS'
    WINDOW_NAME_LABEL = 'LABEL'

    ''' CNN SPECIFIC'''
    # Input size is window-length multiplied by the number of sensor axes.
    CNN_INPUT_SIZE = 600
    CNN_OUTPUT_SIZE = 10
    # Number of training iterations. In our report, we have used 20000.
    CNN_NUMBER_OF_ITERATIONS = 20000
    # Number of instanses used between changing training parameters
    CNN_BATCH_SIZE = 100
    # Length of kernel in horizontal orientation
    CNN_FILTER_X = 30
    # Length of kernel in vertical orientation
    CNN_FILTER_Y = 1
    # Used to resize the input
    CNN_RESIZE_Y = 6
    CNN_RESIZE_X = 100
    CNN_WINDOW = CNN_INPUT_SIZE / 6

    # Numer of kernels (features) in each layer. Do not change the first value (1).
    CNN_KERNEL_LIST = [1, 20, 40]
    CNN_NUMBER_OF_KERNELS = len(CNN_KERNEL_LIST) - 1
    # Not so easy to grasp, but..
    CNN_CONNECTIONS_INN = CNN_RESIZE_Y * (
        CNN_RESIZE_X - (CNN_NUMBER_OF_KERNELS * CNN_FILTER_X) + CNN_NUMBER_OF_KERNELS) * CNN_KERNEL_LIST[-1]
    CNN_NEURAL_LIST = [CNN_CONNECTIONS_INN] + [1500] + [CNN_OUTPUT_SIZE]
    CNN_PADDING = 'VALID'
    # This name is saved in all tensors throughout the model, meaning that this must be the same when creating and loading a model.
    CNN_MODEL_NAME = str(CNN_INPUT_SIZE) + '_' + "conv_" + format_list(CNN_KERNEL_LIST[1:]) + "neural_" + format_list(
        CNN_NEURAL_LIST[1:-1]) + CNN_PADDING + '_' + str(CNN_NUMBER_OF_ITERATIONS)
    CNN_MODEL_PATH = 'MODELS/' + CNN_MODEL_NAME

    ''' VITERBI '''
    # This is the amount of data windows used to create the transition matrix used in the Viterbi.
    # Our recommended length is the number of windows for the first training subject.
    VITERBI_LENGTH_OF_TRANSITION_DATA = 19676

    # Number of iterations for the Baum Welch algorithm
    VITERBI_BAUM_WELCH_ITERATIONS = 5
    # Paths used for saving viterbi and cnn predictions
    VITERBI_PREDICTION_PATH_TRAINING = 'VITERBI/PREDICTION_TRAINING.csv'
    VITERBI_ACTUAL_PATH_TRAINING = 'VITERBI/ACTUAL_TRAINING.csv'
    VITERBI_PREDICTION_PATH_TESTING = 'VITERBI/PREDICTION_TESTING.csv'
    PREDICTION_RESULT_TESTING = 'VITERBI/PREDICTION_RESULT_TESTING.csv'
    VITERBI_RESULT_TESTING = 'RESULTS/RESULT_TESTING.csv'
    RESULT_TESTING_JSON = 'RESULTS/RESULT_TESTING_JSON.json'

    VITERBI_PREDICTION_PATH_PREDICTING = 'VITERBI/PREDICTION_PREDICTING.csv'
    VITERBI_RESULT_PREDICTING = 'RESULTS/RESULT_PREDICTING.csv'

    # The different states the Viterbi algorithm uses
    VITERBI_STATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # The viterbi algorithm saves and loads the transition matrix. This matrix is saves as a dictionary.
    VITERBI_TRANSITION_DICTIONARY_PATH = 'VITERBI/TRANSITION_DICTIONARY'

    # Convert the labels of the activities. Used for visualization
    VISUALIZATION_CONVERTION = {1: 4, 2: 10, 3: 5, 4: 6, 5: 3, 6: 2, 7: 1, 8: 9, 9: 7, 10: 8}
