
from time import strftime


class VARS(object):

    # Data config
    GENERATE_NEW_WINDOWS = True
    OVERSAMPLING = False
    TRAINING_PATH = "DATA/TRAINING"
    TESTING_PATH = "DATA/TESTING"
    ACTIVITY_NAMES_CONVERTION = {
        0: 'WALKING',
        1: 'RUNNING',
        2: 'STAIRS (UP)',
        3: 'STAIRS (DOWN)',
        4: 'STANDING',
        5: 'SITTING',
        6: 'LYING',
        7: 'BENDING',
        8: 'CYCLING (SITTING)',
        9: 'CYCLING (STANDING)'
    }

    # Path config
    LSTM_BUILD_MODEL_NAME = strftime("%Y%m%d_%H%M_") + "in_lab"
    LSTM_LOAD_MODEL_NAME = "20161209_1648_in_lab"
    LSTM_MODEL_DIR = "MODELS"
    LSTM_RESULT_DIR = "RESULTS"

    # Model config
    LSTM_INPUT_DIM = 6
    LSTM_HIDDEN_DIM = [30]
    LSTM_OUTPUT_DIM = 10
    LSTM_DROPOUT = 0.2
    LSTM_ACTIVATION = "sigmoid"

    # Training config
    LSTM_LOSS = "categorical_crossentropy"
    LSTM_LEARNING_RATE = 0.01
    LSTM_NEPOCH = 5
    LSTM_BATCH_SIZE = 30
    LSTM_VALIDATION_SPLIT = 0.05
