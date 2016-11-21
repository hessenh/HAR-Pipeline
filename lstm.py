
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from time import time
import TRAINING_VARIABLES


V = TRAINING_VARIABLES.VARS()


class LongShortTermMemory:

    def __init__(self,
                 model_name=V.LSTM_MODEL_NAME,
                 input_dim=V.LSTM_INPUT_DIM,
                 hidden_dim=V.LSTM.HIDDEN_DIM,
                 output_dim=V.LSTM_OUTPUT_DIM,
                 dropout = V.LSTM_DROPOUT,
                 activation=V.LSTM_ACTIVATION,
                 objective=V.LSTM_LOSS,
                 lr=V.LSTM_LEARNING_RATE,
                 nepoch=V.LSTM_NEPOCH,
                 batch_size=V.LSTM_BATCH_SIZE,
                 val_split=V.LSTM_VALIDATION_SPLIT):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.activation = activation
        self.objective = objective
        self.lr = lr

        self.nepoch = nepoch
        self.batch_size = batch_size
        self.val_split = val_split

        self.model_name = model_name
        self.model = self.build_model()


    def build_model(self):

        print('\nBuilding model ...')

        rseq = len(self.hidden_dim) > 1
        model = Sequential()

        model.add(LSTM(input_dim=self.input_dim,
                       output_dim=self.hidden_dim[0],
                       return_sequences=rseq))
        model.add(Dropout(self.dropout))

        for i in range(1, len(self.hidden_dim)):
            if i == len(self.hidden_dim) - 1:
                rseq = False
            model.add(LSTM(self.hidden_dim[i], return_sequences=rseq))
            model.add(Dropout(self.dropout))

        model.add(Dense(output_dim=self.output_dim))
        model.add(Activation(self.activation))

        return model


    def compile_model(self):

        print('\nCompiling model ...')

        start = time()
        self.model.compile(loss=self.objective, optimizer=RMSprop(lr=self.lr))
        t = time() - start
        print('Compilation time : %fs' % t)

        return t


    def train_model(self, X, y):

        print('\nInitiating training ...')

        start = time()
        self.model.fit(X, y,
                       batch_size=self.batch_size,
                       nb_epoch=self.nepoch,
                       validation_split=self.val_split)
        t = time() - start
        print('Training time : %fs' % t)

        return t
