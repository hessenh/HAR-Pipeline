
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from time import time
import TRAINING_VARIABLES
import os


V = TRAINING_VARIABLES.VARS()


class LongShortTermMemory:

    def __init__(self, model_name, model_dir,
                 input_dim, hidden_dim, output_dim,
                 dropout, activation, objective,
                 lr, nepoch, batch_size, val_split):

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
        self.model_path = os.path.join(model_dir, self.model_name + ".h5")
        self.model = Sequential()


    def build_model(self):

        print("Building model ...")

        rseq = len(self.hidden_dim) > 1
        self.model = Sequential()

        self.model.add(LSTM(input_dim=self.input_dim,
                       output_dim=self.hidden_dim[0],
                       return_sequences=rseq))
        self.model.add(Dropout(self.dropout))

        for i in range(1, len(self.hidden_dim)):
            if i == len(self.hidden_dim) - 1:
                rseq = False
            self.model.add(LSTM(self.hidden_dim[i], return_sequences=rseq))
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(output_dim=self.output_dim))
        self.model.add(Activation(self.activation))


    def load_model(self):

        print("Loading model ...")

        self.model = load_model(self.model_path)
        print("Model loaded from " + self.model_path)


    def compile_model(self):

        print("\nCompiling model ...")

        start = time()
        self.model.compile(loss=self.objective, optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])
        t = time() - start
        print("Compilation time : %fs" % t)

        return t


    def train_model(self, X, y):

        print("Initiating training ...")

        start = time()
        self.model.fit(X, y,
                       batch_size=self.batch_size,
                       nb_epoch=self.nepoch,
                       validation_split=self.val_split)
        t = time() - start
        print("Training time : %fs" % t)

        self.model.save(self.model_path)
        print("Model saved in " + self.model_path)

        return t


    def test_model(self, X, y):

        print("Testing ...")

        start = time()
        score = self.model.evaluate(X, y, verbose=1)
        t = time() - start

        print("Loss: %.3f" % score[0])
        print("Accuracy: %.2f %%" % (score[1]*100))

        return score, t
