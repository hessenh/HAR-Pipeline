
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from time import time
import os
import numpy as np
import json
import LSTM_PARAMETERS


V = LSTM_PARAMETERS.VARS()


class LongShortTermMemory:

    def __init__(self, model_name, model_dir, result_dir,
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

        self.model = Sequential()
        self.model_name = model_name
        self.model_dir = model_dir
        self.result_dir = result_dir


    def build(self):

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


    def save(self):

        desc = {
            "MODEL NAME": self.model_name,
            "INPUT DIMENSION": self.input_dim,
            "HIDDEN DIMENSION": self.hidden_dim,
            "OUTPUT DIMENSION": self.output_dim,
            "DROPOUT": self.dropout,
            "ACTIVATION": self.activation,
            "OBJECTIVE": self.objective,
            "LEARNING RATE": self.lr,
            "EPOCHS": self.nepoch,
            "BATCH SIZE": self.batch_size,
            "VALIDATION SPLIT": self.val_split
        }

        path = os.path.join(self.model_dir, self.model_name) + ".h5"
        self.model.save(path)
        print("Model saved in " + path)

        path = os.path.join(self.model_dir, self.model_name) + ".json"
        with open(path, 'w') as f:
            json.dump(desc, f)
        print("Model description saved in " + path)


    def load(self):

        print("Loading model ...")

        path = os.path.join(self.model_dir, self.model_name) + ".h5"
        self.model = load_model(path)
        print("Model loaded from " + path)


    def compile(self):

        print("Compiling model ...")

        start = time()
        self.model.compile(loss=self.objective, optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])
        t = time() - start
        print("Compilation time : %fs" % t)

        return t


    def train(self, x, y):

        print("Initiating training ...")

        start = time()
        self.model.fit(x, y,
                       batch_size=self.batch_size,
                       nb_epoch=self.nepoch,
                       validation_split=self.val_split)
        t = time() - start
        print("Training time : %fs" % t)

        return t


    def classify(self, x):

        def one_hot(mat):
            most_probable = np.argmax(mat, axis=1)
            one_hot_vecs = np.zeros(mat.shape)
            one_hot_vecs[np.arange(most_probable.size), most_probable] = 1
            return one_hot_vecs


        print("Classifying data ...")

        start = time()

        predicted = self.model.predict(x)
        predicted = one_hot(predicted)

        t = time() - start
        print("Classification time : %fs" % t)

        return predicted


    def performance(self, predicted, y):

        actual_pos = np.sum(y, axis=0)
        actual_neg = np.asarray([y.shape[0]] * y.shape[1]) - actual_pos

        true_pos = np.sum(np.logical_and(predicted, y), axis=0)
        true_neg = np.sum(np.equal(predicted, y), axis=0) - true_pos
        false_pos = np.sum(np.greater(predicted, y), axis=0)
        false_neg = np.sum(np.less(predicted, y), axis=0)

        accuracy = np.sum(true_pos.astype(float)) / np.sum(actual_pos.astype(float))
        accuracy = float(np.mean(accuracy[~np.isnan(accuracy)]))

        recall = true_pos.astype(float) / actual_pos.astype(float)
        mean_recall = float(np.mean(recall[~np.isnan(recall)]))

        precision = true_pos.astype(float) / (true_pos + false_pos).astype(float)
        mean_precision = float(np.mean(precision[~np.isnan(precision)]))

        specificity = true_neg.astype(float) / actual_neg.astype(float)
        mean_specificity = float(np.mean(specificity[~np.isnan(specificity)]))

        results = {
            "ACCURACY": accuracy,
            "PRECISION": {},
            "RECALL": {},
            "SPECIFICITY": {}
        }

        for i in range(len(recall)):
            results["PRECISION"][V.ACTIVITY_NAMES_CONVERTION[i]] = precision[i]
            results["RECALL"][V.ACTIVITY_NAMES_CONVERTION[i]] = recall[i]
            results["SPECIFICITY"][V.ACTIVITY_NAMES_CONVERTION[i]] = specificity[i]

        print("Accuracy: %.2f %%" % (accuracy * 100))
        print("Precision: %.2f %%" % (mean_precision * 100))
        print("Recall: %.2f %%" % (mean_recall * 100))
        print("Specificity: %.2f %%" % (mean_specificity * 100))

        path = os.path.join(self.result_dir, "RESULT_TESTING") + ".json"
        with open(path, 'w') as f:
            json.dump(results, f)
        print("Performance saved in " + path)

        return accuracy, precision, recall, specificity


    def test(self, x, y):

        print("Testing ...")

        start = time()
        score = self.model.evaluate(x, y, verbose=1)
        t = time() - start

        print("Accuracy: %.2f %%" % (score[1]*100))

        return score, t
