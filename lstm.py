
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop

from time import time


def build_model(in_dim, hid_dim, out_dim, drop, act):

    print('\nBuilding model ...')

    rseq = len(hid_dim) > 1
    model = Sequential()

    model.add(LSTM(input_dim=in_dim,
                   output_dim=hid_dim[0],
                   return_sequences=rseq))
    model.add(Dropout(drop))

    for i in range(1, len(hid_dim)):
        if i == len(hid_dim) - 1: rseq = False
        model.add(LSTM(hid_dim[i], return_sequences=rseq))
        model.add(Dropout(drop))

    model.add(Dense(output_dim=out_dim))
    model.add(Activation(act))

    return model


def compile_model(model, obj, learn):

    print('\nCompiling model ...')

    start = time()
    model.compile(loss=obj, optimizer=RMSprop(lr=learn))
    t = time() - start
    print('Compilation time : %fs' % t)

    return t


def train_model(model, X, y, b_size, n_epochs, v_split):

    print('\nInitiating training ...')

    start = time()
    model.fit(X, y, batch_size=b_size, nb_epoch=n_epochs, validation_split=v_split)
    t = time() - start
    print('Training time : %fs' % t)

    return t
