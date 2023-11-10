from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, BatchNormalization, Activation


def model_init(n_steps_in, n_steps_out, n_features):
    model = Sequential()

    model.add(LSTM(300, input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(300, return_sequences=True))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Dense(n_steps_out)))

    return model
