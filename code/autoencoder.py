
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dense
from keras.layers import Conv1D, UpSampling1D
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling1D

from keras.layers import Dropout
from keras.layers import Flatten

import organize_data as od
import conf
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np



feature_size = 1 #od.feature_size #x, y, pupil mean, blinks
time_series_size = conf.time_series_size

encoding_dim = int(time_series_size / 4.0) #32  # this is the size of our encoded representations
autoencoder_model_file = 'models/' + str(time_series_size) + '_decoder_blinks_dcgan.h5'

BUFFER_SIZE = 60000
BATCH_SIZE = 64


(x, p_labels) = od.load_data_blinks(time_series_size)

train_split = (int)(x.shape[0] * 0.75)

blinks = x[0:train_split, :, 4:5]  # only blinks as binary variables
blinks_test = x[train_split:, :, 4:5]  # only blinks as binary variables

train_blink = tf.data.Dataset.from_tensor_slices((blinks, blinks)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_blink = tf.data.Dataset.from_tensor_slices((blinks_test, blinks_test)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def train_encoder_decoder():

    inp = Input(shape=(time_series_size, feature_size))

    inp2 = Flatten()(inp)

    # add a Dense layer with a L1 activity regularizer
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='tanh')(inp2)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(time_series_size, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs= inp, outputs = decoded)

    opt = Adam(0.001)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    autoencoder.fit(train_blink,
                             epochs=20,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             validation_data=test_blink)

    autoencoder.save(autoencoder_model_file)

def train_deep_encoder_decoder():
    # (600, 1)
    inp = Input(shape=(time_series_size, feature_size))

    # (600, 64)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(time_series_size, feature_size))(inp)
    # (300, 64)
    x = MaxPooling1D(pool_size=2)(x)
    # (300, 32)
    x = Conv1D(filters=32, kernel_size=3, activation='relu',padding='same')(x)
    # (150, 32)
    x = MaxPooling1D(pool_size=2)(x)
    # (150, 1)
    encoded = Conv1D(filters=1, kernel_size=3, activation='relu', padding='same')(x)

    # (150)
    x = Flatten()(encoded)
    # (150, 1)
    x = Reshape((encoding_dim, 1))(x)
    # (150, 32)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    # (300, 32)
    x = UpSampling1D(2)(x)
    # (300, 64)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    # (600, 32)
    x = UpSampling1D(2)(x)
    # (600, 1)
    decoded = Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs=inp, outputs=decoded)

    opt = Adam(0.001)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    autoencoder.fit(train_blink,
                    epochs=3,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=test_blink)

    autoencoder.save(autoencoder_model_file)


def load_encoder():
    autoencoder = tf.keras.models.load_model(autoencoder_model_file)
    inp = Input(shape=(time_series_size, feature_size))
    inp2 = Flatten()(inp)
    encoder_layer = autoencoder.layers[-2]
    encoder = Model(inp, encoder_layer(inp2))
    return encoder

def load_decoder():

    autoencoder = tf.keras.models.load_model(autoencoder_model_file)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    return decoder



def load_encoder_deep():
    autoencoder = tf.keras.models.load_model(autoencoder_model_file)
    inp = Input(shape=(time_series_size, feature_size))
    encoder_layer = autoencoder.layers[6]

    encoder = Model(inp, encoder_layer(inp))
    return encoder


def load_decoder_deep():
    autoencoder = tf.keras.models.load_model(autoencoder_model_file)
    # create a placeholder for an encoded (32-dimensional) input
    inp = Input(shape=(encoding_dim,))

    # encoded_input = Reshape((encoding_dim, 1))(inp)
    # encoded_input = RepeatVector(1)(inp)
    encoded_input = inp
    # # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-7]

    out = decoder_layer(encoded_input)
    # # create the decoder model
    decoder = Model(inp, out)

    return decoder




def test_encoder_decoder(enc_func, dec_func):
    encoder = enc_func()
    decoder = dec_func()

    encoded_imgs = encoder.predict(blinks_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    print(blinks_test.shape)
    print(encoded_imgs.shape)
    print(decoded_imgs.shape)

    for i in range(4000,4010):
        plt.plot(blinks_test[i, :, 0], 'r')
        plt.plot(decoded_imgs[i, :], 'b')
        plt.show()


# train_encoder_decoder()
# test_encoder_decoder(load_encoder, load_decoder)
# train_deep_encoder_decoder()
# test_encoder_decoder(load_encoder_deep, load_decoder_deep)




