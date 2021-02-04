from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
import organize_data as od
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import argmax

from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
import autoencoder as ae
# calculate inception score with Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import conf

feature_size = 4
time_series_size = conf.time_series_size
BUFFER_SIZE = 60000
BATCH_SIZE = 64

n_epochs = 20
latent_dim = 100
p_dimension = 1

train_mode = False

decoder = ae.load_decoder()

def generate_fake_data(time_series_size, data_cnt):
    model_file = 'models/' + str(time_series_size) + '_' + str(p_dimension) + '_dcgan.h5'
    model = tf.keras.models.load_model(model_file)

    p = np.random.randint(3, size=(data_cnt, 5)) + np.ones((data_cnt, 5))
    test_labels = np.array(od.convert_personality_to_class_label(p, p_dimension))



    z_input = tf.random.normal([data_cnt, latent_dim])

    predictions_all = model.predict([z_input, test_labels])

    xy_pupil = predictions_all[0][:, :, :, 0]
    xy = predictions_all[0][:, :, 0:2, 0]

    blinks = decoder(predictions_all[1]).numpy().reshape(data_cnt, time_series_size, 1)

    if feature_size > 3:
        x = np.concatenate((xy_pupil, blinks), axis=2)
    elif feature_size == 3:
        x = xy_pupil
    else:
        x = xy



    return x, test_labels


#
(x_fake, p_labels_fake) = generate_fake_data(time_series_size, 1000)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = p_labels_fake.reshape(len(p_labels_fake), 1)
p_labels_fake_enc = onehot_encoder.fit_transform(integer_encoded)




def organize_train_test(xt):

    # first two are x and y
    xy = xt[:, :, 0:2]
    xy = 2 * xy - 1
    
    # take the mean of left and right pupils
    pupils = (xt[:, :, 2] + xt[:, :, 3]) /2.0
    min_pupils = min(pupils.flatten())
    max_pupils = max(pupils.flatten())
    
    # normalize pupil dimensions bw [-1 1]
    pupils = 2 * (pupils - min_pupils) /(max_pupils - min_pupils) - 1
    pupils = pupils.reshape(pupils.shape[0], pupils.shape[1], 1)
    
    #add blinks as binary
    blinks = xt[:, :, 4:5]
    
    
    if feature_size > 3:
        x = np.concatenate((xy, pupils, blinks), axis = 2)
    elif feature_size == 3:
        x = np.concatenate((xy, pupils), axis = 2)
    else:
        x = xy


    plt.show()
    return x

(x, p_labels) = od.load_data(time_series_size)


xt = organize_train_test(x)
p_labels = od.convert_personality_to_class_label(p_labels, p_dimension)
# count different p_labels
#
# (unique, counts) = np.unique(p_labels, return_counts=True)
# print(len(unique))
# print(counts)

# One-hot encoding gives better results than integer encoding

integer_encoded = p_labels.reshape(len(p_labels), 1)
p_labels_enc = onehot_encoder.fit_transform(integer_encoded)


x_train, x_test, p_labels_train_enc, p_labels_test_enc = train_test_split(xt, p_labels_enc, test_size=0.01, shuffle=True, random_state=42)

# further divide the test data into cross validation and test sets
split_size = int(x_test.shape[0]/2)
x_cross_val = x_test[:split_size, :, :]
p_labels_cross_val_enc = p_labels_test_enc[:split_size, :]
x_test = x_test[split_size:, :, :]
p_labels_test_enc = p_labels_test_enc[split_size:, :]



# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score

if train_mode:

    model = Sequential()


    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(time_series_size, feature_size)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))

    if p_dimension is None:
        model.add(Dense(24, activation='softmax'))
    else:
        model.add(Dense(3, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.5)

    # chk = ModelCheckpoint('pers_class_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, p_labels_train_enc, epochs=n_epochs, batch_size=64,  validation_data=(x_cross_val, p_labels_cross_val_enc))
    # model.fit(x_train, p_labels_train, epochs=n_epochs, batch_size=64,  validation_data=(x_fake, p_labels_fake_enc))

    model.save('models/pers_class_model_' + str(p_dimension) + '.h5')
else:
    model = load_model('models/pers_class_model_' + str(p_dimension) + '.h5')
    # test_preds = model.predict_classes(x_fake)

    p_yx = model.predict(x_fake)
    # p_yx = model.predict(x_train)
    print(p_yx)
    # # conditional probabilities for personality classes
    score = calculate_inception_score(p_yx)
    print(score)

    # onehot_encoder2 = OneHotEncoder(categories=[range(0, 24)])
    #
    # test_preds = model.predict_classes(x_test)
    # integer_encoded = test_preds.reshape(len(test_preds), 1)
    # test_preds_enc = onehot_encoder2.fit_transform(integer_encoded, )
    #
    #
    # sc = accuracy_score(p_labels_test_enc, test_preds_enc)



    # print(sc)





