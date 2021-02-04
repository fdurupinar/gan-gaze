from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Embedding, Flatten, Concatenate
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Dropout,  MaxPooling1D
from keras.layers import Conv1D, Conv1DTranspose
import organize_data as od
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import time
import keras.backend as K
import conf
import sys


tf.random.set_seed(5)

feature_size = 2 # only x and y positions
time_series_size = conf.time_series_size

# size of the latent space
# latent_dim = int(time_series_size / 4) #100
latent_dim = 100
# latent_padding = int(latent_dim/2)
latent_padding = 5

# p_dimension = None
p_dimension = conf.N
if p_dimension is None:
    n_classes = 243  # all personalities
else:
    n_classes = 3  # one personality dimension

model_file = 'models/' + str(time_series_size) + '_' + str(p_dimension) + '_cnngan.h5'

tf.compat.v1.enable_eager_execution()

(x, p_values) = od.load_data(time_series_size)

x_train = x[:, :, 0:2]
x_train = 2 * x_train - 1

p_values = p_values
p_labels = od.convert_personality_to_class_label(p_values, p_dimension)

# slice along the 1st dimension, i.e. each row is a slice
train_data = tf.data.Dataset.from_tensor_slices((x_train, p_labels)).shuffle(buffer_size=60000).batch(conf.batch_size)

def plot_real_data():
    x_vals = x[:,:,0]
    y_vals = x[:,:,1]
    plt.subplot(211)
    plt.plot(range(time_series_size*4), np.reshape(x_vals[0:4], (time_series_size*4)))
    plt.subplot(212)
    plt.plot(range(time_series_size*4), np.reshape(y_vals[0:4], (time_series_size*4)))
    plt.show()
    # quit()

# plot_real_data()

def write_real_data():
    p = [2, 2, 2, 2, 2]
    p_str = conf.personality_name[int(p_dimension)]
    f_ext = '_'.join(map(str, p)) + '_dim_' + p_str

    x_vals = x[:, :, 0]
    y_vals = x[:, :, 1]

    vx = np.reshape(x_vals[0:4], (time_series_size*4))
    vy = np.reshape(y_vals[0:4], (time_series_size*4))

    v = list(zip(vx, vy))

    with open('out/gaze_positions_' + f_ext + '.csv', 'w') as fp:
        np.savetxt(fp, v, delimiter=',')
#
#
# write_real_data()
# quit()

def define_discriminator():

    in_label = Input(shape=(1,))
    # embedding for categorical input
    x = Embedding(n_classes, 50)(in_label)
    # # scale up to image dimensions with linear activation
    #
    x = Dense(time_series_size)(x)
    #
    # # reshape to additional channel
    x = Reshape((time_series_size, 1))(x)

    # image input
    inp = Input(shape=(time_series_size, feature_size))

    # concat label as a channel
    merge = Concatenate()([inp, x])
    # todo
    # merge = inp
    # # downsample
    cnn1 = Conv1D(filters=128, kernel_size=conf.kernel_size, strides = 2,  padding='same')(merge)
    cnn1 = LeakyReLU()(cnn1)
    cnn1 = Dropout(0.3)(cnn1)
    # downsample again
    cnn1 = Conv1D(filters=128, kernel_size=conf.kernel_size, strides = 2, padding='same')(cnn1)
    # # cnn1 = BatchNormalization()(cnn1)
    cnn1 = LeakyReLU()(cnn1)
    cnn1 = Dropout(0.3)(cnn1)

    dense = Flatten()(cnn1)
    out_layer = Dense(1)(dense)

    model = Model([inp, in_label], out_layer)
    # model = Model(inp, out_layer)

    return model

# define the standalone generator model
def define_generator():

    dim = int(time_series_size / 4) - 2 * latent_padding
    n_nodes = dim * feature_size

    noise_start = Input(shape=(latent_padding,  feature_size*2))
    noise_end = Input(shape=(latent_padding,  feature_size*2))


    noise = Input(shape=(latent_dim))
    gen = Dense(n_nodes, use_bias=False)(noise)
    gen = Reshape((dim, feature_size))(gen)

    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    li = Dense(n_nodes, use_bias=False)(li)
    li = Reshape((dim, feature_size))(li)

    latent = Concatenate()([gen, li])

    merge = Concatenate(axis=1)([noise_start, latent, noise_end])

    # merge = noise
    # merge =  Concatenate()([merge1, li]) # added as another channel

    # convolve to  ts/4, 128
    gen = Conv1DTranspose(128, conf.kernel_size, strides=1, padding='same', use_bias=False)(merge)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)


    # # upsample (strides=2) to ts/2, 64
    gen = Conv1DTranspose(64, conf.kernel_size, strides=2, padding='same', use_bias=False)(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)

    # # upsample (strides=2) to ts, 2
    gen = Conv1DTranspose(feature_size, conf.kernel_size, strides=2, activation='tanh', padding='same', use_bias=False)(gen)


    model = Model([noise_start, noise_end, noise, in_label], gen)
    # model = Model([noise_start, noise_end], gen)
    # model = Model(noise, gen)

    model.summary()
    return model



# This method returns a helper function to compute cross entropy loss
cross_entropy = BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

discriminator = define_discriminator()
generator = define_generator()


generator_optimizer = Adam(0.0001, beta_1=0.5)
discriminator_optimizer = Adam(0.0001, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_cnn_")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# tf.function causes the function to be "compiled".
@tf.function
def train_step(x, p_labels):
    noise_start = tf.random.normal([conf.batch_size, latent_padding, feature_size*2])
    noise_end = tf.random.normal([conf.batch_size, latent_padding, feature_size*2])
    noise = tf.random.normal([conf.batch_size, latent_dim])

    generated_labels = randint(0, n_classes, conf.batch_size)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator([noise_start, noise_end, noise, generated_labels],  training=True)
        # generated_images = generator([noise_start, noise_end],  training=True)
        # generated_images = generator(noise,  training=True)


        # real_output = discriminator(x, training=True)
        real_output = discriminator([x, p_labels], training=True)
        fake_output = discriminator([generated_images, generated_labels], training=True)
        # fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return (gen_loss ,disc_loss)

# always use the same random values for prediction
seed_start = tf.random.normal([1, latent_padding, feature_size*2])
seed_end = tf.random.normal([1, latent_padding, feature_size*2])
seed = tf.random.normal([1, latent_dim])

def train(dataset, epochs):

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            x, p_labels = image_batch

            (gen_loss, disc_loss) = train_step(x, p_labels)

        K.print_tensor(gen_loss, message='gen loss =')
        K.print_tensor(disc_loss, message='disc loss =')

        # generate and save at each epoch
        plot_predictions(generator, seed_start, seed_end, seed)
        # plot_predictions(generator, seed)

        generator.save(model_file)

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def plot_predictions(model, padding_start, padding_end, noise):
# def plot_predictions(model, noise):

    test_labels1 = np.array([od.convert_personality_to_class_label([[1, 1, 1, 1, 1]], p_dimension)])
    test_labels2 = np.array([od.convert_personality_to_class_label([[2, 2, 2, 2, 2]], p_dimension)])
    test_labels3 = np.array([od.convert_personality_to_class_label([[3, 3, 3, 3, 3]],  p_dimension)])

    # predictions1 = model.predict(noise)
    # predictions2 = model.predict(noise)
    # predictions3 = model.predict(noise)
    predictions1 = model.predict([padding_start, padding_end, noise, test_labels1])
    predictions2 = model.predict([padding_start, padding_end, noise, test_labels2])
    predictions3 = model.predict([padding_start, padding_end, noise, test_labels3])

    # map them back to 0 1 region
    vx1 = [(i + 1) / 2.0 for i in predictions1[:, :, 0]][0]
    vx2 = [(i + 1) / 2.0 for i in predictions2[:, :, 0]][0]
    vx3 = [(i + 1) / 2.0 for i in predictions3[:, :, 0]][0]

    plt.subplot(211)
    plt.title("X")
    plt.plot(np.arange(0, time_series_size), vx1, c='b')
    plt.plot(np.arange(0, time_series_size), vx2, c='g')
    plt.plot(np.arange(0, time_series_size), vx3, c='r')



    vy1 = [(i + 1) / 2.0 for i in predictions1[:, :, 1]][0]
    vy2 = [(i + 1) / 2.0 for i in predictions2[:, :, 1]][0]
    vy3 = [(i + 1) / 2.0 for i in predictions3[:, :, 1]][0]
    plt.subplot(212)
    plt.title("Y")
    plt.plot(np.arange(0, time_series_size), vy1, c='b')  # 1
    plt.plot(np.arange(0, time_series_size), vy2, c='g')  # 2
    plt.plot(np.arange(0, time_series_size), vy3, c='r')  # 3
    plt.show()


def generate_and_save_data(model, p, iter_cnt, will_plot=False):


    test_labels = np.array([od.convert_personality_to_class_label([p], p_dimension)])

    noise_padding_start = None

    for it in range(iter_cnt):

        noise = tf.random.normal([1, latent_dim])
        noise_padding_end = tf.random.normal([1, latent_padding, feature_size*2])  # this will shift

        if noise_padding_start is None:
            noise_padding_start = tf.random.normal([1, latent_padding, feature_size*2])

        predictions = model.predict([noise_padding_start,  noise_padding_end, noise, test_labels])


        noise_padding_start = noise_padding_end  # shift padding for the next iteration

        if it == 0:
            predictions_all = predictions[0]
        else:
            clipped_start = latent_padding
            # each time clip the beginning part
            predictions_all = np.concatenate((predictions_all, predictions[0][clipped_start:,:]), axis=0)
            # predictions_all = np.concatenate((predictions_all, predictions[0]), axis=0)

    # map back to 0 1 region
    vx = [(i + 1) / 2.0 for i in predictions_all[:, 0]]
    vy = [(i + 1) / 2.0 for i in predictions_all[:, 1]]

    p_str = conf.personality_name[int(p_dimension)]
    f_ext = '_'.join(map(str, p)) + '_dim_' + p_str

    v = list(zip(vx, vy))
    with open('out/gaze_positions_' + f_ext + '.csv', 'w') as fp:
        np.savetxt(fp, v, delimiter=',')

    if will_plot:
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(211)
        plt.title('X '+ f_ext)
        plt.plot(np.arange(0, len(vx)), vx, c = 'magenta')
        plt.subplot(212)
        plt.title('Y ' + f_ext)
        plt.plot(np.arange(0, len(vy)), vy, c = 'cyan')

        for it in range(0, iter_cnt*2):
            plt.axvline(it*int(time_series_size/2), color='g', lw=0.3)

        plt.show()


def generate_and_save_data_multiple(model, iter_cnt=1):
    for n in range(1, 4):
        for e in range(1, 4):
            for o in range(1, 4):
                for a in range(1, 4):
                    for c in range(1, 4):
                        p = [n, e, o , a, c]
                        generate_and_save_data(model, p, iter_cnt, will_plot=False)


def generate_and_save_data_1d(model, iter_cnt=time_series_size):
    p = [2, 2, 2, 2, 2]

    p[int(p_dimension)] = 1
    generate_and_save_data(model, p, iter_cnt, will_plot=True)

    p[int(p_dimension)] = 2
    generate_and_save_data(model, p, iter_cnt, will_plot=True)

    p[int(p_dimension)] = 3
    generate_and_save_data(model, p, iter_cnt, will_plot=True)


train_mode = True
if len(sys.argv) > 1 and sys.argv[1] == "false":
    train_mode = False
# train_mode = False

if train_mode:
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    train(train_data, conf.n_epochs)

if not(train_mode):
    model = tf.keras.models.load_model(model_file)
    generate_and_save_data_1d(model,  conf.iter_cnt)


