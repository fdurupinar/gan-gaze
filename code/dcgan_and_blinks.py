from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Concatenate
import organize_data as od
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import time
from IPython import display
import keras.backend as K
import autoencoder as ae
import matplotlib
import conf


train_mode = True

feature_size = 4
time_series_size = conf.time_series_size

# size of the latent space
latent_dim = 100
half_latent_dim = 50

encoding_dim = ae.encoding_dim


# p_dimension = None
p_dimension = conf.N


tf.random.set_seed(2)
if p_dimension is None:
    n_classes = 243  # all personalities
else:
    n_classes = 3  # one personality dimension

model_file = 'models/' + str(time_series_size) + '_'+ str(p_dimension) + '_dcgan.h5'



tf.compat.v1.enable_eager_execution()

(x, p_values) = od.load_data(time_series_size)


train_split = int(x.shape[0] * 1.0)



# first two are x and y
xy = x[0:train_split, :, 0:2]
xy = 2 * xy - 1

# take the mean of left and right pupils
pupils = (x[0:train_split, :, 2] + x[0:train_split, :, 3]) / 2.0
min_pupils = min(pupils.flatten())
max_pupils = max(pupils.flatten())

# normalize pupil dimensions bw [-1 1]
pupils = 2 * (pupils - min_pupils) / (max_pupils - min_pupils) - 1
pupils = pupils.reshape(pupils.shape[0], pupils.shape[1], 1)

# add blinks as binary
blinks = x[0:train_split, :, 4:5]

if feature_size > 3:
    x_train = np.concatenate((xy, pupils, blinks), axis=2)
elif feature_size == 3:
    x_train = np.concatenate((xy, pupils), axis=2)
else:
    x_train = xy


p_values = p_values[0:train_split]

p_labels = od.convert_personality_to_class_label(p_values, p_dimension)

# slice along the 1st dimension, i.e. each row is a slice
train_data = tf.data.Dataset.from_tensor_slices((x_train, p_labels)).shuffle(buffer_size=60000).batch(conf.batch_size)


# define the standalone discriminator model
# TODO: n_classes will be 243 for all the personalities
def define_discriminator():

    in_label = Input(shape=(1,))
    # embedding for categorical input
    x = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation

    x = Dense(time_series_size)(x)

    # reshape to additional channel
    x = Reshape((time_series_size, 1))(x)


    # image input
    inp = Input(shape=(time_series_size, feature_size))

    # concat label as a channel
    merge = Concatenate()([inp, x])

    # # downsample

    cnn1 = Conv1D(filters=128, kernel_size=conf.kernel_size, strides = 2,  padding='same')(merge)

    cnn1 = LeakyReLU()(cnn1)
    cnn1 = Dropout(0.3)(cnn1)
    # downsample again
    cnn1 = Conv1D(filters=128, kernel_size=conf.kernel_size, strides = 2, padding='same')(cnn1)
    # cnn1 = BatchNormalization()(cnn1)
    cnn1 = LeakyReLU()(cnn1)
    cnn1 = Dropout(0.3)(cnn1)

    dense = Flatten()(cnn1)
    out_layer = Dense(1)(dense)

    model = Model([inp, in_label], out_layer)

    return model

# define the standalone generator model
# TODO: n_classes will be 243 for all the personalities
def define_generator():
    dim = int(time_series_size / 4)

    noise = Input(shape=latent_dim)

    # #low resolution sample
    n_nodes = dim * feature_size

    in_label = Input(shape=(1,))

    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    li = Dense(n_nodes, use_bias=False)(li)
    # Reshape to additional channel
    li = Reshape((dim, feature_size))(li)

    gen = Dense(n_nodes, use_bias=False)(noise)
    gen = Reshape((dim, feature_size))(gen)

    # merge  gen and label input
    merge = Concatenate()([gen, li])

    # convolve to  ts/4, 128
    gen = Conv1DTranspose(128, conf.kernel_size, strides=1, padding='same', use_bias=False)(merge)
    gen = BatchNormalization()(gen)
    # gen = LeakyReLU()(gen) # TODO: bu acikti
    gen = ReLU()(gen)

    # upsample (strides=2) to ts/2, 64
    gen = Conv1DTranspose(64, conf.kernel_size, strides=2, padding='same', use_bias=False)(gen)
    gen = BatchNormalization()(gen)
    # gen = LeakyReLU()(gen) # TODO: bu acikti
    gen = ReLU()(gen)


    # upsample (strides=2) to ts, feature_size
    gen = Conv1DTranspose(feature_size, conf.kernel_size, strides=2, activation='tanh', padding='same', use_bias=False)(gen)


    # generate blinks separately

    # size will be (None, 300, 3, 1)
    # out_layer_cont = gen[:, :, 0:feature_size-1, :]
    out_layer_cont = gen[:, :, 0:feature_size-1]

    # size will be (None, 300, 1)
    # gen2 = gen[:, :, 0:feature_size-1:feature_size, :]
    gen2 = gen[:, :, 0:feature_size-1:feature_size]
    gen2 = Flatten()(gen2)
    out_layer_blinks = Dense(encoding_dim)(gen2)

    model = Model([noise, in_label], [out_layer_cont, out_layer_blinks])

    return model



generator_optimizer = Adam(0.0001, beta_1=0.5)
discriminator_optimizer = Adam(0.0001, beta_1=0.5)

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


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_and_save_data(model, p, motion_length, will_plot=False, p_str=None):

    iter_cnt = int(motion_length/time_series_size)

    prev_half_noise = None

    decoded_predictions = np.array([])
    for it in range(iter_cnt):
        test_labels = np.array([od.convert_personality_to_class_label([p], p_dimension)])

        half_noise = tf.random.normal([1, half_latent_dim])
        if prev_half_noise is None:
            prev_half_noise = tf.random.normal([1, half_latent_dim])

        noise = tf.concat([prev_half_noise, half_noise], axis=1)
        # noise = tf.random.normal([1, latent_dim])

        predictions_all = model.predict([noise, test_labels])

        prev_half_noise = half_noise

        if it == 0:
            predictions0 = predictions_all[0][0]
        else:
            predictions0 = np.concatenate((predictions0, predictions_all[0][0]), axis=0)

        predictions1 = predictions_all[1]

        if feature_size > 3:
            decoded_predictions_partial = np.array(decoder(predictions1))[0]
            decoded_predictions = np.concatenate((decoded_predictions, decoded_predictions_partial), axis = 0)

    vx = [(i + 1) / 2.0 for i in predictions0[:, 0]]
    vy = [(i + 1) / 2.0 for i in predictions0[:, 1]]

    vx_s = smooth(vx, 5)
    vy_s = smooth(vy, 5)


    eps = 1e-3
    i = 0
    while abs(vx_s[i] - vx[i]) > eps:  #
        vx_s[i] = vx[i]
        i = i + 1

    i = -1
    while abs(vx_s[i] - vx[i]) > eps:  #
        vx_s[i] = vx[i]
        i = i - 1

    eps = 1e-3
    i = 0
    while abs(vy_s[i] - vy[i]) > eps:  #
        vy_s[i] = vy[i]
        i = i + 1

    i = -1
    while abs(vy_s[i] - vy[i]) > eps:  #
        vy_s[i] = vy[i]
        i = i - 1

    v = list(zip(vx_s, vy_s))

    if will_plot:
        plt.title("x-dashed, y normal")
        plt.plot(np.arange(0, motion_length), vx, linestyle='dashed')
        plt.plot(np.arange(0, motion_length), vy)

        plt.axvline(150)
        plt.axvline(300)
        plt.axvline(450)
        plt.show()

    if p_str:
        f_ext = '_'.join(map(str, p)) + '_dim_' + p_str
    else:
        f_ext = '_'.join(map(str, p))

    with open('out/gaze_positions_' + f_ext + '.csv', 'w') as fp:
        np.savetxt(fp, v, delimiter=',')

    if feature_size > 2:
        vp = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in predictions0[:, 2]]
        vp = list(zip(vp, vp))
        if will_plot:
            plt.title("Pupils")
            plt.plot(np.arange(0, motion_length), vp)
            plt.axvline(150)
            plt.axvline(300)
            plt.axvline(450)
            plt.show()

        with open('out/pupil_diameter_' + f_ext + '.csv', 'w') as fp:
            np.savetxt(fp, vp, delimiter=',')

    # TODO: closed blinks for now
    # if feature_size > 3:
    #     # decoded_predictions = np.array(decoder(predictions1))
    #     threshold = np.max(decoded_predictions) * 0.5
    #
    #     blinks = [str("Blink") if i > threshold else str("Saccade") for i in decoded_predictions]
    #
    #     if will_plot:
    #         plt.title("Blinks")
    #         plt.plot(1, threshold, 'ro')
    #         plt.plot(range(motion_length), decoded_predictions)
    #         plt.show()
    #     # with open('events.csv', 'w') as fp:
    #     with open('out/events_' + f_ext + '.csv', 'w') as fp:
    #         np.savetxt(fp, blinks, fmt="%s")


def generate_and_save_data_multiple(model, motion_length=time_series_size):
    #Low N
    # test_labels = np.array([2, 0, 0, 0,0])
    for n in range(1, 4):
        for e in range(1, 4):
            for o in range(1, 4):
                for a in range(1, 4):
                    for c in range(1, 4):
                        p = [n, e, o , a, c]
                        generate_and_save_data(model, p, motion_length, will_plot=False)
                        # test_labels = np.array([od.convert_personality_to_class_label([3, 1, 1, 1, 3])])


def generate_and_save_data_1d(model, motion_length=time_series_size):
    p = [2, 2, 2, 2, 2]

    p[int(p_dimension)] = 1
    generate_and_save_data(model, p, motion_length, will_plot=True, p_str=conf.personality_name[int(p_dimension)])

    # p[int(p_dimension)] = 2
    # generate_and_save_data(model, p, motion_length, will_plot=True, p_str=conf.personality_name[int(p_dimension)])
    #
    # p[int(p_dimension)] = 3
    # generate_and_save_data(model, p, motion_length, will_plot=True, p_str=conf.personality_name[int(p_dimension)])


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_predictions(model, test_input):

    test_labels1 = np.repeat(np.array([od.convert_personality_to_class_label([[1, 1, 1, 1, 1]], p_dimension)]), test_input.shape[0], axis=0)
    test_labels2 = np.repeat(np.array([od.convert_personality_to_class_label([[2, 2, 2, 2, 2]], p_dimension)]), test_input.shape[0], axis=0)
    test_labels3 = np.repeat(np.array([od.convert_personality_to_class_label([[3, 3, 3, 3, 3]],  p_dimension)]), test_input.shape[0], axis=0)

    predictions1 = model.predict([test_input, test_labels1])
    predictions2 = model.predict([test_input, test_labels2])
    predictions3 = model.predict([test_input, test_labels3])

    # map them back to 0 1 region
    vx1 = [(i + 1) / 2.0 for i in predictions1[0][:, :, 0]][0]
    vx2 = [(i + 1) / 2.0 for i in predictions2[0][:, :, 0]][0]
    vx3 = [(i + 1) / 2.0 for i in predictions3[0][:, :, 0]][0]


    plt.title("X")
    plt.plot(np.arange(0, time_series_size), vx1)
    plt.plot(np.arange(0, time_series_size), vx2)
    plt.plot(np.arange(0, time_series_size), vx3)
    plt.show()

    if feature_size > 1:
        vy1 = [(i + 1) / 2.0 for i in predictions1[0][:, :, 1]][0]
        vy2 = [(i + 1) / 2.0 for i in predictions2[0][:, :, 1]][0]
        vy3 = [(i + 1) / 2.0 for i in predictions3[0][:, :, 1]][0]

        plt.title("Y")
        plt.plot(np.arange(0, time_series_size), vy1, c='b')  # 1
        plt.plot(np.arange(0, time_series_size), vy2, c='g')  # 2
        plt.plot(np.arange(0, time_series_size), vy3, c='r')  # 3
        plt.show()

        if conf.plot_stats:
            matplotlib.rcParams.update({'axes.titlesize': 18})

            plt.rc('axes', labelsize=16)

            if p_dimension is None:
                plt.title("OCEAN")
            else:
                plt.title(conf.personality_name[p_dimension])

            plt.xlabel('x')
            plt.ylabel('y')

            plt.scatter(vx1, vy1, c='b',  label='low')
            plt.scatter(vx2, vy2, c='g',  label='med')
            plt.scatter(vx3, vy3, c='r',  label='high')
            plt.legend()
            plt.show()

    if feature_size > 2:
        p1 = predictions1[0][:, :, 2][0]
        p2 = predictions2[0][:, :, 2][0]
        p3 = predictions2[0][:, :, 2][0]

        if conf.plot_stats:
            d1 = predictions1[0][:, :, 2].flatten()
            d2 = predictions2[0][:, :, 2].flatten()
            d3 = predictions3[0][:, :, 2].flatten()

            d1 = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in d1]
            d2 = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in d2]
            d3 = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in d3]

            d_plot = [d1, d2, d3]


            if p_dimension is None:
                plt.title("OCEAN")
            else:
                plt.title(conf.personality_name[p_dimension])

            bp = plt.boxplot(d_plot, patch_artist=True)
            colors = ['blue', 'green', 'red']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            plt.xticks([1, 2, 3], ["low", "med", "high"])
            plt.show()


        vp1 = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in p1]
        vp2 = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in p2]
        vp3 = [(max_pupils - min_pupils) * (i + 1) / 2.0 + min_pupils for i in p3]

        vp1 = list(zip(vp1, vp1))
        vp2 = list(zip(vp2, vp2))
        vp3 = list(zip(vp3, vp3))

        plt.title('Avg. pupil size')


        plt.ylim(2, 4)
        plt.plot(np.arange(0, time_series_size), vp1, c='b', label='low')
        plt.plot(np.arange(0, time_series_size), vp2, c='g', label='med')
        plt.plot(np.arange(0, time_series_size), vp3, c='r', label='high')

        plt.show()
    if feature_size > 3:
        blinks1 = decoder(predictions1[1])[0]
        blinks2 = decoder(predictions2[1])[0]
        blinks3 = decoder(predictions3[1])[0]


        threshold1 = np.max(blinks1) * 0.5
        threshold2 = np.max(blinks2) * 0.5
        threshold3 = np.max(blinks3) * 0.5

        blinks1 = [1 if i > threshold1 else 0 for i in blinks1]
        blinks2 = [1 if i > threshold2 else 0 for i in blinks1]
        blinks3 = [1 if i > threshold3 else 0 for i in blinks1]


        plt.title("Blinks")
        plt.plot(np.arange(0, time_series_size), blinks1, c='b')  # 1
        plt.plot(np.arange(0, time_series_size), blinks2, c='g')  # 2
        plt.plot(np.arange(0, time_series_size), blinks3, c='r')  # 3
        plt.show()


# always use the same random values for prediction
seed = tf.random.normal([1, latent_dim])


# tf.function causes the function to be "compiled".
@tf.function
def train_step(x, p_labels, prev_half_noise=None):

    if prev_half_noise is None:
        prev_half_noise = tf.random.normal([conf.batch_size, half_latent_dim])


    # else:
    #     print("prev_half_noise")
    half_noise = tf.random.normal([conf.batch_size, half_latent_dim])


    # noise = tf.concat([prev_half_noise, half_noise], axis=1)

    noise = tf.random.normal([conf.batch_size, latent_dim])

    generated_labels = randint(0, n_classes, conf.batch_size)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images, generated_blinks = generator([noise, generated_labels],  training=True)

        decoded_blinks = decoder(generated_blinks)

        decoded_blinks = tf.reshape(decoded_blinks, (decoded_blinks.shape[0], decoded_blinks.shape[1], 1))

        generated_data = tf.concat((generated_images, decoded_blinks), axis=2)


        real_output = discriminator([x, p_labels], training=True)
        fake_output = discriminator([generated_data, generated_labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    return (gen_loss ,disc_loss, half_noise)


def train(dataset, epochs):
    prev_half_noise = None

    for epoch in range(epochs):
        start = time.time()

        # generate_and_save_data(generator)

        for image_batch in dataset:
            x, p_labels = image_batch

            (gen_loss, disc_loss, half_noise) = train_step(x, p_labels, prev_half_noise)

            prev_half_noise = half_noise

        K.print_tensor(gen_loss, message='gen loss =')
        K.print_tensor(disc_loss, message='disc loss =')

        # generate and save at each epoch
        plot_predictions(generator, seed)

        generator.save(model_file)

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


def evaluate_model(model):

    # p_class_label = od.convert_personality_to_class_label(p)

    #plot average x for synthetic and real data
    feature_ind = 0 # x
    # feature_ind = 1 # y
    # feature_ind = 2 # pupil

    x_train_avg_all = []
    x_fake_avg_all = []
    for p_class_label in range(1):
        x_train_avg = []
        for i in range(p_labels.shape[0]):
            if p_labels[i][0] == p_class_label:
                x_train_avg = np.concatenate((x_train_avg, [np.average(x_train[i,:, feature_ind])]), axis=0)
        x_train_avg_all = np.concatenate((x_train_avg_all, x_train_avg), axis=0)
        if len(x_train_avg) > 0:
            #make as many predictions
            test_labels = np.repeat(np.array([p_class_label]), x_train_avg.shape[0], axis = 0)
            z_input = tf.random.normal([x_train_avg.shape[0], latent_dim])


            predictions_all = model.predict([z_input, test_labels])
            predictions0 = np.average(predictions_all[0], axis = 1)


            vx = predictions0[:, feature_ind, 0]

            x_fake_avg_all = np.concatenate((x_fake_avg_all, vx), axis=0)

    g = np.sort(x_fake_avg_all)
    r = np.sort(x_train_avg_all)

    g = (g + 1) / 2.0
    r = (r + 1) / 2.0
    # plt.title(p_class_label)


    fig, ax = plt.subplots()
    ax.scatter(g, r, s=0.5)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show()



#
decoder = ae.load_decoder()
if train_mode:
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    train(train_data, conf.n_epochs)


model = tf.keras.models.load_model(model_file)
# evaluate_model(model)
generate_and_save_data_1d(model, time_series_size * 2)


# generate_and_save_data_multiple(model)


# bootstrap_size = 1000
# seed = tf.random.normal([bootstrap_size, latent_dim])
# plot_predictions(model, seed)




