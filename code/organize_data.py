import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import conf
from sklearn.preprocessing import LabelEncoder

personality = np.genfromtxt("../features/binned_personality.csv", delimiter=',', skip_header=1)



personality_name = ["N", "E", "O", "A", "C"]

prepare_for_blinks = False

prepare_open = False
if len(sys.argv) > 1 and sys.argv[1] == "true":
    prepare_open = True



def convert_personality_to_class_label(p_per_arr, p_dimension=None):
    """
    :param p_dimension:
    :param p_per_arr: list of a 5-d array for N, E, O , A, C with values 1, 2, 3
    :return:
    """
    p_label = []
    for p in p_per_arr:
        if p_dimension is not None:
            p_label = np.concatenate((p_label, [int(p[int(p_dimension)] - 1)]))
        else:
            p_val = 0
            for i in range(0, 5):
                p_val += (p[i] - 1) * pow(3, i)
            p_label = np.concatenate((p_label, [p_val]), axis=0)

    # # TODO: reduces it to n classes
    # labelencoder = LabelEncoder()
    # p_label = labelencoder.fit_transform(p_label)


    return  p_label


def participant_data(p_ind):
    """
    Combines events, gaze positions and pupil diameter
    :param p_ind: array of p_ind gaze_X gaze_Y pupil_L pupil_R blinkOrNot
    :return:
    """
    p_folder = '/Participant' + str(p_ind).zfill(2)
    gaze_positions_path = "../data" + p_folder + "/gaze_positions.csv"
    events_path = "../data" + p_folder + "/events.csv"
    pupil_diameter_path = "../data" + p_folder + "/pupil_diameter.csv"

    pg = np.genfromtxt(gaze_positions_path, usecols=(1,2), delimiter=',', skip_header=1)
    pp = np.genfromtxt(pupil_diameter_path, usecols=(1,2), delimiter=',', skip_header=1)
    pe = np.genfromtxt(events_path, dtype=str, usecols=0,  skip_header=1)

    #1-hot encode pe just for blinks, saccades and fixations

    pe_encoded = np.array([[1,0,0] if str(s) == "Blink" else [0,1,0] if  str(s) == "Saccade" else [0, 0, 1] for s in pe])

    p_data = np.concatenate((pg, pp, pe_encoded), axis=1)
    # p_data = np.concatenate((p_data, np.resize(pe_encoded, (pe_encoded.shape[0], 1))), axis=1)

    p_ind_arr = np.tile(p_ind, (p_data.shape[0], 1))

    return np.concatenate((p_ind_arr, p_data), axis=1)


def save_all_participant_data():

    p_data = np.empty((0, 5)) # 5 is for personality
    data = np.empty((0, conf.feature_size + 1)) # +1 is for participant index
    for p in range(0, conf.participant_cnt):
        p_in_time = participant_data(p)
        data = np.concatenate((data, p_in_time))

        # p_rep = np.repeat([[participant_personality(p)]], p_in_time.shape[0], axis=0)  # repeat personality for all the time steps
        p_rep = np.repeat([personality[p, 1:6]], p_in_time.shape[0], axis=0)  # repeat personality for all the time steps
        p_data = np.concatenate((p_data, p_rep),axis=0)

    np.save("all_data/participant_data.npy", data)
    np.save("all_data/participant_personality.npy", p_data)


def all_participant_data():
    """
    Columns are as: pId, x, y, pupilLeft, pupilRight, blinkOrNot
    :return:
    """
    data = np.load("all_data/participant_data.npy")
    p_data = np.load("all_data/participant_personality.npy")

    return data, p_data


def save_data(dataset, p_data, history_size):
    """

    :param w_size:
    :param p_data: -- personalities for all participants
    :param start_index: training or validation start index, e.g. 0
    :param end_index: training or validation end index, e.g. TRAIN_SPLIT
    :param history_size: sliding window size
    :return:
    """

    data = []
    p_labels = []
    start_index = history_size
    end_index = dataset.shape[0]

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)

        # check if participant indices are all equal, othersise, don't append
        if np.all((dataset[indices] == dataset[i - history_size, 0])[:, 0]):
            # plt.title("unclean " + str(i))
            # plt.plot(np.arange(time_series_size), dataset[indices][:, 1])
            #
            # plt.show()

            # also check if the data is clean enough
            if is_clean_enough(dataset[indices]):
                # print("clean")
                data.append(np.delete(dataset[indices], 0, axis=1))  # remove the first column with the ppt ids
                p_labels.append(p_data[i - history_size])
                #
                # plt.title("clean " + str(i))
                # plt.plot(np.arange(time_series_size), dataset[indices][:,1])
                #
                # plt.show()

    if prepare_for_blinks:
        np.save('clean_data/clean_data_blinks' + '_' + str(history_size) + '.npy', np.array(data))
        np.save('clean_data/clean_labels_blinks' + '_' + str(history_size) + '.npy', np.array(p_labels))
    else:
        np.save('clean_data/clean_data' + '_' + str(history_size) + '.npy', np.array(data))
        np.save('clean_data/clean_labels' + '_' + str(history_size) + '.npy', np.array(p_labels))

    print(len(data))



def load_data(history_size):
    """

    :param w_size:
    :param p_data: -- personalities for all participants
    :param history_size: sliding window size
    :return:
    """

    data = np.load('clean_data/clean_data' + '_' + str(history_size) + '.npy')
    p_labels = np.load('clean_data/clean_labels'  + '_' + str(history_size) + '.npy')


    return data[::conf.window_delta,:, 0:conf.feature_size], p_labels[::conf.window_delta]


def load_data_blinks(history_size):
    """

    :param w_size:
    :param p_data: -- personalities for all participants
    :param history_size: sliding window size
    :return:
    """

    data = np.load('clean_data/clean_data_blinks' + '_' + str(history_size) + '.npy')
    p_labels = np.load('clean_data/clean_labels_blinks' + '_' + str(history_size) + '.npy')

    return data[:,:, 0:conf.feature_size], p_labels


def plot_data(history_size, p_dim):
    data, p_labels = load_data(history_size)

    x = data[:, :, 0]
    y = data[:, :, 1]
    indices_1 = np.where(p_labels[:, p_dim] == 1)
    indices_2 = np.where(p_labels[:, p_dim] == 2)
    indices_3 = np.where(p_labels[:, p_dim] == 3)

    x1 = x[indices_1]
    x2 = x[indices_2]
    x3 = x[indices_3]



    x1 = np.mean(x1, axis=0)
    x2 = np.mean(x2, axis=0)
    x3 = np.mean(x3, axis=0)


    y1 = np.mean(y[indices_1], axis = 0)
    y2 = np.mean(y[indices_2], axis = 0)
    y3 = np.mean(y[indices_3], axis = 0)
    matplotlib.rcParams.update({'axes.titlesize': 18})

    plt.rc('axes', labelsize=16)
    plt.xlabel('x')
    plt.ylabel('y')
    if p_dim is None:
        plt.title("OCEAN")
    else:
        plt.title(personality_name[p_dim])

    plt.scatter(x1, y1, c = 'b',  label='low')
    plt.scatter(x2, y2, c = 'g',  label='med')
    plt.scatter(x3, y3, c='r',  label='high')
    plt.show()
    # pupils
    pupils = (data[:, :, 2] + data[:, :, 3]) / 2.0


    vp1 = pupils[indices_1]
    vp2 = pupils[indices_2]
    vp3 = pupils[indices_3]

    d_plot = [vp1, vp2, vp3]

    if p_dim is None:
        plt.title("OCEAN")
    else:
        plt.title(personality_name[p_dim])

    bp = plt.boxplot(d_plot, patch_artist=True)
    colors = ['blue', 'green', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks([1, 2, 3], ["low", "med", "high"])
    plt.show()



    vp1 = np.mean(vp1, axis=0)
    vp2 = np.mean(vp2, axis=0)
    vp3 = np.mean(vp3, axis=0)

    if p_dim is None:
        plt.title("OCEAN")
    else:
        plt.title(personality_name[p_dim])

    plt.xlabel('Time step')
    plt.ylabel('Avg. pupil size')


    plt.ylim(2, 4)
    plt.plot(np.arange(0, conf.time_series_size), vp1, c='b')  # 1
    plt.plot(np.arange(0, conf.time_series_size), vp2, c='g')  # 2
    plt.plot(np.arange(0, conf.time_series_size), vp3, c='r')  # 3
    plt.show()


    # blinks1 = data[:,:, 4][indices_1]
    # blinks2 = data[:, :, 4][indices_2]
    # blinks3 = data[:, :, 4][indices_3]
    # plt.title("Blinks")
    #
    # plt.plot(np.arange(0, time_series_size), blinks1[0], c='b')  # 1
    # plt.plot(np.arange(0, time_series_size),  blinks2[0], c='g')  # 2
    # plt.plot(np.arange(0, time_series_size),   blinks3[0], c='r')  # 3
    # plt.show()

# plot_data(time_series_size, 3)

def is_clean_enough(dataset):
    """
    Checks if dataset has less than 50% erroneous samples or it includes saccades or fixations
    :param dataset: p_ind, pos x, pos y, pupil l, pupil r, blink or not
    :return: true or false
    """

    if prepare_for_blinks:

        # if np.count_nonzero(dataset[:, 5] == 1 ) == dataset.shape[0] : #if they are only blinks
        if np.count_nonzero(dataset[:, 5] == 0 ) == dataset.shape[0] or np.count_nonzero(dataset[:, 5] == 1 ) == dataset.shape[0] : #if there are no blinks
           return False


        error_cnt = np.count_nonzero(dataset[:, 1] < -0.5) + np.count_nonzero(dataset[:, 1] > 1.5) +np.count_nonzero( dataset[:, 2] < -0.5) + \
        np.count_nonzero(dataset[:, 2] > 1.5 ) + np.count_nonzero(np.isnan(dataset[:, 1]) ) + np.count_nonzero(np.isnan(dataset[:, 2]) )

        return error_cnt < dataset.shape[0] * 0.01


    else:
        error_cnt = np.count_nonzero(dataset[:, 1] < 0) + np.count_nonzero(dataset[:, 1] > 1) +np.count_nonzero( dataset[:, 2] < 0) + \
        np.count_nonzero(dataset[:, 2] > 1 ) + np.count_nonzero(np.isnan(dataset[:, 1]) ) + np.count_nonzero(np.isnan(dataset[:, 2]) ) + \
                    np.count_nonzero(dataset[:, 3] == 0) + np.count_nonzero(dataset[:, 4] == 0)

        return error_cnt == 0
    # return error_cnt < float(dataset.shape[0]) / 2.0


def prepare_data():

    save_all_participant_data()
    dataset, pers = all_participant_data()
    save_data(dataset[:,0:conf.feature_size+1], pers,  conf.time_series_size)



def participant_statistics(history_size):
    (trainX,  p_labels) = load_data(history_size)

    lab_flat = np.ndarray.flatten(p_labels)
    unique, counts = np.unique(lab_flat, return_counts=True)
    print(dict(zip(unique, counts)))




if prepare_open:
    prepare_data()
