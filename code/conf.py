participant_cnt = 42
feature_size = 7   # x, y, pl, pr, one_hot(blink, fixation, saccade)
time_series_size = 240
iter_cnt = 4 # how many consecutive time series to generate
window_delta = int(time_series_size / 10)
batch_size = 64
n_epochs = 100
kernel_size = 15 #3
plot_stats = False  #  Boxplots and average gaze positions for further analysis

#  Personality constants
N = 0
E = 1
O = 2
A = 3
C = 4
personality_name = ["N", "E", "O", "A", "C"]