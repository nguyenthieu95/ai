from pandas import read_csv
import tensorflow as tf
from sklearn import preprocessing
from preprocessing import TimeSeries
from cluster_test import Clustering

class Model(object):
    def __init__(self, dataset_original=None, list_idx=(1000, 2000, 0), output_index=0, epoch=100, batch_size=32, learning_rate=0.1,
                 sliding=2, method_statistic=0,
                 max_cluster=15, positive_number=0.15, sti_level=0.15, dis_level=0.25, mutation_id=1, couple_acti=(2,3), fig_id=0, pathsave=None):
        self.dataset_original = dataset_original
        self.output_index = output_index
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.max_cluster = max_cluster
        self.learning_rate = learning_rate
        self.stimulation_level = sti_level
        self.distance_level = dis_level
        self.positive_number = positive_number
        self.learning_rate = learning_rate
        self.mutation_id = mutation_id
        self.activation_id1 = couple_acti[0]
        if couple_acti[1] == 0:
            self.activation2 = tf.nn.elu
        elif couple_acti[1] == 1:
            self.activation2 = tf.nn.relu
        elif couple_acti[1] == 2:
            self.activation2 = tf.nn.tanh
        else:
            self.activation2 = tf.nn.sigmoid

        self.method_statistic = method_statistic
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        self.fig_id = fig_id
        self.pathsave = pathsave
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        if list_idx[2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(list_idx[0] + (list_idx[1] - list_idx[0]) / 2)

        self.filename = '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_Slid=' + str(
            sliding) + '_PN=' + str(positive_number) + 'SL=' + str(sti_level) + 'DL=' + str(dis_level)

    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        print("Processing data done!!!")

    def clustering_data(self):
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number, max_cluster=self.max_cluster,
                                distance_level=self.distance_level, mutation_id=self.mutation_id, activation_id=self.activation_id1, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers, self.y = self.clustering.kmeans_with_mutation()
        print("Encoder features done!!!")

    def transform_data(self):
        self.S_train = self.clustering.transform_features(self.X_train)
        self.S_test = self.clustering.transform_features(self.X_test)
        if self.valid_idx != 0:
            self.S_valid = self.clustering.transform_features(self.X_valid)
        # print("Transform features done!!!")

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        if self.count_centers <= self.max_cluster:
            self.transform_data()


pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sonia/result/cpu/"
fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

# pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/3m/sonia/result/cpu/"
# fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"

filename3 = "data_resource_usage_3Minutes_6176858948.csv"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
filename8 = "data_resource_usage_8Minutes_6176858948.csv"
filename10 = "data_resource_usage_10Minutes_6176858948.csv"
df = read_csv(fullpath+ filename10, header=None, index_col=False, usecols=[3, 4], engine='python')
dataset_original = df.values

list_num3 = (11120, 13900, 0)
list_num5 = (6640, 8300, 0)
list_num8 = (4160, 5200, 0)
list_num10 = (3280, 4100, 0)

output_index = 0                # 0: cpu, 1: ram
method_statistic = 0
max_cluster=30
mutation_id=1
couple_acti = (2, 0)        # 0: elu, 1:relu, 2:tanh, 3:sigmoid

epochs = [480]
batch_sizes = [8]
learning_rates = [0.15]

sliding_windows = [ 2 ]
positive_numbers =  [0.05]
stimulation_levels = [0.15]
distance_levels = [0.65]

fig_id = 1
so_vong_lap = 0
for epoch in epochs:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for sliding in sliding_windows:
                for positive_number in positive_numbers:
                    for sti_level in stimulation_levels:
                        for dis_level in distance_levels:
                            my_model = Model(dataset_original, list_num10, output_index, epoch, batch_size, learning_rate, sliding, method_statistic, max_cluster,
                                             positive_number, sti_level, dis_level, mutation_id, couple_acti, fig_id, pathsave)
                            my_model.fit()
                            so_vong_lap += 1
                            fig_id += 2
                            if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
