#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

@author: thieunv
"""
from preprocessing import TimeSeries
from cluster_test import Clustering
from pandas import read_csv
from utils import MathHelper
from sklearn import preprocessing

class Model(object):
    def __init__(self, dataset_original=None, list_idx=(1000, 2000, 1), output_index=0, sliding=2, method_statistic=0, max_cluster=15,
                 positive_number=0.15, sti_level=0.15, dis_level=0.25, neighbourhood_density=0.1, gauss_width=1.0, mutation_id=1, couple_acti=(2, 0), fig_id=0, pathsave=None,
                 max_gens=100, num_bees=45, num_sites=3, elite_sites=1, patch_size=3.0, patch_factor=0.985, e_bees=7, o_bees=2,
                 low_up_w=(-1, 1), low_up_b=(-1, 1) ):
        self.dataset_original = dataset_original
        self.output_index = output_index
        self.sliding = sliding
        self.max_cluster = max_cluster
        self.method_statistic = method_statistic
        self.stimulation_level = sti_level
        self.distance_level = dis_level
        self.positive_number = positive_number
        self.neighbourhood_density = neighbourhood_density
        self.gauss_width = gauss_width
        self.mutation_id = mutation_id
        self.pathsave = pathsave
        self.fig_id = fig_id
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()

        self.max_gens = max_gens
        self.num_bees = num_bees
        self.num_sites = num_sites
        self.elite_sites = elite_sites
        self.patch_size = patch_size
        self.patch_factor = patch_factor
        self.e_bees = e_bees
        self.o_bees = o_bees
        self.low_up_w = low_up_w
        self.low_up_b = low_up_b

        self.activation_id1 = couple_acti[0]
        if couple_acti[0] == 0:
            self.activation1 = MathHelper.elu
        elif couple_acti[0] == 1:
            self.activation1 = MathHelper.relu
        elif couple_acti[0] == 2:
            self.activation1 = MathHelper.tanh
        else:
            self.activation1 = MathHelper.sigmoid

        if couple_acti[1] == 0:
            self.activation2 = MathHelper.elu
        elif couple_acti[1] == 1:
            self.activation2 = MathHelper.relu
        elif couple_acti[1] == 2:
            self.activation2 = MathHelper.tanh
        else:
            self.activation2 = MathHelper.sigmoid

        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        if list_idx[2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(list_idx[0] + (list_idx[1] - list_idx[0]) / 2)

        self.filename = 'Slid=' + str(sliding) + '_PN=' + str(positive_number) + '_SL=' + str(sti_level) + '_DL=' + str(
            dis_level) + '_MG=' + str(max_gens) + '_NB=' + str(num_bees)


    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        print("Processing data done!!!")

    def clustering_data(self):
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number, max_cluster=self.max_cluster,
                                     neighbourhood_density=self.neighbourhood_density, gauss_width=self.gauss_width,
                                distance_level=self.distance_level, mutation_id=self.mutation_id, activation_id=self.activation_id1, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers, self.y = self.clustering.sobee_new_no_mutation()
        print("Encoder features done!!!")

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()


pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sopso/result/cpu_ram_cpu/"
fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

# pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/3m/sobee/result/cpu/"
# fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"

filename3 = "data_resource_usage_3Minutes_6176858948.csv"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
filename8 = "data_resource_usage_8Minutes_6176858948.csv"
filename10 = "data_resource_usage_10Minutes_6176858948.csv"
df = read_csv(fullpath+ filename10, header=None, index_col=False, usecols=[3, 5], engine='python')
dataset_original = df.values

list_num3 = (11120, 13900, 0)
list_num5 = (6640, 8300, 0)
list_num8 = (4160, 5200, 0)
list_num10 = (3280, 4100, 0)

output_index = 0
method_statistic = 0                    #### changes
max_cluster=20
neighbourhood_density=0.2
gauss_width=1.0
mutation_id=1
couple_acti = (0, 0)           # 0: elu, 1:relu, 2:tanh, 3:sigmoid

list_max_gens = [180]
list_num_bees = [16]                # number of bees - population
num_sites = 3                               # phan vung, 3 dia diem
elite_sites = 1
patch_size = 5.0
patch_factor = 0.97
e_bees = 7
o_bees = 3
low_up_w = [-0.2, 0.6]                      # Lower and upper values for weights
low_up_b = [-0.5, 0.5]


sliding_windows = [ 1 ]
positive_numbers =  [0.15]                  #### changes
stimulation_levels = [0.20]                 #### changes
distance_levels = [0.65]

fig_id = 1
so_vong_lap = 0
for sliding in sliding_windows:
    for pos_number in positive_numbers:
        for sti_level in stimulation_levels:
            for dist_level in distance_levels:

                for max_gens in list_max_gens:
                    for num_bees in list_num_bees:

                        my_model = Model(dataset_original, list_num10, output_index, sliding, method_statistic, max_cluster,
                                               pos_number, sti_level, dist_level, neighbourhood_density, gauss_width, mutation_id, couple_acti, fig_id, pathsave,
                                               max_gens, num_bees, num_sites, elite_sites, patch_size, patch_factor, e_bees, o_bees, low_up_w, low_up_b)
                        my_model.fit()
                        so_vong_lap += 1
                        fig_id += 2
                        if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)
print "Processing DONE !!!"

