#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

@author: thieunv
"""
from preprocessing import TimeSeries
from cluster import Clustering
from utils import MathHelper, GraphUtilClient, IOHelper

from math import sqrt
from pandas import read_csv
import numpy as np
from copy import deepcopy
from random import random, randint
from operator import add, itemgetter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing


class Model(object):
    def __init__(self, dataset_original=None, list_idx=(1000, 2000, 0), output_index=0, sliding=2, method_statistic=0, max_cluster=25,
                 positive_number=0.15, sti_level=0.15, dis_level=0.25, mutation_id=1, activation_id=0, activation_id2=0, pathsave=None,
                 max_gens=100, num_bees=45, num_sites=3, elite_sites=1, patch_size=3.0, patch_factor=0.985, e_bees=7, o_bees=2,
                 low_up_w=(-1, 1), low_up_b=(-1, 1)):
        self.dataset_original = dataset_original
        self.output_index = output_index
        self.sliding = sliding
        self.max_cluster = max_cluster
        self.method_statistic = method_statistic
        self.stimulation_level = sti_level
        self.distance_level = dis_level
        self.positive_number = positive_number
        self.mutation_id = mutation_id
        self.activation_id = activation_id
        self.activation_id2 = activation_id2
        self.pathsave = pathsave
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

        if activation_id2 == 0:
            self.activation2 = MathHelper.elu
        elif activation_id2 == 1:
            self.activation2 = MathHelper.relu
        elif activation_id2 == 2:
            self.activation2 = MathHelper.tanh
        else:
            self.activation2 = MathHelper.sigmoid

        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        if list_idx[2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(list_idx[0] + (list_idx[1] - list_idx[0]) / 2)

        self.filename = 'Slid=' + str(sliding) + '_PN=' + str(positive_number) + '_SL=' + str(sti_level) + '_DL=' + str(dis_level) + '_MG=' + str(max_gens) + '_NB=' + str(num_bees)

    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        print("Processing data done!!!")


    def clustering_data(self):
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number, max_cluster=self.max_cluster,
                                distance_level=self.distance_level, mutation_id=self.mutation_id, activation_id=self.activation_id, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers, self.y = self.clustering.sobee_new_with_mutation()
        print("Encoder features done!!!")

    def transform_data(self):
        self.S_train = self.clustering.transform_features(self.X_train)
        self.S_test = self.clustering.transform_features(self.X_test)
        if self.valid_idx != 0:
            self.S_valid = self.clustering.transform_features(self.X_valid)
        print("Transform features done!!!")


    def create_search_space(self, low_up_w=None, low_up_b=None):  # [ [-1, 1], [-1, 1], ... ]
        self.number_node_input = len(self.list_clusters)
        self.number_node_output = self.y_train.shape[1]
        self.size_w2 = self.number_node_input * self.number_node_output
        self.size_b2 = self.number_node_output
        w2 = [low_up_w for i in range(self.size_w2)]
        b2 = [low_up_b for i in range(self.size_b2)]
        search_space = w2 + b2
        return search_space

    def create_candidate(self, minmax=None):
        candidate = [(minmax[i][1] - minmax[i][0]) * random() + minmax[i][0] for i in range(len(minmax))]
        #        return candidate
        fitness = self.get_mae(candidate, self.S_train, self.y_train)
        return [candidate, fitness]

    def get_mae(self, bee=None, X_data=None, y_data=None):
        mae_list = []

        w2 = np.reshape(bee[:self.size_w2], (self.number_node_input, -1))
        b2 = np.reshape(bee[self.size_w2:], (-1, self.size_b2))

        for i in range(0, len(X_data)):
            output = np.add(np.matmul(X_data[i], w2), b2)
            y_pred = np.apply_along_axis(self.activation2, 1, output)
            mae_list.append(np.average(np.power((y_data[i].flatten() - y_pred), 2)))
        temp = reduce(add, mae_list, 0)
        return temp / (len(X_data) * 1.0)

    def create_random_bee(self, search_space):
        return self.create_candidate(search_space)

    def objective_function(self, vector):
        return self.get_mae(vector, self.S_train, self.y_train)

    def create_neigh_bee(self, individual, patch_size, search_space):
        t1 = randint(0, len(individual) - 1)

        bee = deepcopy(individual)
        if random() < 0.5:
            bee[t1] = individual[t1] + random() * patch_size
        else:
            bee[t1] = individual[t1] - random() * patch_size

        if bee[t1] < search_space[t1][0]:
            bee[t1] = search_space[t1][0]
        if bee[t1] > search_space[t1][1]:
            bee[t1] = search_space[t1][1]

        fitness = self.get_mae(bee, self.S_train, self.y_train)
        return [bee, fitness]


    def create_neigh_bee2(self, individual, patch_size, search_space):
        bee = []
        for x in range(0, len(individual)):
            if random() < 0.5:
                elem = individual[x] + random() * patch_size
            else:
                elem = individual[x] - random() * patch_size

            if elem < search_space[x][0]:
                elem = search_space[x][0]
            if elem > search_space[x][1]:
                elem = search_space[x][1]
            bee.append(deepcopy(elem))

        fitness = self.get_mae(bee, self.S_train, self.y_train)
        return [bee, fitness]

    def search_neigh(self, parent, neigh_size, patch_size, search_space):  # parent:  [ vector_individual, fitness ]
        """
        Tim kiem trong so neigh_size, chi lay 1 hang xom tot nhat
        """
        neigh = [self.create_neigh_bee(parent[0], patch_size, search_space) for x in range(0, neigh_size)]
        #        neigh = [(bee, self.objective_function(bee)) for bee in neigh]
        neigh_sorted = sorted(neigh, key=itemgetter(1))
        return neigh_sorted[0]

    def create_scout_bees(self, search_space, num_scouts):  # So luong ong trinh tham
        return [self.create_random_bee(search_space) for x in range(0, num_scouts)]

    def search(self, max_gens, search_space, num_bees, num_sites, elite_sites, patch_size, patch_factor, e_bees, o_bees):
        pop = [self.create_random_bee(search_space) for x in range(0, num_bees)]
        self.loss_train = []
        for j in range(0, max_gens):
            #            pop = [(bee, self.objective_function(bee)) for bee in pop]
            pop_sorted = sorted(pop, key=itemgetter(1))
            best = pop_sorted[0]

            next_gen = []
            for i in range(0, num_sites):
                if i < elite_sites:
                    neigh_size = e_bees
                else:
                    neigh_size = o_bees
                next_gen.append(self.search_neigh(pop_sorted[i], neigh_size, patch_size, search_space))

            scouts = self.create_scout_bees(search_space, (num_bees - num_sites))  # Ong trinh tham
            pop = next_gen + scouts
            patch_size = patch_size * patch_factor
            self.loss_train.append(best[1])
            print("Epoch = {0}, patch_size = {1}, best = {2}".format(j + 1, patch_size, best[1]))
        return best

    def build_and_train(self):
        search_space = self.create_search_space(self.low_up_w, self.low_up_b)
        best = self.search(self.max_gens, search_space, self.num_bees, self.num_sites, self.elite_sites,
                           self.patch_size, self.patch_factor, self.e_bees, self.o_bees)
        print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
        self.bee = best[0]



    def predict(self):
        w2 = np.reshape(self.bee[:self.size_w2], (self.number_node_input, -1))
        b2 = np.reshape(self.bee[self.size_w2:], (-1, self.size_b2))
        y_pred = []
        for i in range(0, len(self.S_test)):
            output = np.add(np.matmul(self.S_test[i], w2), b2)
            out = np.apply_along_axis(self.activation2, 1, output)
            y_pred.append(self.activation2(out.flatten()))

        # Evaluate models on the test set
        y_test_inverse = self.min_max_scaler.inverse_transform(self.y_test)
        y_pred_inverse = self.min_max_scaler.inverse_transform(np.array(y_pred).reshape(-1, self.y_test.shape[1]))

        testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

        self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_pred, testScoreRMSE, testScoreMAE
        self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse

        print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        print("Predict done!!!")


    def draw_result(self):
        GraphUtilClient.draw_loss(fig_id, self.max_gens, self.loss_train, "Loss on training per epoch")
        GraphUtilClient.draw_predict_with_mae(fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.score_test_RMSE,
                                        self.score_test_MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, pathsave)

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        if self.count_centers <= self.max_cluster:
            self.transform_data()
            self.build_and_train()
            self.predict()
            self.draw_result()
            # self.save_result()


pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sonia/result/cpu_ram_cpu/"
fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"
filename3 = "data_resource_usage_3Minutes_6176858948.csv"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
filename8 = "data_resource_usage_8Minutes_6176858948.csv"
filename10 = "data_resource_usage_10Minutes_6176858948.csv"
df = read_csv(fullpath+ filename3, header=None, index_col=False, usecols=[3], engine='python')
dataset_original = df.values


list_num3 = (11120, 13900, 0)
list_num5 = (6640, 8300, 0)
list_num8 = (4160, 5200, 0)
list_num10 = (3280, 4100, 0)

output_index = 0
method_statistic = 0
max_cluster=15
neighbourhood_density=0.2
gauss_width=1.0
mutation_id=1
activation_id= 0            # 0: elu, 1:relu, 2:tanh, 3:sigmoid
activation_id2 = 0


sliding_windows = [2]  # [ 2, 3, 5]
positive_numbers = [0.25]  # [0.05, 0.15, 0.35]
stimulation_levels = [0.20]  # [0.10, 0.25, 0.45]
distance_levels = [0.75] # [0.65, 0.75, 0.85]

list_max_gens = [100]  # epoch
list_num_bees = [24]  # number of bees - population
num_sites = 3  # phan vung, 3 dia diem
elite_sites = 1
patch_size = 5.0
patch_factor = 0.97
e_bees = 10
o_bees = 3
low_up_w = [-1, 1]          # Lower and upper values for weights
low_up_b = [-1, 1]


fig_id = 1
so_vong_lap = 0
for sliding in sliding_windows:
    for pos_number in positive_numbers:
        for sti_level in stimulation_levels:
            for dist_level in distance_levels:

                for max_gens in list_max_gens:
                    for num_bees in list_num_bees:

                        my_model = Model(dataset_original, list_num3, output_index, sliding, method_statistic, max_cluster,
                                         pos_number, sti_level, dist_level, mutation_id, activation_id, activation_id2, pathsave,
                                         max_gens, num_bees, num_sites, elite_sites, patch_size, patch_factor, e_bees, o_bees, low_up_w, low_up_b)
                        my_model.fit()
                        so_vong_lap += 1
                        fig_id += 2
                        if so_vong_lap % 5000 == 0:
                            print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"


