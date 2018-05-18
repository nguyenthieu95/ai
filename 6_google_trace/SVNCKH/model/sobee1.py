#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

@author: thieunv
"""
from preprocessing import TimeSeries
from cluster import Clustering
from utils import MathHelper, GraphUtil, IOHelper
from algorithm import Bee1

from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

class Model(object):
    def __init__(self, para_data=None, para_net=None, para_bee=None):
        self.dataset_original = para_data["dataset"]
        self.train_idx = para_data["list_index"][0]
        self.test_idx = para_data["list_index"][1]
        if para_data["list_index"][2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(para_data["list_index"][0] + (para_data["list_index"][1] - para_data["list_index"][0]) / 2)
        self.output_index = para_data["output_index"]
        self.method_statistic = para_data["method_statistic"]
        self.sliding = para_data["sliding"]

        self.max_cluster = para_net["max_cluster"]
        self.positive_number = para_net["pos_number"]
        self.stimulation_level = para_net["sti_level"]
        self.distance_level = para_net["dist_level"]
        self.mutation_id = para_net["mutation_id"]
        self.activation_id1 = para_net["couple_activation"][0]
        if para_net["couple_activation"][0] == 0:
            self.activation1 = MathHelper.elu
        elif para_net["couple_activation"][0] == 1:
            self.activation1 = MathHelper.relu
        elif para_net["couple_activation"][0] == 2:
            self.activation1 = MathHelper.tanh
        else:
            self.activation1 = MathHelper.sigmoid
        if para_net["couple_activation"][1] == 0:
            self.activation2 = MathHelper.elu
        elif para_net["couple_activation"][1] == 1:
            self.activation2 = MathHelper.relu
        elif para_net["couple_activation"][1] == 2:
            self.activation2 = MathHelper.tanh
        else:
            self.activation2 = MathHelper.sigmoid
        self.pathsave = para_net["path_save"]
        self.fig_id = para_net["fig_id"]

        self.max_gens = para_bee["max_gens"]
        self.num_bees = para_bee["num_bees"]
        self.num_sites = para_bee["num_sites"]
        self.elite_sites = para_bee["elite_sites"]
        self.patch_size = para_bee["patch_size"]
        self.patch_factor = para_bee["patch_factor"]
        self.e_bees = para_bee["e_bees"]
        self.o_bees = para_bee["o_bees"]
        self.low_up_w = para_bee["lowup_w"]
        self.low_up_b = para_bee["lowup_b"]

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()

        self.filename = 'Slid=' + str(self.sliding) + '_PN=' + str(self.positive_number) + '_SL=' + str(self.stimulation_level) + '_DL=' + str(
            self.distance_level) + '_MG=' + str(self.max_gens) + '_NB=' + str(self.num_bees)


    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        # print("Processing data done!!!")


    def clustering_data(self):
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number, max_cluster=self.max_cluster,
                                distance_level=self.distance_level, mutation_id=self.mutation_id, activation_id=self.activation_id1, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers, self.y = self.clustering.sobee_new_with_mutation()
        # print("Encoder features done!!!")

    def transform_data(self):
        self.S_train = self.clustering.transform_features(self.X_train)
        self.S_test = self.clustering.transform_features(self.X_test)
        if self.valid_idx != 0:
            self.S_valid = self.clustering.transform_features(self.X_valid)
        # print("Transform features done!!!")

    def train_bee(self):
        self.number_node_input = len(self.list_clusters)
        self.number_node_output = self.y_train.shape[1]
        self.size_w2 = self.number_node_input * self.number_node_output

        bee_para = {
            "max_gens": self.max_gens, "num_bees": self.num_bees, "num_sites": self.num_sites,
                       "elite_sites": self.elite_sites, "patch_size": self.patch_size, "patch_factor": self.patch_factor,
            "e_bees": self.e_bees, "o_bees": self.o_bees, "lowup_w": self.low_up_w, "lowup_b": self.low_up_b
        }
        other_para = {
            "number_node_input": self.number_node_input, "number_node_output": self.number_node_output,
            "X_data": self.S_train, "y_data": self.y_train, "activation": self.activation2
        }
        bee = Bee1(other_para, bee_para)
        self.bee, self.loss_train = bee.build_and_train()

    def predict(self):
        w2 = np.reshape(self.bee[:self.size_w2], (self.number_node_input, -1))
        b2 = np.reshape(self.bee[self.size_w2:], (-1, self.number_node_output))
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

        # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        # print("Predict done!!!")


    def draw_result(self):
        GraphUtil.draw_loss(self.fig_id, self.max_gens, self.loss_train, "Loss on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.score_test_RMSE,
                                        self.score_test_MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)
        IOHelper.save_loss_to_csv(self.loss_train, self.filename, self.pathsave)

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        if self.count_centers <= self.max_cluster:
            self.transform_data()
            self.train_bee()
            self.predict()
            self.draw_result()
            self.save_result()
