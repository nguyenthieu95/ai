#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

@author: thieunv
"""

import tensorflow as tf
from math import sqrt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

from preprocessing import TimeSeries
from cluster import Clustering
from utils import MathHelper, GraphUtil, IOHelper

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
        # print("Processing data done!!!")


    def clustering_data(self):
        start_time_cluster = time.time()
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number, max_cluster=self.max_cluster,
                                distance_level=self.distance_level, mutation_id=self.mutation_id, activation_id=self.activation_id1, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers, self.y = self.clustering.sonia_with_mutation()
        self.time_cluster = round(time.time() - start_time_cluster, 4)
        print("Encoder features done!!!")


    def transform_data(self):
        self.S_train = self.clustering.transform_features(self.X_train)
        self.S_test = self.clustering.transform_features(self.X_test)
        if self.valid_idx != 0:
            self.S_valid = self.clustering.transform_features(self.X_valid)
        # print("Transform features done!!!")


    def build_and_train(self):
        start_time_train = time.time()
        ## Build layer's sizes
        X_train, y_train = self.S_train, self.y_train

        X_size = X_train.shape[1]
        h_size = len(self.list_clusters)
        y_size = y_train.shape[1]
        ## Symbols
        X = tf.placeholder("float64", shape=[None, X_size])
        y = tf.placeholder("float64", shape=[None, y_size])

        W = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03, dtype=tf.float64), name="W")
        b = tf.Variable(tf.random_normal([y_size], dtype=tf.float64), name="b")
        y_ = self.activation2(tf.add(tf.matmul(X, W), b))

        # Backward propagation
        cost = tf.reduce_mean(tf.square(y_ - y))
        updates = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(cost)

        loss_plot = []
        # start the session
        with tf.Session() as sess:
            # initialise the variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            total_batch = int(len(X_train) / self.batch_size) + 1
            for epoch in range(self.epoch):

                for i in range(total_batch):
                    ## Get next batch
                    batch_x, batch_y = MathHelper.get_batch_data_next(X_train, y_train, i, self.batch_size)
                    if len(batch_x) == 0:
                        break
                    sess.run(updates, feed_dict={X: batch_x, y: batch_y})

                loss_epoch = sess.run(cost, feed_dict={X: X_train, y: y_train})
                weight, bias = sess.run([W, b])
                loss_plot.append(loss_epoch)
                # print("Epoch: {}".format(epoch + 1), "cost = {}".format(loss_epoch))

        self.weight, self.bias, self.loss_train = weight, bias, loss_plot
        self.time_train = round(time.time() - start_time_train, 4)
        print("Build model and train done!!!")

    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.S_test, self.y_test

        X_size = X_test.shape[1]
        y_size = y_test.shape[1]
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')

        W = tf.Variable(self.weight)
        b = tf.Variable(self.bias)
        y_ = self.activation2(tf.add(tf.matmul(X, W), b))

        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})

            y_test_inverse = self.min_max_scaler.inverse_transform(y_test)
            y_pred_inverse = self.min_max_scaler.inverse_transform(y_est_np)

            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

            self.y_predict, self.RMSE, self.MAE = y_est_np, testScoreRMSE, testScoreMAE
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse

            # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))

        # print("Predict done!!!")

    def draw_result(self):
        GraphUtil.draw_loss(self.fig_id, self.epoch, self.loss_train, "Loss on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.RMSE,
                                        self.MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        if self.count_centers <= self.max_cluster:
            self.transform_data()
            self.build_and_train()
            self.predict()
            self.draw_result()
            self.save_result()


from pandas import read_csv

pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/model/"
fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

filename3 = "data_resource_usage_3Minutes_6176858948.csv"
df = read_csv(fullpath+ filename3, header=None, index_col=False, usecols=[3], engine='python')
dataset_original = df.values

list_num3 = (11120, 13900, 0)
output_index = 0                # 0: cpu, 1: ram
method_statistic = 0
max_cluster=30
mutation_id=1
couple_acti = (2, 0)        # 0: elu, 1:relu, 2:tanh, 3:sigmoid

epochs = [30, 120]
batch_sizes = [8, 16]
learning_rates = [0.05]
sliding_windows = [2, 5]
positive_numbers = [0.05, 0.15]
stimulation_levels = [0.35, 0.55]
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

                            start_time = time.time()
                            my_model = Model(dataset_original, list_num3, output_index, epoch, batch_size, learning_rate, sliding, method_statistic, max_cluster,
                                             positive_number, sti_level, dis_level, mutation_id, couple_acti, fig_id, pathsave)
                            my_model.fit()
                            time_model = round(time.time() - start_time, 4)

                            temp = [my_model.time_cluster, my_model.time_train, time_model]
                            IOHelper.save_model(my_model.list_clusters, my_model.weight, my_model.bias, temp, my_model.filename, my_model.pathsave)

                            so_vong_lap += 1
                            fig_id += 2
                            if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
