#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

So sanh: 3 model ANN, SONIA, SoBee

@author: thieunv
"""
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from preprocessing import TimeSeries
from cluster import Clustering
from algorithm import Bee1
from utils import MathHelper, GraphUtil, IOHelper

class ANN(object):
    def __init__(self, para_data=None, para_net=None):

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
        self.tf = para_data["tf"]

        self.epoch = para_net["epoch"]
        self.batch_size = para_net["batch_size"]
        self.learning_rate = para_net["learning_rate"]
        self.num_hidden = para_net["num_hidden"]
        self.couple_activation = para_net["couple_activation"]
        self.pathsave = para_net["path_save"]
        self.fig_id = para_net["fig_id"]
        self.filename = para_net["model_name"]
        self.model_name = para_net["model_name"]

        if para_net["couple_activation"][0] == 0:
            self.activation1 = self.tf.nn.elu
        elif para_net["couple_activation"][0] == 1:
            self.activation1 = self.tf.nn.relu
        elif para_net["couple_activation"][0] == 2:
            self.activation1 = self.tf.nn.tanh
        else:
            self.activation1 = self.tf.nn.sigmoid

        if para_net["couple_activation"][1] == 0:
            self.activation2 = self.tf.nn.elu
        elif para_net["couple_activation"][1] == 1:
            self.activation2 = self.tf.nn.relu
        elif para_net["couple_activation"][1] == 2:
            self.activation2 = self.tf.nn.tanh
        else:
            self.activation2 = self.tf.nn.sigmoid

        if para_net["optimizer"] == 0:
            self.optimizer = self.tf.train.GradientDescentOptimizer
        elif para_net["optimizer"] == 1:
            self.optimizer = self.tf.train.AdamOptimizer
        elif para_net["optimizer"] == 2:
            self.optimizer = self.tf.train.AdagradOptimizer
        else:
            self.optimizer = self.tf.train.AdadeltaOptimizer

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.loss_train = None

    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        # print("Processing data done!!!")

    # Create and train a tensorflow model of a neural network
    def build_model_and_train(self):
        # Reset the graph
        self.tf.reset_default_graph()

        X_train, y_train = self.X_train, self.y_train

        X_size = X_train.shape[1]
        h_size = self.num_hidden
        y_size = y_train.shape[1]

        # Placeholders for input and output data
        X = self.tf.placeholder("float64", shape=[None, X_size], name='X')
        y = self.tf.placeholder("float64", shape=[None, y_size], name='y')

        # now declare the weights connecting the input to the hidden layer
        W1 = self.tf.Variable(self.tf.random_normal([X_size, h_size], stddev=0.03, dtype=self.tf.float64), name="W1")
        b1 = self.tf.Variable(self.tf.random_normal([h_size], dtype=self.tf.float64), name="b1")
        # and the weights connecting the hidden layer to the output layer
        W2 = self.tf.Variable(self.tf.random_normal([h_size, y_size], stddev=0.03, dtype=self.tf.float64), name='W2')
        b2 = self.tf.Variable(self.tf.random_normal([y_size], dtype=self.tf.float64), name='b2')

        # calculate the output of the hidden layer
        hidden_out = self.activation1(self.tf.add(self.tf.matmul(X, W1), b1))
        # Forward propagation # now calculate the hidden layer output
        y_ = self.activation2(self.tf.add(self.tf.matmul(hidden_out, W2), b2))

        # Loss function
        deltas = self.tf.square(y_ - y)
        loss = self.tf.reduce_mean(deltas)

        # Backward propagation
        opt = self.optimizer(learning_rate=self.learning_rate)
        train = opt.minimize(loss)

        # Initialize variables and run session
        init = self.tf.global_variables_initializer()
        sess = self.tf.Session()
        sess.run(init)

        # Go through num_iters iterations
        loss_plot = []
        weights1 = None
        bias1 = None
        weights2 = None
        bias2 = None
        total_batch = int(len(X_train) / self.batch_size) + 1
        for epoch in range(self.epoch):
            for i in range(total_batch):
                ## Get next batch
                batch_x, batch_y = MathHelper.get_batch_data_next(X_train, y_train, i, self.batch_size)
                if len(batch_x) == 0:
                    break
                sess.run(train, feed_dict={X: batch_x, y: batch_y})

            loss_epoch = sess.run(loss, feed_dict={X: X_train, y: y_train})
            weights1, bias1, weights2, bias2 = sess.run([W1, b1, W2, b2])
            loss_plot.append(loss_epoch)
            # print("Epoch: {}".format(epoch + 1), "loss = {}".format(loss_epoch))

        sess.close()
        self.w1, self.b1, self.w2, self.b2, self.loss_train = weights1, bias1, weights2, bias2, loss_plot
        # print("Build model and train done!!!")


    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.X_test, self.y_test

        X_size = X_test.shape[1]
        y_size = y_test.shape[1]

        X = self.tf.placeholder("float64", shape=[None, X_size], name='X')
        y = self.tf.placeholder("float64", shape=[None, y_size], name='y')

        W1 = self.tf.Variable(self.w1)
        b1 = self.tf.Variable(self.b1)
        W2 = self.tf.Variable(self.w2)
        b2 = self.tf.Variable(self.b2)

        hidden_out = self.activation1(self.tf.add(self.tf.matmul(X, W1), b1))
        y_ = self.activation2(self.tf.add(self.tf.matmul(hidden_out, W2), b2))

        # Calculate the predicted outputs
        init = self.tf.global_variables_initializer()
        with self.tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})

            y_test_inverse = self.min_max_scaler.inverse_transform(y_test)
            y_pred_inverse = self.min_max_scaler.inverse_transform(y_est_np)

            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

            self.y_predict, self.RMSE, self.MAE = y_est_np, round(testScoreRMSE, 4), round(testScoreMAE, 4)
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse

            # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        # print("Predict done!!!")


    def draw_result(self):
        #GraphUtil.draw_loss(self.fig_id, self.epoch, self.loss_train, "Loss on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id, self.y_test_inverse, self.y_pred_inverse, self.RMSE, self.MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)

    def fit(self):
        self.preprocessing_data()
        self.build_model_and_train()
        self.predict()
        self.draw_result()
        self.save_result()




class SONIA(object):
    def __init__(self, para_data=None, para_net=None):
        self.dataset_original = para_data["dataset"]
        self.train_idx = para_data["list_index"][0]
        self.test_idx = para_data["list_index"][1]
        if para_data["list_index"][2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(
                para_data["list_index"][0] + (para_data["list_index"][1] - para_data["list_index"][0]) / 2)
        self.output_index = para_data["output_index"]
        self.method_statistic = para_data["method_statistic"]
        self.sliding = para_data["sliding"]
        self.tf = para_data["tf"]


        self.epoch = para_net["epoch"]
        self.batch_size = para_net["batch_size"]
        self.learning_rate = para_net["learning_rate"]
        self.max_cluster = para_net["max_cluster"]
        self.positive_number = para_net["pos_number"]
        self.stimulation_level = para_net["sti_level"]
        self.distance_level = para_net["dist_level"]
        self.mutation_id = para_net["mutation_id"]
        self.activation_id1 = para_net["couple_activation"][0]

        if para_net["couple_activation"][1] == 0:
            self.activation2 = self.tf.nn.elu
        elif para_net["couple_activation"][1] == 1:
            self.activation2 = self.tf.nn.relu
        elif para_net["couple_activation"][1] == 2:
            self.activation2 = self.tf.nn.tanh
        else:
            self.activation2 = self.tf.nn.sigmoid

        self.pathsave = para_net["path_save"]
        self.fig_id = para_net["fig_id"]

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        self.weight = None
        self.bias = None
        self.loss_train = None

        self.filename = para_net["model_name"]
        self.model_name = para_net["model_name"]

    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        # print("Processing data done!!!")


    def clustering_data(self):
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                     max_cluster=self.max_cluster,
                                     distance_level=self.distance_level, mutation_id=self.mutation_id,
                                     activation_id=self.activation_id1, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers, self.y = self.clustering.sonia_with_mutation()
        # print("Encoder features done!!!")


    def transform_data(self):
        self.S_train = self.clustering.transform_features(self.X_train)
        self.S_test = self.clustering.transform_features(self.X_test)
        if self.valid_idx != 0:
            self.S_valid = self.clustering.transform_features(self.X_valid)
        # print("Transform features done!!!")


    def build_and_train(self):
        ## Build layer's sizes
        X_train, y_train = self.S_train, self.y_train

        X_size = X_train.shape[1]
        h_size = len(self.list_clusters)
        y_size = y_train.shape[1]
        ## Symbols
        X = self.tf.placeholder("float64", shape=[None, X_size])
        y = self.tf.placeholder("float64", shape=[None, y_size])

        W = self.tf.Variable(self.tf.random_normal([h_size, y_size], stddev=0.03, dtype=self.tf.float64), name="W")
        b = self.tf.Variable(self.tf.random_normal([y_size], dtype=self.tf.float64), name="b")
        y_ = self.activation2(self.tf.add(self.tf.matmul(X, W), b))

        # Backward propagation
        cost = self.tf.reduce_mean(self.tf.square(y_ - y))
        updates = self.tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(cost)

        loss_plot = []
        weight = None
        bias = None
        # start the session
        with self.tf.Session() as sess:
            # initialise the variables
            init_op = self.tf.global_variables_initializer()
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
        # print("Build model and train done!!!")

    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.S_test, self.y_test

        X_size = X_test.shape[1]
        y_size = y_test.shape[1]
        X = self.tf.placeholder("float64", shape=[None, X_size], name='X')
        y = self.tf.placeholder("float64", shape=[None, y_size], name='y')

        W = self.tf.Variable(self.weight)
        b = self.tf.Variable(self.bias)
        y_ = self.activation2(self.tf.add(self.tf.matmul(X, W), b))

        # Calculate the predicted outputs
        init = self.tf.global_variables_initializer()
        with self.tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})

            y_test_inverse = self.min_max_scaler.inverse_transform(y_test)
            y_pred_inverse = self.min_max_scaler.inverse_transform(y_est_np)

            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

            self.y_predict, self.RMSE, self.MAE = y_est_np, round(testScoreRMSE, 4), round(testScoreMAE, 4)
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse

            # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))

        # print("Predict done!!!")

    def draw_result(self):
        #GraphUtil.draw_loss(self.fig_id, self.epoch, self.loss_train, "Error on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.RMSE,
                                        self.MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)


    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        #if self.count_centers <= self.max_cluster:
        self.transform_data()
        self.build_and_train()
        self.predict()
        self.draw_result()
        self.save_result()



class SOBEE(object):
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
        self.filename = para_net["model_name"]
        self.model_name = para_net["model_name"]

        self.max_gens = para_bee["max_gens"]
        self.num_bees = para_bee["num_bees"]
        self.num_sites = para_bee["num_sites"]
        self.elite_sites = para_bee["elite_sites"]
        self.patch_size = para_bee["patch_size"]
        self.patch_factor = para_bee["patch_factor"]
        self.e_bees = para_bee["couple_bees"][0]
        self.o_bees = para_bee["couple_bees"][1]
        self.low_up_w = para_bee["lowup_w"]
        self.low_up_b = para_bee["lowup_b"]

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()

        self.bee = None
        self.loss_train = None
        self.y_predict = None
        self.RMSE = None
        self.MAE = None
        self.y_test_inverse = None
        self.y_pred_inverse = None
        self.weight = None
        self.bias = None


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
        output = np.add(np.matmul(self.S_test, w2), b2)
        y_pred = self.activation2(output)

        # Evaluate models on the test set
        y_test_inverse = self.min_max_scaler.inverse_transform(self.y_test)
        y_pred_inverse = self.min_max_scaler.inverse_transform(y_pred)

        testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

        self.y_predict, self.RMSE, self.MAE = y_pred, round(testScoreRMSE, 4), round(testScoreMAE, 4)
        self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse
        self.weight = w2
        self.bias = b2

        # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        # print("Predict done!!!")

    def draw_result(self):
        #GraphUtil.draw_loss(self.fig_id, self.max_gens, self.loss_train, "Loss on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.RMSE,
                                        self.MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        #if self.count_centers <= self.max_cluster:
        self.transform_data()
        self.train_bee()
        self.predict()
        self.draw_result()
        self.save_result()
