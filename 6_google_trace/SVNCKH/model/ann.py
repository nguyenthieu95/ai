#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

@author: thieunv

- Optimizer: GradientDescentOptimizer=0, AdamOptimizer=1, AdagradOptimizer=2, AdadeltaOptimizer=3)

"""
import tensorflow as tf
from preprocessing import TimeSeries
from utils import MathHelper, GraphUtil, IOHelper
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

class Model(object):

    def __init__(self, dataset_original=None, list_idx=(1000,2000,0), epoch=100, batch_size=32, sliding=2,
                 method_statistic=0, output_index = 0,
                 num_hidden=8, learning_rate=0.15, couple_act=(0, 0), optimizer=0, pathsave=None, fig_id=None):
        self.dataset_original = dataset_original
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.method_statistic = method_statistic
        self.output_index = output_index
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        self.pathsave = pathsave
        self.fig_id = fig_id
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        if list_idx[2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(list_idx[0] + (list_idx[1] - list_idx[0]) / 2)

        if couple_act[0] == 0:
            self.activation1 = tf.nn.elu
        elif couple_act[0] == 1:
            self.activation1 = tf.nn.relu
        elif couple_act[0] == 2:
            self.activation1 = tf.nn.tanh
        else:
            self.activation1 = tf.nn.sigmoid

        if couple_act[1] == 0:
            self.activation2 = tf.nn.elu
        elif couple_act[1] == 1:
            self.activation2 = tf.nn.relu
        elif couple_act[1] == 2:
            self.activation2 = tf.nn.tanh
        else:
            self.activation2 = tf.nn.sigmoid

        if optimizer == 0:
            self.optimizer = tf.train.GradientDescentOptimizer
        elif optimizer == 1:
            self.optimizer = tf.train.AdamOptimizer
        elif optimizer == 2:
            self.optimizer = tf.train.AdagradDAOptimizer
        else:
            self.optimizer = tf.train.AdadeltaOptimizer

        self.filename = 'Epo=' + str(epoch) + '_Bs=' + str(batch_size) + '_Slid=' + str(sliding) + '_Lr=' + str(learning_rate) + '_Op=' + str(optimizer) \
        + '_Act1=' + str(couple_act[0]) + '_Act2=' + str(couple_act[1])


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
        tf.reset_default_graph()

        X_train, y_train = self.X_train, self.y_train

        X_size = X_train.shape[1]
        h_size = self.num_hidden
        y_size = y_train.shape[1]

        # Placeholders for input and output data
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')

        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal([X_size, h_size], stddev=0.03, dtype=tf.float64), name="W1")
        b1 = tf.Variable(tf.random_normal([h_size], dtype=tf.float64), name="b1")
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03, dtype=tf.float64), name='W2')
        b2 = tf.Variable(tf.random_normal([y_size], dtype=tf.float64), name='b2')

        # calculate the output of the hidden layer
        hidden_out = self.activation1(tf.add(tf.matmul(X, W1), b1))
        # Forward propagation # now calculate the hidden layer output
        y_ = self.activation2(tf.add(tf.matmul(hidden_out, W2), b2))

        # Loss function
        deltas = tf.square(y_ - y)
        loss = tf.reduce_mean(deltas)

        # Backward propagation
        opt = self.optimizer(learning_rate=self.learning_rate)
        train = opt.minimize(loss)

        # Initialize variables and run session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Go through num_iters iterations
        loss_plot = []
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

        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')

        W1 = tf.Variable(self.w1)
        b1 = tf.Variable(self.b1)
        W2 = tf.Variable(self.w2)
        b2 = tf.Variable(self.b2)

        hidden_out = self.activation1(tf.add(tf.matmul(X, W1), b1))
        y_ = self.activation2(tf.add(tf.matmul(hidden_out, W2), b2))

        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})

            y_test_inverse = self.min_max_scaler.inverse_transform(y_test)
            y_pred_inverse = self.min_max_scaler.inverse_transform(y_est_np)

            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

            self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_est_np, testScoreRMSE, testScoreMAE
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse

            # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))

        # print("Predict done!!!")


    def draw_result(self):
        GraphUtil.draw_loss(self.fig_id, self.epoch, self.loss_train, "Loss on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.score_test_RMSE, self.score_test_MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)

    def fit(self):
        self.preprocessing_data()
        self.build_model_and_train()
        self.predict()
        self.draw_result()
        self.save_result()

