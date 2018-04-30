#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 02:04:24 2018

@author: thieunv
- Cac tham so co the thay doi: 
    + (1): so dung cac function khac nhau: 0, 1, 2, 3, 4
    + (2): activation function: 1-sigmoid, 2-relu, else-ko dung 
    + (3): sliding window. 1, 2, 4, ...
    + (4): epoch : 100, 200, 500, 1000, ...
    + (5): batch_size: 8, 16, 32, 64, ...
    + (6): validation: 0.1, 0.2, 0.25, ... < 0.5
    + (7): learning rate: 0.01, 0.02, 0.03, ...
    + (8): epsilon: 0.001, 0.002, 0.005, 0,01...
    + (9): training size
    + (10): 0,1-CPU,RAM, 2-Disk-ioTime, 3-Disk-space
"""


import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from math import exp, sqrt, ceil
import copy
from random import randint, uniform
from operator import itemgetter
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

## import helper function
from flnn_functions import get_batch_data_next, my_invert_min_max_scaler, my_min_max_scaler
## get list functions expand
import flnn_functions as gen_fun


## Load data frame
#fullpath="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#fullpathsaveresult= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
fullpath = "/home/thieunv/Desktop/Link to LabThayMinh/code/data/GoogleTrace/"
fullpathsaveresult = "/home/thieunv/university/LabThayMinh/code/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
filename = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv" 

df = read_csv(fullpath + filename, header=None, index_col=False, usecols=[0], engine='python')    # (10)
dataset_original = df.values

list_function_expand = [1] #[1, 2, 4, 8, 9]                             # (1)
list_activation = [2] #[0, 1, 2]                                                 # (2) 0 dung, sigmoid, relu  
sliding_windows = [3] # [1, 2, 3, 5]                                           # (3)

length = len(dataset_original)
num_features = df.shape[1]

train_size = 2880                                                                       # (9)
test_size = length - train_size
epochs = [2000] #[250, 500, 1000, 1500, 2000]                        # (4)
batch_sizes = [32] #[8, 16, 32, 64, 128]                                                     # (5)

valid = 0.25                                                                                # (6)
learning_rates = [0.25] #[0.025, 0.05, 0.1, 0.15, 0.20, 0.25]                # (7)
epsilons = [0.00001]# [0.00001, 0.00005, 0.0001, 0.0005]                     # (8)



class Model(object):
    def __init__(self, dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, method_statistic = 0):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        self.method_statistic = method_statistic
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Transform all dataset
        dataset_split = dataset_original[:test_idx + sliding]               # Save orginal data to take y_test
        dataset_transform = self.min_max_scaler.fit_transform(dataset_split)
        ## Handle data with sliding
        dataset_sliding = dataset_transform[:len(dataset_transform)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_transform[i+1: len(dataset_transform)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        ## window value: x1 \ x2 \ x3  (dataset_sliding)
        ## Now we using different method on this window value 
        dataset_y = copy.deepcopy(dataset_transform[sliding:])      # Now we need to find dataset_X
        if self.method_statistic == 0:
            dataset_X = copy.deepcopy(dataset_sliding)

        if self.method_statistic == 1:
            """
            mean(x1, x2, x3, ...)
            """
            dataset_X = np.reshape(np.mean(dataset_sliding, axis = 1), (-1, 1))
        if self.method_statistic == 2:
            """
            min(x1, x2, x3, ...), mean(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding, axis = 1), (-1, 1))
            mean_X = np.reshape(np.mean(dataset_sliding, axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding, axis = 1), (-1, 1))
            dataset_X = np.concatenate( (min_X, mean_X, max_X), axis=1 )
        if self.method_statistic == 3:
            """
            min(x1, x2, x3, ...), median(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding, axis = 1), (-1, 1))
            median_X = np.reshape(np.median(dataset_sliding, axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding, axis = 1), (-1, 1))
            dataset_X = np.concatenate( (min_X, median_X, max_X), axis=1 )
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
        self.X_test, self.y_test = dataset_X[train_idx:], dataset_y[train_idx:]
        print("Processing data done!!!")
    
    
    def clustering_data(self):
        train_X = copy.deepcopy(self.X_train)
        stimulation_level, positive_number = self.stimulation_level, self.positive_number
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        hu1 = [0, Model.get_random_input_vector(train_X)]   # hidden unit 1 (t1, wH)
        list_hu = [copy.deepcopy(hu1)]         # list hidden units
        matrix_Wih = copy.deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])     # Mang 2 chieu 
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(train_X):
            list_dist_mj = []      # Danh sach cac dist(mj)
             # number of hidden units
            for j in range(0, len(list_hu)):                # j: la chi so cua hidden thu j
                dist_sum = 0.0
                for i in range(0, len(train_X[0])):        # i: la chi so cua input unit thu i
                    dist_sum += pow(train_X[m][i] - matrix_Wih[j][i], 2.0)
                list_dist_mj.append([j, sqrt(dist_sum)])
            list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
            
            c = list_dist_mj[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj[0][1] # distmc: Gia tri khoang cach nho nhat
            
            if distmc < stimulation_level:
                list_hu[c][0] += 1                  # update hidden unit cth
                
                # Find Neighbourhood
                list_distjc = []
                for i in range(0, len(matrix_Wih)):
                    list_distjc.append([i, Model.distance_func(matrix_Wih[c], matrix_Wih[i])])
                list_distjc = sorted(list_distjc, key=itemgetter(1))
                
                # Update BMU (Best matching unit and it's neighbourhood)
                neighbourhood_node = int( 1 + ceil( 0.2 * (len(list_hu) - 1) ) )
                for i in range(0, neighbourhood_node ):
                    if i == 0:
                        list_hu[c][1] += (positive_number * distmc) * (train_X[m] - list_hu[c][1])
                        matrix_Wih[c] += (positive_number * distmc) * (train_X[m] - list_hu[c][1])
                    else:
                        c_temp = list_distjc[i][0]
                        distjc = list_distjc[i][1]
                        hic = exp(-distjc * distjc)
                        delta = (positive_number * hic) * (train_X[m] - list_hu[c_temp][1])
                        
                        list_hu[c_temp][1] += delta
                        matrix_Wih[c_temp] += delta
                
                # Tiep tuc vs cac example khac
                m += 1
                if m % 1000 == 0:
                    print "distmc = {0}".format(distmc)
                    print "m = {0}".format(m)
            else:
                print "Failed !!!. distmc = {0}".format(distmc)
                list_hu.append([0, copy.deepcopy(train_X[m]) ])
                print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
                matrix_Wih = np.append(matrix_Wih, [copy.deepcopy(train_X[m])], axis = 0)
                for hu in list_hu:
                    hu[0] = 0
                # then go to step 1
                m = 0
                ### +++
        ### +++ Get the last matrix weight 
        self.matrix_Wih = copy.deepcopy(matrix_Wih)
        self.list_hu = copy.deepcopy(list_hu)
        print("Encoder features done!!!")
        
        
    def mutation_hidden_node(self):
        self.threshold_number = int (len(self.X_train) / len(self.list_hu))
        ### Qua trinh mutated hidden unit (Pha 2- Adding artificial local data)
        # Adding 2 hidden unit in begining and ending points of input space
        t1 = np.zeros(self.sliding)
        t2 = np.ones(self.sliding)
        self.list_hu.append([0, t1])
        self.list_hu.append([0, t2])
        
        self.matrix_Wih = np.concatenate((self.matrix_Wih, np.array([t1])), axis=0)
        self.matrix_Wih = np.concatenate((self.matrix_Wih, np.array([t2])), axis=0)
    
    #    # Sort matrix weights input and hidden, Sort list hidden unit by list weights
        for i in range(0, self.matrix_Wih.shape[1]):
            self.matrix_Wih = sorted(self.matrix_Wih, key=lambda elem_list: elem_list[i])
            self.list_hu = sorted(self.list_hu, key=lambda elem_list: elem_list[1][i])
             
            for i in range(len(self.list_hu) - 1):
                ta, wHa = self.list_hu[i][0], self.list_hu[i][1]
                tb, wHb = self.list_hu[i+1][0], self.list_hu[i+1][1]
                dab = Model.distance_func(wHa, wHb)
                
                if dab > self.distance_level and ta < self.threshold_number and tb < self.threshold_number:
                    # Create new mutated hidden unit (Dot Bien)
                    temp_node = Model.get_mutate_vector_weight(wHa, wHb, mutation_id=1)
                    self.list_hu.insert(i+1, [0, copy.deepcopy(temp_node)])
                    self.matrix_Wih = np.insert(self.matrix_Wih, [i+1], temp_node, axis=0)
                    print "New hidden unit created. {0}".format(len(self.matrix_Wih))
        print("Finished mutation hidden unit!!!")
    
    
    def transform_features(self):
        temp1 = []
        for i in range(0, len(self.X_train)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_train[i])))
            temp1.append(np.array(Sih))
        temp2 = []
        for i in range(0, len(self.X_test)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_test[i])))
            temp2.append(np.array(Sih))
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
        print("Transform features done!!!")
        
    
    def build_bpnn_and_train(self):
        ## Build layer's sizes
        X_train, y_train = self.S_train, self.y_train

        X_size = X_train.shape[1]   
        h_size = len(self.list_hu)
        y_size = y_train.shape[1]
        ## Symbols
        X = tf.placeholder("float64", shape=[None, X_size])
        y = tf.placeholder("float64", shape=[None, y_size])
        
        W = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03, dtype=tf.float64), name="W")
        b = tf.Variable(tf.random_normal([y_size], dtype=tf.float64), name="b")
      
        y_ = tf.nn.elu(tf.add(tf.matmul(X, W), b))
        # Backward propagation
        cost    = tf.reduce_mean( tf.square(y_ - y) )
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
                    batch_x, batch_y = Model.get_batch_data_next(X_train, y_train, i, batch_size)
                    if len(batch_x) == 0:
                        break
                    sess.run(updates, feed_dict={X: batch_x, y: batch_y})
                
                loss_epoch = sess.run(cost, feed_dict={X: X_train, y: y_train})
                weight, bias = sess.run([W, b])
                loss_plot.append(loss_epoch)
                print("Epoch: {}".format(epoch + 1), "cost = {}".format(loss_epoch))
                
        self.weight, self.bias, self.loss_train = weight, bias, loss_plot
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
     
        y_ = tf.nn.elu(tf.add(tf.matmul(X, W), b))
        
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
            
            print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        
        print("Predict done!!!")

    def draw_loss(self):
        plt.figure(1)
        plt.plot(range(self.epoch), self.loss_train, label="Loss on training per epoch")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Loss', fontsize=12)  
        
        
    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.y_test_inverse)
        plt.plot(self.y_pred_inverse)
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
    
    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        self.mutation_hidden_node()
        self.transform_features()
        self.build_bpnn_and_train()
        self.predict()
        self.draw_loss()
        self.draw_predict()
    
    @staticmethod
    def distance_func(a, b):
        return distance.euclidean(a, b)
    
    @staticmethod
    def sigmoid_activation(x):
        return 1.0 / (1.0 + exp(-x))
    
    @staticmethod
    def get_random_input_vector(train_X):
        return copy.deepcopy(train_X[randint(0, len(train_X)-1)])
    
    @staticmethod
    def get_batch_data_next(trainX, trainY, index, batch_size):
        real_index = index*batch_size
        if (len(trainX) % batch_size != 0 and index == (len(trainX)/batch_size +1) ):
            return (trainX[real_index:], trainY[real_index:])
        elif (real_index == len(trainX)):
            return ([], [])
        else:
            return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])

    @staticmethod
    def get_mutate_vector_weight(wHa, wHb, mutation_id = 1):
        temp = []
        if mutation_id == 1:    # Lay trung binh cong
            for i in range(len(wHa)):
                temp.append( (wHa[i] + wHb[i]) / 2 )
        if mutation_id == 2:    # Lay uniform
            for i in range(len(wHa)):
                temp.append(uniform(wHa[i], wHb[i]))
        return np.array(temp)
    


#### My model here
def myFLNN(trainX, trainY, testX, epoch, batch_size, validation, activation, weight_size, epsilon=0.001, eta = 0.01):

    training_detail_file_name = fullpathsaveresult + 'FE=' + str(func_index) + '_Acti=' + str(act_index) + '_Sliding=' + str(sliding) + '_Epoch=' + str(epoch) + '_BatchS=' + str(batch_size) + '_Eta=' + str(eta) + '_Epsilon=' + str(epsilon) +'_NetworkDetail.txt'
    
    print "epoch: {0}, batch_size: {1}, learning-rate: {2}".format(epoch, batch_size, eta)
    ## 1. Init weight [-0.5, +0.5]
    w = np.random.rand(weight_size, 1) - 0.5
    print "init weight: "
    print w
                    
    
    ## 2. Split validation dataset
    valid_len = int(len(trainX)*validation)
    valid_X = trainX[(len(trainX) - valid_len):]
    valid_y = trainY[(len(trainX) - valid_len):]
    train_X = trainX[0:(len(trainX) - valid_len)]
    train_y = trainY[0:(len(trainX) - valid_len)]
    
    valid_list_loss = [0.1]
    train_list_loss = [0.1]
    
    ## 3. Loop in epochs
    for i in range(epoch):
        if (i % 50 == 0):
            print "Epoch thu: {0}".format(i)
        
        ## 3.1 Update w after 1 batch
        num_loop = int(len(trainX) / batch_size)
        for ind in range(num_loop):
            
            ## 3.1.1 Get next batch
            X_train_next, y_train_next = get_batch_data_next(train_X, train_y, ind, batch_size)
            if (len(X_train_next) == 0):
                break
            
            ## 3.1.2 Calculate all delta weight in 1 batch 
            delta_ws = []
            for j in range(len(X_train_next)):
                y_output = np.dot(X_train_next[j], w)
                y_output = activation_func(y_output)
                ek = y_train_next[j] - y_output
                delta_w = eta * ek * X_train_next[j]
                
                delta_ws.append(delta_w)
            
            ## 3.1.3 Sum all delta weight to get mean delta weight
            delta_wbar = np.array(np.sum(delta_ws, axis = 0) / len(X_train_next))
            
            ## 3.1.4 Change new weight with delta_wbar (mean delta weight)
            w += np.reshape(delta_wbar, (weight_size, 1))
        
        ## 3.2 Calculate validation loss after each epoch train to adjust hyperparameter
        
        # Calculate validation error
        valid_predict = []
        for j in range(len(valid_X)):
            valid_ybar = np.dot(valid_X[j], w)
            valid_ybar = activation_func(valid_ybar)
            valid_predict.append(valid_ybar)
        valid_MSE = mean_squared_error(valid_y, np.array(valid_predict))
        
        # Calculate train error
        train_predict = []
        for j in range(len(train_X)): 
            train_ybar = np.dot(train_X[j], w)
            train_ybar = activation_func(valid_ybar)
            train_predict.append(train_ybar)
        train_MSE = mean_squared_error(train_y, np.array(train_predict))
        train_list_loss.append(train_MSE)
        
        # Avoiding overfitting
        if (valid_MSE > (valid_list_loss[-1] + epsilon)):
            print "Break epoch to finish training."
            print w
                
            break
        valid_list_loss.append(valid_MSE)
        if (i % 50 == 0):
                print w

    print "Final weight: "
    print w
  
    ## 4. Predict test dataset
    predict = []
    for i in range(len(testX)):
        ybar = np.dot(testX[i], w)
        ybar = activation_func(ybar)
        predict.append(ybar)
     
    return (w, np.reshape(predict, (len(predict), 1)), valid_list_loss, train_list_loss, training_detail_file_name)


pl1 = 1
pl2 = 100000
### Loop to test 5 way expand
for func_index in list_function_expand:       # 5 way to expand 
    
    ### Loop to test 3 way use activation function (sigmoid, relu, 0 dung)
    for act_index in list_activation:
        
        list_label_func_expand = gen_fun.get_list_function(func_index)       #(1)
        activation_func = gen_fun.get_activation_func(act_index)            #(9)
        
        ## expand input by list functions expand
        dataset = []
        for val in dataset_original:
            data = []
            for dimen_index in range(dataset_original.shape[1]):
                for func in list_label_func_expand:
                    data.append(func(val[dimen_index]))
            dataset.append(data)
        dataset = (np.asarray(dataset)).astype(np.float64)
        
        # normalize the dataset
        GoogleTrace_orin_unnormal = dataset[:, 0].reshape(-1, 1)    # keep orginal data to test
        min_GT = min(GoogleTrace_orin_unnormal)
        max_GT = max(GoogleTrace_orin_unnormal)
        
        ## Scaling min max
        data_scaler = []
        for i in range( dataset.shape[1] ):
            data_scaler.append( my_min_max_scaler(dataset[:, i].reshape(-1, 1)) ) 

        for sliding in sliding_windows:
            
            num_expand = len(list_label_func_expand)
            weight_size = num_features*num_expand*sliding
            
            for epoch in epochs: 
                
                for batch_size in batch_sizes:
                    
                    for learning_rate in learning_rates:
                        
                        for epsilon in epsilons:
            
                            
                            data = []
                            for i in range(length-sliding):
                                detail=[]
                                for ds in data_scaler:
                                    for j in range(sliding):
                                        detail.append(ds[i+j])
                                data.append(detail)
                            data = np.reshape(np.array(data), (len(data), weight_size) )
                            
                            trainX, trainY = data[0:train_size], data[sliding:train_size+sliding, 0:1]
                            testX = data[train_size:length-sliding]
                            testY =  GoogleTrace_orin_unnormal[train_size+sliding:length]
                            #transformTestY = data[train_size:length-sliding, 0:1]
                            
                                
                            w, predict, valid_list_loss, train_list_loss, training_detail_file_name = myFLNN(trainX, trainY, testX, epoch=epoch, batch_size = batch_size, 
                                                validation=valid, activation=activation_func, weight_size=weight_size, epsilon=epsilon, eta=learning_rate)
                            
                            # invert predictions        
                            testPredictInverse = my_invert_min_max_scaler(predict, min_GT, max_GT)
#                            print 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
                            
                            # calculate root mean squared error
                            testScoreRMSE = sqrt(mean_squared_error(testY, testPredictInverse))
                            testScoreMAE = mean_absolute_error(testY, testPredictInverse)
#                            print('Test Score: %f RMSE' % (testScoreRMSE))
#                            print('Test Score: %f MAE' % (testScoreMAE))
                            
#                            if testScoreMAE < 1.0:
                            # summarize history for point prediction
                            plt.figure(pl1)
                            plt.plot(testY)
                            plt.plot(testPredictInverse)
                            
                            plt.title('model predict')
                            plt.ylabel('real value')
                            plt.xlabel('point')
                            plt.legend(['realY...Test Score RMSE= ' + str(testScoreRMSE), 'predictY...Test Score MAE= ' + str(testScoreMAE) ], loc='upper right')
#                                pic1_fn = full_path + 'pointPredict_FE=' + str(func_index) + '_Acti=' + str(act_index) + '_Sliding=' + str(sliding) + '_Epoch=' + str(epoch) + '_BatchS=' + str(batch_size) + '_Eta=' + str(learning_rate) + '_Epsilon=' + str(epsilon) +'.png'
#                                plt.savefig(pic1_fn)
#                                plt.close()
                            
                            plt.figure(pl2)
                            plt.plot(valid_list_loss)
                            plt.plot(train_list_loss)
                            
                            plt.title('model loss')
                            plt.ylabel('real value')
                            plt.xlabel('epoch')
                            plt.legend(['validation error...Test Score RMSE= ' + str(testScoreRMSE), 'train error...Test Score MAE= ' + str(testScoreMAE) ], loc='upper right')
#                                pic2_fn = full_path + 'validAndTrainLoss_FE=' + str(func_index) + '_Acti=' + str(act_index) + '_Sliding=' + str(sliding) + '_Epoch=' + str(epoch) + '_BatchS=' + str(batch_size) + '_Eta=' + str(learning_rate) + '_Epsilon=' + str(epsilon) +'.png'
#                                plt.savefig(pic2_fn)
#                                plt.close()
                            
                            pl1 += 1
                            pl2 += 1

print "Processing DONE!!!"

