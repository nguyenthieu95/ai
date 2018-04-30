#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 02:37:52 2018

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
from math import exp, sqrt, ceil, sin, cos, pi, tan, log
import copy
from random import uniform, randint
from operator import itemgetter
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def selfx(x):
    return x
def one(x):
    return randint(-9, 9)*1.0

def sinpix(x):
    return sin(pi*x)
def cospix(x):
    return cos(pi*x)
def sin2pix(x):
    return sin(2*pi*x)
def cos2pix(x):
    return cos(2*pi*x)
def get_list_orthogonals():
    return [selfx, sinpix, cospix, sin2pix, cos2pix]


def c2x2(x):
    return 2 * x**2- 1
def c4x3(x):
    return 4 * x**3 - 3*x
def c8x3(x):
    return 8 * x**3 - 8 * x**2 + 1
def get_list_chebyshev_polynomials():
    return [selfx, one, c2x2, c4x3, c8x3]


def l3x2(x):
    return (3 * x**2 - 1) / 2 
def l5x3(x):
    return (5 * x**3 - 3*x) / 2
def l35x4(x):
    (5 * x**4 - 8*x**2 + 16*x - 3) / 8
def get_list_legendre_polynomials():
    return [selfx, one, l3x2, l5x3, l35x4]


def p2(x):
    return x**2
def p3(x):
    return x**3
def p4(x):
    return x**4
def get_list_power_series():
    return [selfx, one, p2, p3, p4]

def rp2(x):
    return randint(-29, 29) * x**2
def rp3(x):
    return randint(-19, 19) * x**3
def rp4(x):
    return randint(-9, 9) * x**4
def get_list_random_power_series():
    return [selfx, one, rp2, rp3, rp4]


def t2(x):
    return (cos(pi* x**2) + sin(pi * x) + x)
def t3(x):
    return tan(x)
def t4(x):
    return sin(pi*x)*cos(pi*x)
def get_list_t_orthogonals():
    return [selfx, one, t2, t3, t4]

def th2(x):
    return sin(pi* (cos(x**2 + log(x)))**(-1) )
def th3(x):
    return (sin(pi*x) + cos(pi*x))
def th4(x):
    return cos(pi*log(sin(pi*x)))
def get_list_th_orthogonals():
    return [selfx, one, th2, th3, th4]


def thi2(x):
    return sin(pi*x / 2) 
def thi3(x):
    return cos(pi*x / 2)
def thi4(x):
    return sin(pi*x / 2) * cos(pi*x / 2)
def thi5(x):
    return sin(pi*x / 2) + cos(pi*x / 2)
def get_list_thi_orthogonals():
    return [selfx, thi2, thi3, thi4, thi5]


def thie2(x):
    return sin(pi*x / 4) * sin(pi*x / 2) 
def thie3(x):
    return cos(pi*x / 4) * cos(pi*x / 2)
def thie4(x):
    return sin(pi*x / 4) + sin(pi*x / 2)
def thie5(x):
    return cos(pi*x / 4) + cos(pi*x / 2)
def get_list_thie_orthogonals():
    return [selfx, thie2, thie3, thie4, thie5]


def get_list_functions_transform(index):
    if index == 2:
        return get_list_chebyshev_polynomials()
    elif index == 3:
        return get_list_legendre_polynomials()
    elif index == 4:
        return get_list_power_series()
    elif index == 5:
        return get_list_random_power_series()
    elif index == 6:
        return get_list_t_orthogonals()
    elif index == 7:
        return get_list_th_orthogonals()
    elif index == 8:
        return get_list_thi_orthogonals()
    elif index == 9:
        return get_list_thie_orthogonals()
    else:
        return get_list_orthogonals()
    

class Model(object):

    def __init__(self, dataset_original, list_idx, func_index, sliding, sti_level, pos_number, epoch, batch_size, learning_rate, method_statistic = 0):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.functions_expand_index = func_index
        self.sliding = sliding
        self.stimulation_level = sti_level
        self.positive_number = pos_number
        self.epoch = epoch
        self.batch_size = batch_size
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
        
    
    def transform_features_cluster(self):
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
        print("Transform features cluster done!!!")
        
        
    def transform_features_functional_link(self):
        list_label_func_expand = get_list_functions_transform(self.functions_expand_index)
        ## expand input by list functions expand
        dataset_X_train = []
        for val in self.S_train:
            data = [] 
            for dimension_index in range(0, self.dataset_original.shape[1]):
                for func in list_label_func_expand:
                    data.append(func(val[dimension_index]))
            dataset_X_train.append(data)
        self.S_train = (np.asarray(dataset_X_train)).astype(np.float64)
        
        dataset_X_test = []
        for val in self.S_test:
            data = [] 
            for dimension_index in range(0, self.dataset_original.shape[1]):
                for func in list_label_func_expand:
                    data.append(func(val[dimension_index]))
            dataset_X_test.append(data)
        self.S_test = (np.asarray(dataset_X_test)).astype(np.float64)
        print("Transform features functional link done!!!")
        
    
    def build_bpnn_and_train(self):
        ## Build layer's sizes
        X_train, y_train = self.S_train, self.y_train

        X_size = X_train.shape[1]   
        y_size = y_train.shape[1]
        ## Symbols
        X = tf.placeholder("float64", shape=[None, X_size])
        y = tf.placeholder("float64", shape=[None, y_size])
        
        W = tf.Variable(tf.random_normal([X_size, y_size], stddev=0.03, dtype=tf.float64), name="W")
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
        self.transform_features_cluster()
        self.transform_features_functional_link()
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



## Load data frame
#fullpath="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#fullpathsaveresult= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
fullpath = "/home/thieunv/Desktop/Link to LabThayMinh/code/data/GoogleTrace/"
fullpathsaveresult = "/home/thieunv/university/LabThayMinh/code/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
filename = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv" 

df = read_csv(fullpath + filename, header=None, index_col=False, usecols=[0], engine='python')    # (10)
dataset_original = df.values

list_function_expand = [4] #[1, 2, 4, 8, 9]                             # (1)           # (2) 0 dung, sigmoid, relu  
sliding_windows = [3] # [1, 2, 3, 5]                
stimulation_levels = [0.60]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.15] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]                           # (3)
epochs = [2000] #[250, 500, 1000, 1500, 2000]                        # (4)
batch_sizes = [32] #[8, 16, 32, 64, 128]                                                     # (5)                                                                              # (6)
learning_rates = [0.25] #[0.025, 0.05, 0.1, 0.15, 0.20, 0.25]                # (7)
list_num = [(3000, 4100)]



pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for list_idx in list_num:
    for sliding in sliding_windows:
        for func_index in list_function_expand:
            for sti_level in stimulation_levels:
                for pos_number in positive_numbers:
                    for epoch in epochs:
                        for batch_size in batch_sizes:
                            for learning_rate in learning_rates:
                           
                                febpnn = Model(dataset_original, list_idx, func_index, sliding, sti_level, pos_number, epoch, batch_size, learning_rate, method_statistic=2)
                                febpnn.fit()
                                
                                so_vong_lap += 1
                                if so_vong_lap % 5000 == 0:
                                    print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    







