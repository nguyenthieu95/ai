#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:51:06 2018

@author: thieunv
"""

def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Transform all dataset
        list_split = []     # ram, disk, cpu, 
        for i in range(self.dimension-1, -1, -1):
            list_split.append(dataset_original[:test_idx + sliding, i:i+1])
            
        list_transform = []     # ram, disk, cpu
        for i in range(self.dimension):
            list_transform.append(self.min_max_scaler.fit_transform(list_split[i]))
        
        ## Handle data with sliding
        if self.merge_data_transform == 0:
            """
            cpu(t), ram(t), cpu(t-1), ram(t-1),. ..
            """
            dataset_sliding = list_transform[self.dimension-1][:test_idx]      # CPU first
            
            for i in range(self.dimension - 2, -1, -1):  # remove CPU
                dataset_sliding = np.concatenate((dataset_sliding, list_transform[i][:test_idx]), axis=1)
                
            # Handle sliding 
            for i in range(sliding-1):
                for j in range(self.dimension-1, -1, -1):
                    d1 = np.array(list_transform[j][i+1: test_idx+i+1])
                    dataset_sliding = np.concatenate((dataset_sliding, d1), axis=1)
            print("done")
            
        if self.merge_data_transform == 1:
            """
            cpu(t), cpu(t-1), ..., ram(t), ram(t-1),... 
            """
            dataset_sliding = np.zeros(shape=(test_idx,1))
            
            for i in range(self.dimension-1, -1, -1):
                for j in range(sliding):
                    d1 = np.array(list_transform[i][j: test_idx+j])
                    dataset_sliding = np.concatenate((dataset_sliding, d1), axis=1)
            dataset_sliding = dataset_sliding[:, 1:]
            print("done")
            
        ## window value: x1 \ x2 \ x3  (dataset_sliding)
        ## Now we using different method on this window value 
        dataset_y = copy.deepcopy(list_transform[self.dimension-1][sliding:])      # Now we need to find dataset_X
        
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