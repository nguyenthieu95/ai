import numpy as np
from copy import deepcopy

class TimeSeries(object):
    def __init__(self, train_idx=None, valid_idx=0, test_idx=None, sliding=None, method_statistic=0, data=None, minmax_scaler=None):
        """
        :param train_idx:
        :param valid_idx:
        :param test_idx:
        :param sliding:
        :param method_statistic:
        :param data:
        :param minmax_scaler:
        """
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.method_statistic = method_statistic
        self.data = data
        self.minmax_scaler = minmax_scaler

    def get_dataset_X(self, dimension=None, list_transform=None, sliding=None, test_idx=None, method_statistic=None):
        ## Handle data with sliding
        dataset_sliding = np.zeros(shape=(test_idx, 1))  # 0 | x1| x2 | x3| t1|t2|t3
        for i in range(dimension):
            for j in range(sliding):
                temp = np.array(list_transform[j: test_idx + j, i:i + 1])
                dataset_sliding = np.concatenate((dataset_sliding, temp), axis=1)
        dataset_sliding = dataset_sliding[:, 1:]

        ## window value: x1 \ x2 \ x3  (dataset_sliding)
        ## Now we using different method on this window value

        if method_statistic == 0:
            dataset_X = deepcopy(dataset_sliding)
        else:
            dataset_X = np.zeros(shape=(test_idx, 1))
            if method_statistic == 1:
                """
                mean(x1, x2, x3, ...), mean(t1, t2, t3,...) 
                """
                for i in range(dimension):
                    meanx = np.reshape(np.mean(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    dataset_X = np.concatenate((dataset_X, meanx), axis=1)

            if method_statistic == 2:
                """
                min(x1, x2, x3, ...), mean(x1, x2, x3, ...), max(x1, x2, x3, ....)
                """
                for i in range(dimension):
                    minx = np.reshape(np.amin(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    meanx = np.reshape(np.mean(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    maxx = np.reshape(np.amax(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    dataset_X = np.concatenate((dataset_X, minx, meanx, maxx), axis=1)

            if method_statistic == 3:
                """
                min(x1, x2, x3, ...), median(x1, x2, x3, ...), max(x1, x2, x3, ....), min(t1, t2, t3, ...), median(t1, t2, t3, ...), max(t1, t2, t3, ....)
                """
                for i in range(dimension):
                    minx = np.reshape(np.amin(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    medix = np.reshape(np.median(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    maxx = np.reshape(np.amax(dataset_sliding[:, i * sliding:(i + 1) * sliding], axis=1), (-1, 1))
                    dataset_X = np.concatenate((dataset_X, minx, medix, maxx), axis=1)
            dataset_X = dataset_X[:, 1:]
        return dataset_X


    def net_multiple_output(self):
        """
            valid_idx = 0 ==> No validate data ||  cpu(t), cpu(t-1), ..., ram(t), ram(t-1),...
        """
        train_idx, valid_idx, test_idx = self.train_idx, self.valid_idx, self.test_idx
        sliding, method_statistic = self.sliding, self.method_statistic
        data, minmax_scaler = self.data[:test_idx+sliding], self.minmax_scaler

        dimension = data.shape[1]
        list_transform = minmax_scaler.fit_transform(data)
        #    print(preprocessing.MinMaxScaler().data_max_)

        dataset_y = deepcopy(list_transform[sliding:])  # Now we need to find dataset_X
        dataset_X = self.get_dataset_X(dimension, list_transform, sliding, test_idx, method_statistic)

        ## Split data to set train and set test
        if valid_idx == 0:
            X_train, y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
            X_test, y_test = dataset_X[train_idx:test_idx], dataset_y[train_idx:test_idx]
            # print("Processing data done!!!")
            return X_train, y_train, X_test, y_test, minmax_scaler
        else:
            X_train, y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
            X_valid, y_valid = dataset_X[train_idx:valid_idx], dataset_y[train_idx:valid_idx]
            X_test, y_test = dataset_X[valid_idx:test_idx], dataset_y[valid_idx:test_idx]
            # print("Processing data done!!!")
            return X_train, y_train, X_valid, y_valid, X_test, y_test, minmax_scaler


    def net_single_output(self, output_index=0):
        """
            valid_idx = 0 ==> No validate data ||  cpu(t), cpu(t-1), ..., ram(t), ram(t-1),...
        """
        train_idx, valid_idx, test_idx = self.train_idx, self.valid_idx, self.test_idx
        sliding, method_statistic = self.sliding, self.method_statistic
        data, minmax_scaler = self.data[:test_idx+sliding], self.minmax_scaler

        dimension = data.shape[1]

        # output_index = 2 ==> 3, 0, 1, 2 ==> minmax_scaler chuan
        # cpu, ram, disk_io, disk_space
        list_transform = np.zeros(shape=(test_idx+sliding, 1))      # disk_space, cpu, ram, disk_io
        for i in range(0, dimension):
            t = output_index - (dimension-1) + i
            d1 = minmax_scaler.fit_transform(data[:test_idx+sliding, t].reshape(-1, 1))
            list_transform = np.concatenate((list_transform, d1), axis=1)
            # print(minmax_scaler.data_max_)
        list_transform = list_transform[:,1:]

        dataset_y = deepcopy((list_transform[sliding:, -1].reshape(-1, 1)))  # Now we need to find dataset_X
        dataset_X = self.get_dataset_X(dimension, list_transform, sliding, test_idx, method_statistic)

        ## Split data to set train and set test
        if valid_idx == 0:
            X_train, y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
            X_test, y_test = dataset_X[train_idx:test_idx], dataset_y[train_idx:test_idx]
            # print("Processing data done!!!")
            return X_train, y_train, X_test, y_test, minmax_scaler
        else:
            X_train, y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
            X_valid, y_valid = dataset_X[train_idx:valid_idx], dataset_y[train_idx:valid_idx]
            X_test, y_test = dataset_X[valid_idx:test_idx], dataset_y[valid_idx:test_idx]
            # print("Processing data done!!!")
            return X_train, y_train, X_valid, y_valid, X_test, y_test, minmax_scaler


