from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from utils.MathUtil import *
from utils.IOUtil import *
from utils.GraphUtil import draw_predict_with_error
from utils.ExpandUtil import ExpandData

class Model:
    def __init__(self, dataset_original=None, train_idx=None, test_idx=None, sliding=None, activation = None, expand_func = None,
                 epoch=None, learning_rate = None, batch_size = None, beta = None, test_name = None, path_save_result = None):
        self.data_original = dataset_original
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.data = dataset_original[:train_idx + test_idx + sliding + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.activation = activation
        self.epoch = epoch
        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.path_save_result = path_save_result
        self.test_name = test_name
        self.filename = "flnn-sliding_{0}-ex_func_{1}-act_func_{2}-epoch_{3}-lr_{4}-batch_{5}-beta_{6}".format(sliding,
            expand_func,activation, epoch, learning_rate, batch_size, beta)


    def preprocessing_data(self):
        data, train_idx, test_idx, sliding, expand_func = self.data, self.train_idx, self.test_idx, self.sliding, self.expand_func

        data_scale = self.scaler.fit_transform(data)
        data_transform = data_scale[:train_idx + test_idx, :]

        for i in range(1, sliding + 1):
            data_transform = np.concatenate((data_transform, data_scale[i:i + train_idx + test_idx, :]), axis=1)

        data_x_not_expanded = data_transform[:, :-1]
        data_y = data_transform[:, [-1]]

        expand_data_obj = ExpandData(data, train_idx, test_idx, sliding, expand_func=expand_func)
        data_expanded = expand_data_obj.process_data()

        data_X = np.concatenate((data_x_not_expanded, data_expanded), axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = data_X[:train_idx, :], data_X[train_idx:, :], data_y[:train_idx, :], data_y[ train_idx:, :]

    def activation_output(self, z):
        if self.activation == 0:
            a = z
        elif self.activation == 1:
            a = elu(z)
        elif self.activation == 2:
            a = relu(z)
        elif self.activation == 3:
            a = tanh(z)
        elif self.activation == 4:
            a = sigmoid(z)
        return a

    def activation_backward(self, a):
        if self.activation == 0:
            return 1
        elif self.activation == 1:
            return np.where(a < 0, a + 0.9, 1)
        elif self.activation == 2:
            return np.where(a < 0, 0, 1)
        elif self.activation == 3:
            return 1 - np.power(a, 2)
        elif self.activation == 4:
            return a * (1-a)


    def random_mini_batches(self, seed=0):
        X, Y = self.X_train.T, self.y_train.T
        mini_batch_size = self.batch_size

        m = X.shape[1]  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(
            math.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def init_parameters(self, d):
        return np.random.randn(d, 1), np.zeros((1, 1))

    def init_momentum_parameters(self, d):
        vdw = np.zeros((d, 1))
        vdb = np.zeros((1, 1))

        return vdw, vdb

    def train(self):
        self.preprocessing_data()

        d = self.X_train.shape[1]

        seed = 0

        w, b = self.init_parameters(d)

        vdw, vdb = self.init_momentum_parameters(d)

        for e in range(self.epoch):

            seed += 1

            mini_batches = self.random_mini_batches(seed=seed)

            total_error = 0

            for mini_batch in mini_batches:
                X_batch, y_batch = mini_batch

                X_batch = X_batch.T
                y_batch = y_batch.T

                m = X_batch.shape[0]

                # Feed Forward
                z = np.dot(X_batch, w) + b
                a = self.activation_output(z)

                total_error += mean_absolute_error(a, y_batch)

                # Backpropagation
                da = a - y_batch
                dz = da * self.activation_backward(a)

                db = 1. / m * np.sum(dz, axis=0, keepdims=True)
                dw = 1. / m * np.matmul(X_batch.T, dz)

                vdw = self.beta * vdw + (1 - self.beta) * dw
                vdb = self.beta * vdb + (1 - self.beta) * db

                # Update weights
                w -= self.lr * vdw
                b -= self.lr * vdb

            #print("> Epoch {0}: MAE {1}".format(e, total_error))

        z = np.dot(self.X_test, w) + b
        a = self.activation_output(z)

        self.pred_inverse = self.scaler.inverse_transform(a)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = round(mean_absolute_error(self.pred_inverse, self.real_inverse), 4)
        self.rmse = round(np.sqrt(mean_squared_error(self.pred_inverse, self.real_inverse)), 4)

        write_to_result_file(self.filename, self.rmse, self.mae, self.test_name, self.path_save_result)
        draw_predict_with_error(1, self.real_inverse, self.pred_inverse, self.rmse, self.mae, self.filename, self.path_save_result)
        save_result_to_csv(self.real_inverse, self.pred_inverse, self.filename, self.path_save_result)
