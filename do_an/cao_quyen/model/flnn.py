from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.MathUtil import *
from utils.PreprocessingUtil import TimeSeries, MiniBatch
from utils.IOUtil import *
from utils.GraphUtil import draw_predict_with_error

class Model:
    def __init__(self, dataset_original=None, train_idx=None, valid_idx=None, test_idx=None, sliding=None, activation=None,
                 expand_func=None, epoch=None, learning_rate=None, batch_size=None, beta=None, test_name=None,
                 path_save_result=None, method_statistic = None, output_index=None, output_multi=None):
        self.data_original = dataset_original
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        if valid_idx == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(train_idx + (test_idx - train_idx) / 2)

        self.sliding = sliding
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.activation = activation
        self.epoch = epoch
        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.output_index = output_index
        self.method_statistic = method_statistic
        self.path_save_result = path_save_result
        self.test_name = test_name
        self.output_multi = output_multi
        self.filename = "flnn-sliding_{0}-ex_func_{1}-act_func_{2}-epoch_{3}-lr_{4}-batch_{5}-beta_{6}".format(
            sliding,expand_func, activation,epoch,learning_rate,batch_size,beta)

        if activation == 0:
            self.activation_function = itself
            self.activation_backward = derivative_self
        elif activation == 1:
            self.activation_function = elu
            self.activation_backward = derivative_elu
        elif activation == 2:
            self.activation_function = relu
            self.activation_backward = derivative_relu
        elif activation == 3:
            self.activation_function = tanh
            self.activation_backward = derivative_tanh
        elif activation == 4:
            self.activation_function = sigmoid
            self.activation_backward = derivative_sigmoid

    def preprocessing_data(self):
        timeseries = TimeSeries(self.expand_func, self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.data_original, self.scaler)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.scaler = timeseries.preprocessing(self.output_index)
        #print("Processing data done!!!")


    def train(self):
        self.preprocessing_data()

        number_input = self.X_train.shape[1]
        number_output = self.y_train.shape[1]

        ## init hyper and momentum parameters
        w, b = np.random.randn(number_input, number_output), np.zeros((1, number_output))
        vdw, vdb = np.zeros((number_input, number_output)), np.zeros((1, number_output))

        seed = 0
        for e in range(self.epoch):
            seed += 1
            mini_batches = MiniBatch(self.X_train, self.y_train, self.batch_size).random_mini_batches(seed=seed)

            total_error = 0
            for mini_batch in mini_batches:
                X_batch, y_batch = mini_batch

                X_batch = X_batch.T
                y_batch = y_batch.T

                m = X_batch.shape[0]

                # Feed Forward
                z = np.add(np.matmul(X_batch, w), b)
                a = self.activation_function(z)

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

            # print("> Epoch {0}: MAE {1}".format(e, total_error))

        z = np.dot(self.X_test, w) + b
        a = self.activation_function(z)

        self.pred_inverse = self.scaler.inverse_transform(a)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = np.round(mean_absolute_error(self.pred_inverse, self.real_inverse, multioutput='raw_values'), 4)
        self.rmse = np.round(np.sqrt(mean_squared_error(self.pred_inverse, self.real_inverse, multioutput='raw_values')), 4)

        #print(self.mae)
        #print(self.rmse)

        if self.output_multi:
            draw_predict_with_error(1, self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.rmse[0], self.mae[0], self.filename, self.path_save_result+"CPU-")
            draw_predict_with_error(2, self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.rmse[1], self.mae[1], self.filename, self.path_save_result+"RAM-")
            write_all_results([self.filename, self.rmse[0], self.rmse[1], self.mae[0], self.mae[1] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.filename, self.path_save_result+"CPU-")
            save_result_to_csv(self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.filename, self.path_save_result+"RAM-")
        else:
            draw_predict_with_error(1, self.real_inverse, self.pred_inverse, self.rmse[0], self.mae[0], self.filename, self.path_save_result)
            write_all_results([self.filename, self.rmse[0], self.mae[0] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse, self.pred_inverse, self.filename, self.path_save_result)



