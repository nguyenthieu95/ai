from utils.PreprocessingUtil import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.MathUtil import *
from utils.GraphUtil import draw_predict_with_error
from utils.IOUtil import *
from model import PSO

class Model:
    def __init__(self, dataset_original=None, train_idx=None, valid_idx=None, test_idx=None, train_valid_rate=None, sliding=None,
                 activation=None, expand_func=None, epoch=None, pop_size=None, c_couple=None, w_minmax=None,
                 lowup_w=(-1, 1), lowup_b=(-1, 1),
                 test_name=None, path_save_result=None, method_statistic = None, output_index=None, output_multi=None):
        self.data_original = dataset_original
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        if valid_idx == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(train_idx + (test_idx - train_idx) / 2)

        self.scaler = MinMaxScaler()

        self.sliding = sliding
        self.expand_func = expand_func
        self.activation = activation
        self.train_valid_rate = train_valid_rate
        self.epoch = epoch
        self.pop_size = pop_size
        self.c_couple = c_couple
        self.w_minmax = w_minmax
        self.lowup_w = lowup_w
        self.lowup_b = lowup_b
        self.output_index = output_index
        self.method_statistic = method_statistic
        self.path_save_result = path_save_result
        self.test_name = test_name
        self.output_multi = output_multi
        self.filename = "FL_PSONN-sliding_{0}-ex_func_{1}-act_func_{2}-epoch_{3}-pop_size_{4}-c1_{5}_c2_{6}-w1_{7}_w2_{8}".format(
            sliding, expand_func, activation, epoch, pop_size, c_couple[0], c_couple[1], w_minmax[0], w_minmax[1])

        if activation == 0:
            self.activation_function = itself
        elif activation == 1:
            self.activation_function = elu
        elif activation == 2:
            self.activation_function = relu
        elif activation == 3:
            self.activation_function = tanh
        elif activation == 4:
            self.activation_function = sigmoid

        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None, None, None
        self.individual, self.loss_train = None, None
        self.number_node_input, self.number_node_output, self.size_w = None, None, None

    def preprocessing_data(self):
        timeseries = TimeSeries(self.expand_func, self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.data_original, self.scaler)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.scaler = timeseries.preprocessing(self.output_index)
        print("Processing data done!!!")

    def predict(self):
        w = np.reshape(self.individual[:self.size_w], (self.number_node_input, self.number_node_output))
        b = np.reshape(self.individual[self.size_w:], (-1, self.number_node_output))
        y_pred = self.activation_function( np.add(np.matmul(self.X_test, w), b) )

        self.pred_inverse = self.scaler.inverse_transform(y_pred)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = np.round(mean_absolute_error(self.pred_inverse, self.real_inverse, multioutput='raw_values'), 4)
        self.rmse = np.round(np.sqrt(mean_squared_error(self.pred_inverse, self.real_inverse, multioutput='raw_values')), 4)

        if self.output_multi:
            write_all_results([self.filename, self.rmse[0], self.rmse[1], self.mae[0], self.mae[1] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.filename, self.path_save_result+"CPU-")
            save_result_to_csv(self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.filename, self.path_save_result+"RAM-")
            draw_predict_with_error(1, self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.rmse[0], self.mae[0], self.filename, self.path_save_result+"CPU-")
            draw_predict_with_error(2, self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.rmse[1], self.mae[1], self.filename, self.path_save_result+"RAM-")
        else:
            write_all_results([self.filename, self.rmse[0], self.mae[0] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse, self.pred_inverse, self.filename, self.path_save_result)
            draw_predict_with_error(1, self.real_inverse, self.pred_inverse, self.rmse[0], self.mae[0], self.filename, self.path_save_result)

    def run(self):
        self.preprocessing_data()

        self.number_node_input = self.X_train.shape[1]
        self.number_node_output = self.y_train.shape[1]
        self.size_w = self.number_node_input * self.number_node_output

        pso_para = {
            "epoch": self.epoch, "pop_size": self.pop_size, "c_couple": self.c_couple, "w_minmax": self.w_minmax,
            "lowup_w": self.lowup_w, "lowup_b": self.lowup_b
        }
        other_para = {
            "number_node_input": self.number_node_input, "number_node_output": self.number_node_output,
            "X_train": self.X_train, "y_train": self.y_train,
            "X_valid": self.X_valid, "y_valid": self.y_valid, "train_valid_rate": self.train_valid_rate,
            "activation": self.activation_function
        }
        pso = PSO.BaseClass(other_para, pso_para)
        self.individual, self.loss_train = pso.train()
        self.predict()

