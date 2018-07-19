from utils.PreprocessingUtil import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.MathUtil import *
from utils.GraphUtil import draw_predict_with_error
from utils.IOUtil import *
from model import ABC


class Model:
    def __init__(self, dataset_original=None, train_idx=None, valid_idx=None, test_idx=None, train_valid_rate=None,
                 sliding=None, activation=None, expand_func=None, max_gens=None, num_bees=None, couple_num_bees=None,
                 patch_variables=None, sites=(3, 1), lowup_w=(-1, 1), lowup_b=(-1, 1),
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
        self.max_gens = max_gens
        self.num_bees = num_bees
        self.couple_num_bees = couple_num_bees
        self.patch_variables = patch_variables
        self.sites = sites
        self.lowup_w = lowup_w
        self.lowup_b = lowup_b
        self.output_index = output_index
        self.method_statistic = method_statistic
        self.path_save_result = path_save_result
        self.test_name = test_name
        self.output_multi = output_multi
        self.filename = "FL_ABCNN-sliding_{0}-ex_func_{1}-act_func_{2}-max_gens_{3}-num_bees_{4}-e_bees_{5}_o_bees_{6}-train_rate_{7}".format(
            sliding, expand_func, activation, max_gens, num_bees, couple_num_bees[0], couple_num_bees[1], train_valid_rate[0])


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
            draw_predict_with_error(1, self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.rmse[0], self.mae[0], self.filename, self.path_save_result+"CPU-")
            draw_predict_with_error(2, self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.rmse[1], self.mae[1], self.filename, self.path_save_result+"RAM-")
            write_all_results([self.filename, self.rmse[0], self.rmse[1], self.mae[0], self.mae[1] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.filename, self.path_save_result+"CPU-")
            save_result_to_csv(self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.filename, self.path_save_result+"RAM-")
        else:
            draw_predict_with_error(1, self.real_inverse, self.pred_inverse, self.rmse[0], self.mae[0], self.filename, self.path_save_result)
            write_all_results([self.filename, self.rmse[0], self.mae[0] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse, self.pred_inverse, self.filename, self.path_save_result)

    def run(self):
        self.preprocessing_data()

        self.number_node_input = self.X_train.shape[1]
        self.number_node_output = self.y_train.shape[1]
        self.size_w = self.number_node_input * self.number_node_output

        bee_para = {
            "max_gens": self.max_gens, "num_bees": self.num_bees, "couple_num_bees": self.couple_num_bees,
            "patch_variables": self.patch_variables, "sites": self.sites,
            "lowup_w": self.lowup_w, "lowup_b": self.lowup_b
        }
        other_para = {
            "number_node_input": self.number_node_input, "number_node_output": self.number_node_output,
            "X_train": self.X_train, "y_train": self.y_train,
            "X_valid": self.X_valid, "y_valid": self.y_valid, "train_valid_rate": self.train_valid_rate,
            "activation": self.activation_function
        }
        abc = ABC.BaseClass(other_para, bee_para)
        self.individual, self.loss_train = abc.train()
        self.predict()

