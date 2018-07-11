import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ga_flnn.population import Population
from utils.GraphUtil import draw_predict_with_error
from utils.IOUtil import save_result_to_csv, write_to_result_file
from utils.ExpandUtil import ExpandData


class Model:
    def __init__(self, data_original, train_idx=None, test_idx=None, sliding=None, expand_func = None, epoch=None,
                 pop_size = None, pc = None, pm = None, activation = None, test_name = None, path_save_result = None):
        self.data_original = data_original
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.data = data_original[:train_idx + test_idx + sliding + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.activation = activation
        self.path_save_result = path_save_result
        self.test_name = test_name
        self.filename = "GA_FLNN-_sliding_{0}-ex_func_{1}-act_func_{2}-epoch_{3}-pop_size_{4}-pc_{5}-pm_{6}".format(sliding,
            expand_func, activation, epoch, pop_size, pc, pm)
    
    def preprocessing_data(self):
        data, train_idx, test_idx, sliding, expand_func = self.data, self.train_idx, self.test_idx, self.sliding, self.expand_func

        data_scale = self.scaler.fit_transform(data)
        data_transform = data_scale[:train_idx + test_idx, :]

        for i in range(1, sliding+1):
            data_transform = np.concatenate((data_transform, data_scale[i:i+train_idx + test_idx, :]), axis = 1)

        data_x_not_expanded = data_transform[:, :-1]
        data_y = data_transform[:, [-1]]

        expand_data_obj = ExpandData(data, train_idx, test_idx, sliding, expand_func = expand_func)
        data_expanded = expand_data_obj.process_data()

        data_X = np.concatenate((data_x_not_expanded, data_expanded), axis = 1)

        self.X_train, self.X_test, self.y_train, self.y_test = data_X[:train_idx, :], data_X[train_idx:, :], data_y[:train_idx, :], data_y[train_idx:, :]

    def train(self):
        self.preprocessing_data()

        p = Population(self.pop_size, self.pc, self.pm, activation = self.activation)

        best = p.train(self.X_train, self.y_train, epochs=self.epoch)

        pred = best.predict(self.X_test) 

        self.pred_inverse = self.scaler.inverse_transform(pred)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = round(mean_absolute_error(self.real_inverse, self.pred_inverse), 4)
        self.rmse = round(np.sqrt(mean_squared_error(self.real_inverse, self.pred_inverse)), 4)

        write_to_result_file(self.filename, self.rmse, self.mae, self.test_name, self.path_save_result)
        draw_predict_with_error(2, self.real_inverse, self.pred_inverse, self.rmse, self.mae, self.filename, self.path_save_result)
        save_result_to_csv(self.real_inverse, self.pred_inverse, self.filename, self.path_save_result)

