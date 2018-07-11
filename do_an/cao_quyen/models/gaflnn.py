import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from ga_flnn.expand_data import ExpandData
from ga_flnn.population import Population


class GAFLNN(BaseEstimator):
    def __init__(self, sliding=2, 
                expand_func=0, 
                pop_size=200,
                pc=0.2,
                pm=0.5,
                activation='relu'):
        # self.data_original = data_original
        # self.train_idx = train_idx
        # self.test_idx = test_idx
        self.sliding = sliding
        # self.data = data_original[:train_idx + test_idx + sliding + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.activation = activation
        # self.path_save_result = path_save_result
        # self.test_name = test_name
        # self.filename = "GA-FLNN-sliding_{0}-expand_func_{1}-pop_size_{2}-pc_{3}-pm_{4}-activation_{5}".format(sliding,
        #                                                                                                        expand_func,
        #                                                                                                        pop_size,
        #                                                                                                        pc, pm,
        #                                                                                                        activation)
    def get_params(self, deep=True):
        return {
            "sliding":self.sliding,
            "expand_func": self.expand_func,
            "pop_size": self.pop_size,
            "pc":self.pc,
            "pm":self.pm,
            "activation": self.activation
        }
    def set_params(self, **params):
        for param,value in params.items():
            self.__setattr__(param,value)
        return self
    def predict(self, X):
        return
    def score(self, X, y):
        return
    def fit(self, X,y=None,**params):
        return self