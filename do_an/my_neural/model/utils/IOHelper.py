#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:49:35 2018

@author: thieunv
"""

import numpy as np


def load_result(name=None):
    dat = np.load(name)
    return dat['y_pred'], dat['y_true']


# import GraphUtil
# if __name__ == "__main__":
#    y1, y2 = load_file("BPNN_2_0.046493553612.npz")
#    GraphUtil.plot_figure(y1, y2, "Point prediction")

def save_result_to_csv(y_test=None, y_pred=None, filename=None, pathsave=None):
    t1 = np.concatenate((y_test, y_pred), axis=1)
    np.savetxt(pathsave + filename + ".csv", t1, delimiter=",")


def save_loss_to_csv(loss_train=None, filename=None, pathsave=None):
    t1 = np.array(loss_train).reshape(-1, 1)
    np.savetxt(pathsave + filename + ".csv", t1, delimiter=",")


def save_model(list_clusters=None, w2=None, b2=None, system_time=None, RMSE=None, MAE=None, modelname=None,
               filesave_model=None):
    file = open(filesave_model, "a")
    file.write("------------------------------------------------------\n")
    file.write("{0}".format(modelname))
    file.write("Time cluster: {0} seconds\n".format(system_time[0]))
    file.write("Time train: {0} seconds\n".format(system_time[1]))
    file.write("Time model: {0} seconds\n".format(system_time[2]))
    file.write("List clusters: {0}\n".format(list_clusters))
    file.write("w2: {0}\n".format(w2))
    file.write("b2: {0}\n".format(b2))
    file.write("RMSE: {0}\n".format(RMSE))
    file.write("MAE: {0}\n\n".format(MAE))
    file.close()

def save_run_test(num_run_test=None, data=None, filepath=None):
    t0 = np.reshape(data, (num_run_test, -1))
    np.savetxt(filepath, t0, delimiter=",")