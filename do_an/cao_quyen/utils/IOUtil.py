#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:49:35 2018
@author: thieunv
"""

import numpy as np

def save_result_to_csv(y_test=None, y_pred=None, filename=None, pathsave=None):
    t1 = np.concatenate((y_test, y_pred), axis=1)
    np.savetxt(pathsave + filename + ".csv", t1, delimiter=",")
    return None


def write_to_result_file(testname=None, RMSE=None, MAE=None, filename=None, pathsave=None):
    with open(pathsave + filename + '.txt', 'a') as file:
        file.write("{0}  -  {1} - {2}\n".format(testname, MAE, RMSE))


def load_result(name=None):
    dat = np.load(name)
    return dat['y_pred'], dat['y_true']
