#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:49:35 2018
@author: thieunv
"""

import numpy as np
import csv

def save_result_to_csv(y_test=None, y_pred=None, filename=None, pathsave=None):
    t1 = np.concatenate((y_test, y_pred), axis=1)
    np.savetxt(pathsave + filename + ".csv", t1, delimiter=",")
    return None

def write_all_results(item=None, filename=None, pathsave=None):
    with open(pathsave + filename + ".csv", "a+") as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerow(item)

def save_run_test(num_run_test=None, data=None, filepath=None):
    t0 = np.reshape(data, (num_run_test, -1))
    np.savetxt(filepath, t0, delimiter=",")