#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:49:35 2018

@author: thieunv
"""

import numpy as np
def load_result(name=None):
    dat = np.load(name)
    return dat['y_pred'],dat['y_true']

#import GraphUtil
#if __name__ == "__main__":
#    y1, y2 = load_file("BPNN_2_0.046493553612.npz")
#    GraphUtil.plot_figure(y1, y2, "Point prediction")

def save_result_to_csv(y_test, y_pred, filename=None, pathsave=None):
    t1 = np.concatenate( (y_test, y_pred), axis = 1)
    np.savetxt(pathsave + filename + ".csv", t1, delimiter=",")
    

    

