#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:05:56 2018
@author: thieunv
"""


import matplotlib.pyplot as plt

def draw_predict(fig_id=None, y_test=None, y_pred=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('CPU')
    plt.xlabel('Timestamp')
    plt.legend(['Actual', 'Prediction'], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None

def draw_predict_with_error(fig_id=None, y_test=None, y_pred=None, RMSE=None, MAE=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('Real value')
    plt.xlabel('Point')
    plt.legend(['Predict y... RMSE= ' + str(RMSE), 'Test y... MAE= ' + str(MAE)], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None


