#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:05:56 2018

@author: thieunv
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

def plot_figure(y_pred=None, y_test=None,title=None, color=['blue', 'red']):
    ax = plt.subplot()
    ax.set_color_cycle(color)
    ax.plot(y_pred,'--',label='Predict')
    ax.plot(y_test,label='Actual')
    ax.legend()
    ax.set_title(title)
    # plt.show()
    return None

def plot_figure_with_label(y_pred=None,y_test=None,title=None, x_label=None, y_label=None):
    ax = plt.subplot()
    ax.set_color_cycle(['blue','red'])
    ax.plot(y_pred,'--', label='Predict')
    ax.plot(y_test, label='Actual')
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.legend()
    return ax

def plot_metric_figure(y_pred=None,y_test=None,metric_type=None,title=None):
    for k, metric in enumerate(metric_type): 
        plot_figure(y_pred[:,k],y_test[:,k],title="%s based on %s Prediction - score %s" %(metric,title,mean_squared_error(y_pred[:, k], y_test[:, k])))
        

def draw_loss(fig_id=None, epoch=None, loss=None, title=None):
    plt.figure(fig_id)
    plt.plot(range(epoch), loss, label="Loss on training per epoch")
    plt.xlabel('Iteration', fontsize=12)  
    plt.ylabel('Loss', fontsize=12)
    plt.title(title)
    # plt.show()
    plt.close()
    return None
    
def draw_predict_with_mae(fig_id=None, y_test=None, y_pred=None, RMSE=None, MAE=None, title=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title(title)
    plt.ylabel('Real value')
    plt.xlabel('Point')
    plt.legend(['Predict y... Test Score RMSE= ' + str(RMSE) , 'Test y... Test Score MAE= '+ str(MAE)], loc='upper right')
    # plt.show()
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None

def draw_data_2d(fig_id=None, data=None, title=None):
    plt.figure(fig_id)
    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.title(title)
    plt.ylabel('Real value')
    plt.xlabel('Real value')
    # plt.show()
    plt.close()
    return None
    
def draw_dataset(fig_id=None, data=None, title=None):
    plt.figure(fig_id)
    plt.plot(data, 'ro')
    plt.title(title)
    plt.ylabel('Real value')
    plt.xlabel('Real value')
    
        
#y1 = [1.5, 2.5, 6.5, 3.5, 2.5, 8.5]
#y2 = [1.3, 2.3, 6.3, 3.3, 2.3, 8.3]

#if __name__ == "__main__":
#    plot_figure(y1, y2, "Point predict")
#    plot_figure_with_label(y1, y2, "Point prediction", "Time (10 minutes)", "CPU Usages")