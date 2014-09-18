#! /usr/bin/env python 
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, August 2014. 

from Tkinter import *
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

class Common():

    def plot_graph(self, x_data, y_data, x_labels, y_labels,
                    xscale, yscale, aspect):
        """ Plot the graph about y_data for each x_data.
        """
        d = len(y_data)
        if not len(x_data) == len(y_data) == len(x_labels) == len(y_labels)\
               == len(xscale) == len(yscale) == len(aspect):
            raise ValueError("Arguments must have the same dimension.")
        if d == 0:
            raise ValueError("At least one data for plot.")
        if d > 9:
            raise ValueError("""So much data for plot in one figure.
                                Please divide two or more data sets.""")

        fig = plt.figure(figsize=(9, 8))
        subplot_positioning = ['11','21','22','22','32','32','33','33','33']
        axes = []
        for n in range(d):
            lmn = int(subplot_positioning[d-1] + str(n+1))
            axes.append(fig.add_subplot(lmn))

        for i, ax in enumerate(axes):
            ymin, ymax = min(y_data[i]), max(y_data[i])
            ax.set_aspect(aspect[i])
            ax.set_xscale(xscale[i])
            ax.set_yscale(yscale[i])
            ax.set_xlabel(x_labels[i], fontsize=16)
            ax.set_ylabel(y_labels[i], fontsize=16)
            ax.set_ymargin(0.05)
            ax.plot(x_data[i], y_data[i], 'o-')

        fig.subplots_adjust(wspace=0.2, hspace=0.5)
        fig.tight_layout()
        plt.show()
        
    def fitting(self, x, y, parameter0, fit_func):
        
        parameter = parameter0
        result = optimize.leastsq(fit_func, parameter0, args=(x, y))
        for i in range(len(parameter)):
            parameter[i] = result[0][i]
        
        return parameter


def fit(view=True):
    p_c = 0.5926 
    common = Common()
    def fit_func(parameter0, p, D):
        a = parameter0[0]
        b = parameter0[1]
        residual = np.log(D) - (a*np.log(abs(p-p_c)) + np.log(b))
        return residual
            
    def fitted(p, a, b):
        return b*abs(p-p_c)**a
    
    parameter0 = [0.5, 0.1]
    p = np.array([0.78947368, 0.81052632,  0.83157895,  0.85263158,  0.87368421,  0.89473684, 0.91578947,  0.93684211,  0.95789474,  0.97894737,  1. ])
    D = np.array([0.2153811 , 0.27223386, 0.30796717,  0.41957614,  0.48825097,  0.48406033, 0.53550554,  0.65654551, 0.88862114,  0.76989119,  1.])
    result = common.fitting(p, D, parameter0, fit_func)
    a = result[0]
    mu = a
    b = result[1]
    print a, b
    if view:
        fig = plt.figure('Diffusion coefficient')
        ax = fig.add_subplot(111)
        ax.plot(p-p_c, D, '-o',
                        label=r"$D_{s}(p)/D(p=1)$")
        ax.plot(p-p_c, fitted(p, a, b), lw=2,
                        label=r"fit func: $\mu = %f$" % mu)
        ax.set_xlabel(r'$p-p_{c}$', fontsize=16)
        ax.set_ylabel(r'$D_{s}(p)/D(p=1)$', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ymargin(0.05)
        fig.tight_layout()
        plt.legend(loc='best')
        plt.show()
    return D

fit()
