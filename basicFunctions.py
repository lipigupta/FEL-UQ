import numpy as np
import matplotlib.pyplot as plt
import sys, os
from IPython.display import clear_output
import h5py
import json
import tensorflow as tf
from tensorflow import keras


#found online for axis formatting
import matplotlib.ticker as mticker

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
        
        
# define class for showing training plot - found online
class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show()

plot_losses = PlotLosses()


def get_scale(data):
    # helper used to get the mean and range of the data set
    offset = np.nanmin(data,axis=0)
    scale= np.nanmax(data,axis=0) - np.nanmin(data, axis=0)
    if np.any(scale) == 0:
        scale = 1
    return offset, scale

def scale_data(data,offset, scale, lower, upper):
    # for mean 0 and std 1 data=(data-data.mean(axis=0))/data.std(axis=0)
    data_scaled=lower+((data-offset)*(upper-lower)/scale)
    return data_scaled

def unscale_data(data,offset,scale,lower,upper):
    data_unscaled=(((data-lower)*scale)/(upper-lower)) + offset
    return data_unscaled


def scaling(xdata,lower,upper, x_scales, x_offsets):  
    l,n = xdata.shape
    scaled_x = np.zeros((l,n))

    for i in range(n):
        dat = xdata[:,i]
        off = x_offsets[i]
        sc = x_scales[i]
        scaled = scale_data(dat,off,sc,lower,upper)
        scaled_x[:,i] = scaled
        
    return scaled_x

def do_scaling(xdata,lower,upper):  
    l,n = xdata.shape
    
    x_scales = []
    x_offsets = []

    scaled_x = np.zeros((l,n))

    for i in range(n):
        dat = xdata[:,i]
        off, sc = get_scale(dat)
        x_offsets.append(off)
        x_scales.append(sc)
        scaled = scale_data(dat,off,sc,lower,upper)
        scaled_x[:,i] = scaled
        
    return scaled_x, x_scales, x_offsets

def do_unscaling(data, lower, upper, scales, offsets):
    n,m = np.shape(data)
    unscaled = np.zeros((n,m))
    for i in range(m):
        dat= data[:,i]
        sc = scales[i]
        off = offsets[i]
        sc_back = unscale_data(dat,off,sc,lower,upper)
        unscaled[:,i] = sc_back
    return unscaled
    
