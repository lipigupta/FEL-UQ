import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
import basicFunctions as bf
import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-bright')
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 25}
mpl.rc('font', **font)

### colorblind-friendly colors
colors = [[0,0,0], [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255],
          [213/255,94/255,0], [0,114/255,178/255]]

class Model():
    """
    This is my custom class for all QR model training and evaluation. Useage is demonstrated in the QR notebooks.
    Most important is the scaling dictionary.
    The model architecture is hard coded.
    
    """
    def __init__(self, scaling_dict = None, NAME = None, PATH = None, QUANTILE = 0.5):
        self.name = NAME
        self.path = PATH
        self.quantile = QUANTILE
        self.__dict__.update(scaling_dict)
        
    def __str__(self):
        s = f"""This model is {self.name} with quantile: {self.quantile}."""
        return s

    def build_model(self):
        scalar_input = keras.Input(shape = 76)
        x = layers.Dense(80, activation='tanh')(scalar_input)
        x = layers.Dense(70, activation='tanh')(x)
        x = layers.Dense(60, activation='tanh')(x)
        x = layers.Dense(50, activation='tanh')(x)
        x = layers.Dense(40, activation='tanh')(x)
        x = layers.Dense(30, activation='tanh')(x)
        x = layers.Dense(20, activation='tanh')(x)
        x = layers.Dense(10, activation='tanh')(x)
        scalar_output = layers.Dense(1, activation='linear')(x)
        return keras.Model(scalar_input, scalar_output)

    def load(self):
        self.model = self.build_model()
        self.model.load_weights(self.path + self.name)
        self.model.compile(loss = lambda y_true, y_pred: self.quantile_regression(y_true, y_pred), optmizer = 'adam')
        
    def quantile_regression(self, y_true, y_pred, sample_weight = None):
        error = tf.subtract(y_true, y_pred)
        loss = tf.reduce_mean(tf.maximum(self.quantile*error, (self.quantile-1)*error), axis = -1)
        return loss
    
    def train(self, inputs, outputs, epochs = 5000, batch_size = 4096, saving = True):
        X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size = 0.2, random_state = 42)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 42)

        if saving:
            np.save(self.path + self.name+ "x_train.npy", X_train)
            np.save(self.path + self.name+"y_train.npy", Y_train)

            np.save(self.path + self.name+ "x_test.npy", X_test)
            np.save(self.path + self.name+"y_test.npy", Y_test)

            np.save(self.path + self.name+"x_val.npy", X_val)
            np.save(self.path + self.name+"y_val.npy", Y_val)


        self.model = self.build_model()
        self.model.compile(loss = lambda y_true, y_pred: self.quantile_regression(y_true, y_pred), optmizer = 'adam')
        self.model.summary()

        mc = tf.keras.callbacks.ModelCheckpoint(self.path + self.name +'best_model_checkpoint.h5', monitor='val_loss', mode='min', save_best_only=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 500)

        start = time.time()
        self.model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = epochs, batch_size = batch_size, verbose= 0, callbacks = [mc, es, bf.plot_losses])
        stop = time.time()

        self.model.save(self.path + self.name+"Model_Save.h5")
        self.model.save_weights(self.path + self.name+"Model_Weights.h5")

        print('Time to run in minutes was: ', (stop-start)/60) 

        model_fit = self.model.predict(X_test)
        self.model_fit = bf.unscale_data(model_fit, self.yoffset, self.yscale, -1, 1)

        self.X_test_fullscale = bf.unscale_data(X_test, self.xoffsets, self.xscales, -1, 1)
        self.Y_test_fullscale = bf.unscale_data(Y_test, self.yoffset, self.yscale, -1, 1)
        print("Training Complete")
    
    def train_on_split_data(self, input_dict, output_dict, epochs = 5000, batch_size = 4096, saving = True):
        X_train = inpsc = bf.scale_data(input_dict["train"], self.xoffsets, self.xscales, -1, 1)
        X_val = bf.scale_data(input_dict["val"], self.xoffsets, self.xscales, -1, 1)
        X_test = bf.scale_data(input_dict["test"], self.xoffsets, self.xscales, -1, 1)
        Y_train = bf.scale_data(output_dict["train"], self.yoffset, self.yscale, -1,1)
        Y_val = bf.scale_data(output_dict["val"], self.yoffset, self.yscale, -1,1)
        Y_test = bf.scale_data(output_dict["test"], self.yoffset, self.yscale, -1,1)

        if saving:
            np.save(self.path + self.name+ "x_train.npy", X_train)
            np.save(self.path + self.name+"y_train.npy", Y_train)

            np.save(self.path + self.name+ "x_test.npy", X_test)
            np.save(self.path + self.name+"y_test.npy", Y_test)

            np.save(self.path + self.name+"x_val.npy", X_val)
            np.save(self.path + self.name+"y_val.npy", Y_val)


        self.model = self.build_model()
        self.model.compile(loss = lambda y_true, y_pred: self.quantile_regression(y_true, y_pred), optmizer = 'adam')
        self.model.summary()

        mc = tf.keras.callbacks.ModelCheckpoint(self.path + self.name +'best_model_checkpoint.h5', monitor='val_loss', mode='min', save_best_only=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 500)

        start = time.time()
        self.model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = epochs, batch_size = batch_size, verbose= 0, callbacks = [mc, es, bf.plot_losses])
        stop = time.time()

        self.model.save(self.path + self.name+"Model_Save.h5")
        self.model.save_weights(self.path + self.name+"Model_Weights.h5")

        print('Time to run in minutes was: ', (stop-start)/60) 

        model_fit = self.model.predict(X_test)
        self.model_fit = bf.unscale_data(model_fit, self.yoffset, self.yscale, -1, 1)

        self.X_test_fullscale = bf.unscale_data(X_test, self.xoffsets, self.xscales, -1, 1)
        self.Y_test_fullscale = bf.unscale_data(Y_test, self.yoffset, self.yscale, -1, 1)
        print("Training Complete")
        
    
    def evaluate(self, inp):
        inpsc = bf.scale_data(inp, self.xoffsets, self.xscales, -1, 1)
        pred = self.model.predict(inpsc)
        return bf.unscale_data(pred, self.yoffset, self.yscale, -1, 1)
        
        
        
##### Helper functions ######

    
def perform_quad_scan(column, inputs, outputs, med, ub, lb, plotting = True):
    """
    Temporary "quad scan."
    Isn't really a scan, just sorts on quad values and plots all of the predictions. The remaining
    inputs VARY, so this isn't a quad scan.
    """
    inds = np.argsort(inputs[:,column])
    inps = inputs[inds,:]
    meas = outputs[inds]
    median_pred = med.evaluate(inps)
    ub_pred = ub.evaluate(inps)
    lb_pred = lb.evaluate(inps)
    if plotting:
        plot_quad_scan(column, inps, ub_pred[:,0], lb_pred[:,0], median_pred[:,0], meas)
        plot_quad_scan_persample(column, inps, ub_pred[:,0], lb_pred[:,0], median_pred[:,0], meas)
    
    return inds, inps, meas, median_pred, ub_pred, lb_pred


def make_data_cuts(inputs, outputs, column, a, b, variable_names, verbose = True):
    """ Function for chopping out sections of inputs data (on the inputs, not outputs).
    -- Best if used on the scaled data meaning:
    -- a and b are scalars in [-1, 1], and a MUST BE less than b.
    """
    allinds = np.arange(0,np.shape(inputs)[0])
    xinds = np.where(np.logical_and(inputs[:,column] >  a , inputs[:,column] < b))
    remaining = np.delete(allinds, xinds[0])
    xs = np.delete(inputs, xinds[0], axis = 0)
    ys = np.delete(outputs, xinds[0])
    
    if verbose:
        plt.hist(inputs[:,column])
        print("Cutting on: " + str(variable_names[column]))
        print("Number of samples removed: {:}".format(len(xinds[0])))
        plt.hist(xs[:,column], alpha = 0.5)
        plt.title(variable_names[column])
        plt.xlabel("Variable magnitude")
    return allinds, remaining, xinds, xs, ys

def scaling_and_dict(inputs, outputs):
    """ 
    Helper function to make the scaling dictionary to pass to Model class
    """
    x_scaled, x_scales, x_offsets = bf.do_scaling(inputs, -1, 1)
    y_offset, y_scale = bf.get_scale(outputs)

    y_scaled = bf.scale_data(outputs, y_offset, y_scale, -1,1)
    d = {}
    d["xscales"] = x_scales
    d["xoffsets"] = x_offsets
    d["yoffset"] = y_offset
    d["yscale"] = y_scale
    return x_scaled, y_scaled, d

def calc_mse_and_mae(meas, pred):
    """ 
    Calculated the MSE and MAE for a 1-d array
    """
    mse = np.mean((meas - pred)**2)
    mae = np.mean(np.abs(meas-pred))
    return mse, mae

def make_cut_dict(column, a, b, allinds, remaining, xinds, xs, ys):
    """ 
    Helper to keep track of cut data, see usage in notebook.
    """
    q = {}
    q['column'] = 45
    q['a'] = a
    q['b'] = b
    q['allinds'] = allinds
    q['remaining'] = remaining
    q['xinds'] = xinds
    q['xs'] = xs
    q['ys'] = ys
    return q



######## Various plotting functions! All usage is in the notebooks.

def plot_sorted_predictions(ub_pred, lb_pred, median_pred, meas):
    ubinds = np.argsort(ub_pred[:,0])
    ub_sort = ub_pred[ubinds,:]

    lbinds = np.argsort(lb_pred[:,0])
    lb_sort = lb_pred[lbinds,0]
    
    seq = []
    step = int(ub_pred.shape[0]/500)

    for i in range(500):
        temp1=lbinds[i*step:(i+1)*step]
        temp2=ubinds[i*step:(i+1)*step]
        for num in temp1:
            if(num in temp2):
                seq.append(num)
                break

    base=np.arange(len(seq))
    markersize = 25
    plt.figure(figsize = (20, 8))
    plt.plot(ub_pred[seq,0], '.-', color = colors[2], label = "97.5% Quantile", markersize = markersize)
    plt.plot(lb_pred[seq,0], '.-', color = colors[2],  markersize = markersize, label = "2.5% Quantile")
    plt.fill_between(base, ub_pred[seq,0],lb_pred[seq,0], color=colors[2], alpha = 0.3, label= '95% Confidence Interval')
    plt.plot(meas[seq], 'x', color = colors[0], markersize = 0.7*markersize, label = "Measured Data")
    plt.plot(median_pred[seq], '.', alpha = 0.75, color = colors[1], markersize = markersize, label = "Median Prediction")
    plt.xlabel("Sample Number")
    plt.ylabel("Pulse Energy (mJ)")
    #plt.title("Quantile Regression Uncertainty Estimates and Median Prediction for FEL Pulse Energy")
    plt.legend()


    
def basic_plotting(ub_pred, lb_pred, median_pred, meas):
    plt.figure(figsize = (20, 8))
    plt.plot(ub_pred[:,0], color = colors[2], alpha = 0.5, label = "97.5% Quantile")
    plt.plot(lb_pred[:,0], color = colors[2], alpha = 0.5, label = "2.5% Quantile")
    plt.plot(meas, 'x', color = colors[0], alpha = 0.5 , label = "Measured Data")
    plt.plot(median_pred[:,0], '.', color = colors[1], label = "Median Prediction")
    plt.fill_between(np.arange(len(ub_pred[:,0])), ub_pred[:,0], lb_pred[:,0], color = colors[2], alpha = 0.5)
    plt.legend()

    
    
def plot_interpolation_predictions(ub_pred, lb_pred, median_pred, meas, outputs, remaining):
    n = len(ub_pred[:,0])
    plt.figure(figsize = (20, 8))

    plt.plot(ub_pred[:,0], color = colors[2], alpha = 0.5, label = "97.5% Quantile")
    plt.plot(lb_pred[:,0], color = colors[2], alpha = 0.5, label = "2.5% Quantile")
    plt.plot(meas, 'x', color = colors[-2], alpha = 0.75 , label = "Measured Data, Removed from Training")
    plt.plot(remaining, outputs[remaining], 'x', color = colors[3], label= "Measured Data, Available for Training")
    plt.plot(median_pred[:,0], '.', color = colors[1], label = "Median Prediction")
    plt.fill_between(np.arange(n), ub_pred[:,0], lb_pred[:,0], color = colors[2], alpha = 0.5)
    plt.xlabel("Sample Number")
    plt.ylabel("Pulse Energy (mJ)")
    plt.legend()
    #plt.show()
    
def plot_quad_scan(column, inps, ub_pred, lb_pred, median_pred,  meas):
    plt.figure(figsize = (20,6))
    plt.title("Quad Scan Performance (per sample)")
    plt.plot(inps[:,column],ub_pred, color = colors[2], label = "97.5% Quantile")
    plt.plot(inps[:,column],lb_pred, color = colors[2], label = "2.5% Quantile")
    plt.plot(inps[:,column],median_pred, '.', color = colors[1], label = "Median Prediction")
    plt.plot(inps[:,column],meas, 'x', color = colors[0], label = "Measured Data")
    plt.fill_between(inps[:,column], ub_pred, lb_pred, color = colors[2], alpha = 0.5)
    plt.legend()
    plt.show()
    
def plot_quad_scan_persample(column, inps, ub_pred, lb_pred, median_pred,  meas):
    plt.figure(figsize = (20,6))
    plt.title("Quad Scan Performance (per sample)")
    plt.plot(ub_pred, color = colors[2], label = "97.5% Quantile")
    plt.plot(lb_pred, color = colors[2], label = "2.5% Quantile")
    plt.plot(median_pred, '.', color = colors[1], label = "Median Prediction")
    plt.plot(meas, 'x', color = colors[0], label = "Measured Data")
    plt.fill_between(np.arange(len(ub_pred)), ub_pred, lb_pred, color = colors[2], alpha = 0.5)
    plt.legend()
    plt.show()
    
    
def plot_individual_points(ub_pred, lb_pred, median_pred,meas, cols = 3, rows = 5):
    fig, axs = plt.subplots(rows, cols, figsize = (int(cols*6),int(rows*6)))
    markersize = 30
    for i in range(rows):
        for j in range(cols):
            ind = np.random.randint(len(meas))
            axs[i,j].plot(median_pred[ind,0], '.', alpha = 0.75, color = colors[1], markersize = markersize, label = "Median Prediction")
            axs[i,j].plot(meas[ind], 'x', color = colors[0], markersize = 0.8*markersize, label = "Measured Data")
            axs[i,j].plot(ub_pred[ind,0],'.', color = colors[2], markersize = markersize, label = "97.5% Quantile")
            axs[i,j].plot(lb_pred[ind,0],'.', color = colors[4], markersize = markersize, label = "2.5% Quantile")
            axs[i,j].vlines(0, ub_pred[ind,0], lb_pred[ind,0], color = colors[0], alpha = 0.25)
            #axs[i,j].set_ylabel("Pulse Energy (mJ)")
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].set_xlim([-0.25, 0.25])
    
    fig.text(0.04, 0.5, 'Pulse Energy (mJ)', va='center', rotation='vertical')
    axs[0,-1].legend( loc='upper center', bbox_to_anchor=(-2, 1.25), fancybox=True, ncol=4, fontsize = 25)

    