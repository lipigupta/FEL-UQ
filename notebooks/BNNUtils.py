import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
import basicFunctions as bf
import pickle



class Model(nn.Module):
    """
    This is my custom class for all BNN model training and evaluation. Useage is demonstrated in the notebooks.
    Most important is the scaling dictionary.
    The model architecture is hard coded.
    
    """
    def __init__(self, scaling_dict = None, NAME = None, PATH = None, QUANTILE = 0.5, input_dimension = 76, output_dimension = 1):
        super().__init__()
        self.name = NAME
        self.path = PATH
        self.__dict__.update(scaling_dict)
        self.input_dim = input_dimension
        self.output_dim = output_dimension
        
    def config(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.regressor = self.build_model()
        self.optimizer = optim.Adam(self.regressor(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        
    def __str__(self):
        s = f"""This model is {self.name}."""
        return s

    def build_model(self): 
        self.blinear1 = BayesianLinear(self.input_dim, 80)
        self.blinear2 = BayesianLinear(80, 60)
        self.blinear3 = BayesianLinear(60, 50)
        self.blinear4 = BayesianLinear(50, 40)
        self.blinear5 = BayesianLinear(40, 20)
        self.blinear6 = BayesianLinear(20, 20)
        self.blinear7 = BayesianLinear(20, 20)
        self.blinear8 = BayesianLinear(20, self.output_dim)
        
    def forward(self, x):
        x_=self.blinear1(x)
        x_=self.blinear2(x_)
        x_=self.blinear3(x_)
        x_=self.blinear4(x_)
        x_=self.blinear5(x_)
        x_=self.blinear6(x_)
        x_=self.blinear7(x_)
        return self.blinear8(x_)   

    def load(self):
        self.regressor = self.build_model()
        self.regressor.load_state_dict(torch.load(self.path))
        self.config()
        

    def train_on_split_data(self, input_dict, output_dict, epochs = 300, batch_size = 4096, saving = True):
        X_train = bf.scale_data(input_dict["train"], self.xoffsets, self.xscales, -1, 1)
        X_test = bf.scale_data(input_dict["test"], self.xoffsets, self.xscales, -1, 1)
        Y_train = bf.scale_data(output_dict["train"], self.yoffset, self.yscale, -1,1)
        Y_test = bf.scale_data(output_dict["test"], self.yoffset, self.yscale, -1,1)

        X_train, Y_train = torch.tensor(X_train).float(), torch.tensor(Y_train).float()
        X_test, Y_test = torch.tensor(X_test).float(), torch.tensor(Y_test).float()
        
        
        ds_train = torch.utils.data.TensorDataset(X_train, Y_train)
        self.dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=2048, shuffle=True)

        ds_test = torch.utils.data.TensorDataset(X_test, Y_test)
        self.dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=2048, shuffle=True)

        
        if saving:
            np.save(self.path + self.name+ "x_train.npy", X_train)
            np.save(self.path + self.name+"y_train.npy", Y_train)

            np.save(self.path + self.name+ "x_test.npy", X_test)
            np.save(self.path + self.name+"y_test.npy", Y_test)



        self.regressor = self.build_model()
        self.regressor.to(self.device)
        self.config()
        
        start = time.time()
        
        
        iteration = 0
        for epoch in range(epochs):
            for i, (datapoints, labels) in enumerate(self.dataloader_train):
                self.optimizer.zero_grad()

                loss = self.regressor.sample_elbo(inputs = datapoints.to(self.device),
                                   labels = labels.to(self.device),
                                   criterion = self.criterion,
                                   sample_nbr = 3,
                                   complexity_cost_weight = 1/X_train.shape[0])
                loss.backward()
                self.optimizer.step()

                iteration += 1
                if iteration%100==0:
                    ic_acc, under_ci_upper, over_ci_lower = self.evaluate_regression(self.regressor,
                                                                                X_test.to(self.device),
                                                                                Y_test.to(self.device),
                                                                                samples=100,
                                                                                std_multiplier=2)

                    print("CI acc: {:.4f}, CI upper acc: {:.4f}, CI lower acc: {:.4f}".format(ic_acc, under_ci_upper, over_ci_lower))

        stop = time.time()


        print('Time to run in minutes was: ', (stop-start)/60) 
        torch.save(self.regressor.state_dict(), self.path + self.name+"Model_Save.h5")
        print("Training Complete")
        
    
    def evaluate(self, inp):
        
        inp = bf.scale_data(inp, self.xoffsets, self.xscales, -1, 1)
        inp = torch.tensor(inp).float()
        temp = self.regressor.forward(inp)
        pred = temp.detach().numpy()
        
        return bf.unscale_data(pred[:,np.newaxis], self.yoffset, self.yscale, -1, 1)
        
