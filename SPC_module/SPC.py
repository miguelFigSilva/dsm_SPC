import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from river import compat
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from sklearn import linear_model
from river import tree
from sklearn import datasets


class SPCAlgorithm:
    def __init__(self, init_estimator):
        self.Pmin = 1.0 # initialization
        self.Smin = 0.0 # initialization
        self.num_negative = 0
        self.num_examples = 0
        self.error_rates = [] # for plotting
        self.init_estimator = init_estimator
        self.reset_model()
        self.warn = -1

    def update(self, y):
        # Update counts
        self.num_examples += 1
        if (y == False):
            self.num_negative += 1
        # Calculate p and s
        p = self.num_negative / self.num_examples
        s = (p * (1 - p) / self.num_examples) ** 0.5
        # Update Pmin and Smin
        if p + s != 0 and p + s < self.Pmin + self.Smin:
            self.Pmin = p
            self.Smin = s
        #print(f"{p}, {s}, {self.Pmin}, {self.Smin}")
        # Check process status
        if p + s < self.Pmin + 2 * self.Smin:
            status = "In-control"
        elif p + s > self.Pmin + 3 * self.Smin:
            status = "Out-control"
        else:
            status = "Warning Level"
        self.error_rates.append(p)
        return status        

    def model_train(self, data):
        for i in range(data.shape[0]):
            try:
                x, y = data.iloc[i, :-1], data.iloc[i, -1]
            except: # single sample fitting
                x, y = data[:-1], data[-1]
        self.model.learn_one(x, y)
    
    def reset_model(self):
        self.model = self.init_estimator()
    
    def model_control(self, data, sample_id):
        x = data.iloc[sample_id, :-1]
        y = data.iloc[sample_id, -1]
        y_pred = self.model.predict_one(x)

        status = self.update(y_pred==y)        
        # check detector status
        if status == 'Warning Level' and self.warn == -1 and sample_id!=0:
            self.warn = sample_id
        elif status == 'Out-control':
            self.reset_model()
            self.model_train(data.iloc[self.warn:sample_id+1,:])
        else:
            self.model_train(data.iloc[sample_id,:])
        
        return status, y, y_pred
    
