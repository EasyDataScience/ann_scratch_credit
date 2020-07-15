#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 00:44:03 2020
@author: quang nguyen
Adapted from: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
- Add bias b
- Run on real dataset: Australian credit card from UCI
(http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load australian credit-card
data = pd.read_csv('data/australian.dat', delimiter='\s+', header = None, engine = 'python')
X, y = data.iloc[:, :14].values, data.iloc[:, [14]].values
#%% Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#%% Initialize
nb_inputs = 20
W1, W2 = np.random.rand(X.shape[1],nb_inputs),np.random.rand(nb_inputs,1)  
b1, b2 = np.random.rand(nb_inputs), np.random.rand(1) 
#%% Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))
def feedforward(x):
    l1 = sigmoid(np.dot(x,W1) + b1)
    l2 = sigmoid(np.dot(l1,W2) + b2)
    return l1, l2
def d_sigmoid(x):
    return x*(1-x)
def backprop(l1, l2):  # negative derivatives
    d_W2 = np.dot(l1.T, 2*(l2-y)*d_sigmoid(l2))
    d_b2 = 2*np.sum((l2-y)*d_sigmoid(l2))    
    d_W1 = np.dot(X.T, np.dot(2*(l2-y)*d_sigmoid(l2), W2.T)*d_sigmoid(l1))
    d_b1 = np.sum(np.dot(2*(l2-y)*d_sigmoid(l2), W2.T)*
                  d_sigmoid(l1),axis = 0)
    return d_W1, d_W2, d_b1, d_b2
#%% Training
avg_cost = []
for i in range(200): # 200 epochs  
    l1, l2 = feedforward(X)
    avg_cost.append(np.mean(np.square(y-l2)))
    d_W1, d_W2, d_b1, d_b2 = backprop(l1, l2)    
    W1 -= d_W1*0.1
    W2 -= d_W2*0.1  
    b1 -= d_b1*0.1
    b2 -= d_b2*0.1
#%% Plot loss
plt.plot(avg_cost)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training process')
#%% Evaluation
_, y_pred = feedforward(X)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
print('The accurary is: ', accuracy_score(y, y_pred > 0.5)) #92.8%
print('The AUC is: ', roc_auc_score(y, y_pred))  # 94.9%
