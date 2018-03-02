#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:32:10 2018

@author: himanshu
"""

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

def calculate_mse (X, y, theta):
    err = np.square (np.matmul(X, theta) - y)
    mse = np.sum (err) / (2 * X.shape[0])
    return mse

def calculate_rmse (X, y, theta):
    return math.sqrt (2 * calculate_mse (X, y, theta))

def gradient_descent (X_train, y_train, X_test, y_test, alpha):
    m = X_train.shape[0]
    n = X_train.shape[1]
    theta = np.zeros (n)
    # print ()
    error = calculate_rmse (X_test, y_test, theta)
    for i in range (num_iterations):
        delta = (np.matmul (np.matmul (X_train, theta) - y_train, X_train)) / m
        theta = theta - alpha * delta
        error = calculate_rmse(X_test, y_test, theta)
        
    return theta, error

def mean_absolute_error (X, y, theta):
    err = np.abs (np.matmul (X, theta) - y)
    mae = np.sum (err) / (2 * X.shape[0])
    return mae

def mean_cubed_error (X, y, theta):
    err = np.power (np.matmul (X, theta), 3)
    mce = np.sum (err) / (2 * X.shape[0])
    return mce

def gradient_descent_with_mae (X_train, y_train, X_test, y_test, alpha):
    m = X_train.shape[0]
    n = X_train.shape[1]
    theta = np.zeros (n)
    
    error = calculate_rmse (X_test, y_test, theta)
    for i in range (num_iterations):
        tmp = np.matmul (X_train, theta) > y_train
        oneOrMOne = []
        for t in tmp:
            if t:
                oneOrMOne.append (1)
            else:
                oneOrMOne.append (-1)
        delta = np.matmul (oneOrMOne, X_train) / (2 * m)
        theta = theta - alpha * delta
        error = calculate_rmse (X_test, y_test, theta)
    
    return theta, error

def gradient_descent_with_mce (X_train, y_train, X_test, y_test, alpha):
    m = X_train.shape[0]
    n = X_train.shape[1]
    theta = np.zeros (n)
    
    error = calculate_rmse (X_test, y_test, theta)
    for i in range (num_iterations):
        delta = (np.matmul (np.square (np.matmul (X_train, theta) - y_train), X_train)) * (3 / (2 * m))
        theta = theta - alpha * delta
        error = calculate_rmse(X_test, y_test, theta)
        
    return theta, error

# Reading data
data = pd.read_csv ('kc_house_data.csv');
test_size = 0.2

y = data['price']
X = data.drop (['price'], axis = 1)

num_examples = X.shape[0];
num_features = X.shape[1];

len_test = (int )(test_size * num_examples)
len_train = num_examples - len_test

# Convert to numpy arrays
X = X.as_matrix ()
y = y.as_matrix ()

# print (type (X), X[:10])
# print (type (y), y[:10])

# Train test split
l1 = random.sample(range(0,num_examples), len_train)
l2 =[]
for i in  range(num_examples):
    l2.append(i)
l3 = (list(set(l2) - set(l1)))

X_train = np.zeros (shape = (len_train, num_features), dtype = float, order = 'C')
X_test = np.zeros (shape = (len_test, num_features), dtype = float, order = 'C')
y_train = np.zeros (shape = len_train, dtype = float)
y_test = np.zeros (shape = len_test, dtype = float)

for i in range (len_train):
    X_train[i] = X[l1[i]]
    y_train[i] = y[l1[i]]
    
for i in range (len_test):
    X_test[i] = X[l3[i]]
    y_test[i] = y[l3[i]]

# print ('X mean: ', mean_x)
# print ('X std: ', std_x)


# print (X_train[:10])

# Feature scaling on training set
mean_x = np.mean (X_train, axis = 0)
std_x = np.std (X_train, axis = 0)

X_train = (X_train - mean_x) / std_x
X_test = (X_test - mean_x) / std_x

# Inserting column of ones at the beginning
X_train = np.concatenate((np.ones(len_train)[:, np.newaxis], X_train), axis=1)
X_test = np.concatenate((np.ones(len_test)[:, np.newaxis], X_test), axis=1)

# print ('X_train: ', X_train[:10])
# print ('X_test: ', X_test[:10])

# parameters
# theta = np.zeros (X_train.shape[1])
# learning rate
alphas = np.linspace (0.001, 0.05, 50)
num_iterations = 20000

print ('(i) mean absolute error')

rmses = []
for alpha in alphas:
    theta, err = gradient_descent_with_mae (X_train, y_train, X_test, y_test, alpha)
    rmses.append (err)

plt.plot (alphas, rmses)
plt.title ('Linear Regression with mean absolute error')
plt.xlabel ('Learning rate')
plt.ylabel ('Root mean squared error')
plt.show ()

print ('(ii) mean squared error')

rmses = []
for alpha in alphas:
    theta, err = gradient_descent (X_train, y_train, X_test, y_test, alpha)
    rmses.append (err)

plt.plot (alphas, rmses)
plt.title ('Linear Regression with mean squared error')
plt.xlabel ('Learning rate')
plt.ylabel ('Root mean squared error')
plt.show ()

print ('(iii) mean cubic error')

rmses = []
for alpha in alphas:
    theta, err = gradient_descent (X_train, y_train, X_test, y_test, alpha)
    rmses.append (err)

plt.plot (alphas, rmses)
plt.title ('Linear Regression with mean cubic error')
plt.xlabel ('Learning rate')
plt.ylabel ('Root mean squared error')
plt.show ()
