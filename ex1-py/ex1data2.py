# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

import logging

import featureNormalize
import plotData
import computeCost
import gradientDescent
import normalEqn


'''
%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
'''

## Initialization

# ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

# Load Data
PATH = 'ex1data2.txt'
data = pd.read_csv(PATH, names = ['Size', 'Bedroom', 'Price'])

#plotData.plotData(data)

# Print out some data points
print('First 10 examples from the dataset: \n')
print(data.head())

# Scale features and set them to zero mean
data_norm, _ , _ = featureNormalize.featureNormalize(data)
print(data_norm.head())

# ================ Part 2: Gradient Descent ================

# Add intercept term to X_norm
data_norm.insert(0, 'X0', 1) 
X = data_norm.iloc[:, 0:-1]
X = np.matrix(X.values)

y = data_norm.iloc[:, -1:] # Select last column of dataframe as a dataframe object
y = np.matrix(y.values)

# Choose some alpha value
alpha = 0.1
num_iters = 100

# Init Theta and Run Gradient Descent 
theta = np.matrix(np.zeros(X.shape[1])) # [0,0,0]

print('Running gradient descent ...\n');
theta, J_history = gradientDescent.gradientDescent(X, y, theta, alpha, num_iters)

'''
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;
'''


# %% ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# % Calculate the parameters from the normal equation
theta = normalEqn.normalEqn(X, y);

# % Display normal equation's result
print('Theta computed from the normal equations: \n');
print('\n', theta)