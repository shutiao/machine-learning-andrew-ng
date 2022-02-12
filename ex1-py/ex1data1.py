# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

import logging
import plotData
import computeCost
import gradientDescent

# Machine Learning Online Class - Exercise 1: Linear Regression
""" %     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m """

""" x refers to the population size in 10,000s
% y refers to the profit in $10,000s """

print('# ======================= Part 2: Plotting =======================')

path = 'ex1data1.txt'
data = pd.read_csv(path, names = ['population', 'profit'])
#plotData.plotData(data)

print('# =================== Part 3: Cost and Gradient descent ===================')

data.insert(0, 'intercept', 1)  # Add a column of ones to x
m = data.shape[0] # number of training examples
n = data.shape[1] # number of columns
X = data.iloc[:, 0:n-1 ]
y = data.iloc[:, n-1: n]

print('check whether X 是所有行，去掉最后一列')
print(X.head())
print('check whether y 是所有行，只含最后一列')
print(y.head())

X = np.matrix(X.values) # initialize X and y
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0])) # initialize fitting parameters

print('check if matrix([[0, 0]]), theta 是一个(1,2)矩阵')
print(theta)

print('X.shape', X.shape)
print('theta.shape', theta.shape)
print('y.shape', y.shape)

print('With theta = [0,0], Cost computed is', computeCost.compute_cost(X, y, theta), 'Expected cost value (approx) 32.07')

#Some gradient descent settings
iters = 1000
alpha = 0.01

theta, J_history = gradientDescent.gradientDescent(X, y, theta, alpha, iters)


print('# ============= Part 4: Visualizing J(theta_0, theta_1) =============')

#Plot the linear fit
plotData.plotData(data, theta)

plot.plot(J_history)
plot.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta.T
print('For population = 35,000, we predict a profit of', predict1)

predict2 = [1, 7] * theta.T
print('For population = 70,000, we predict a profit of ', predict2)