# COMPUTECOST Compute cost for linear regression
# RETRUN J = COMPUTECOST(X, y, theta) computes the cost of using theta as the 
# parameter for linea regression to fit the data points in X and y

import numpy as np
import logging

def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    #print('inner.shape', inner.shape)
    return np.sum(inner) / (2 * len(X))