# COMPUTECOST Compute cost for linear regression
# RETRUN J = COMPUTECOST(X, y, theta) computes the cost of using theta as the 
# parameter for linea regression to fit the data points in X and y

import numpy as np

def compute_cost(X, y, theta):
    costs = np.power(((X * theta.T) - y), 2)
    return np.sum(costs) / (2 * len(X))