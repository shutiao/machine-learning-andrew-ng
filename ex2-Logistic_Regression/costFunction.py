import numpy as np
import sigmoid as sigmoid

# Compute the cost of a particular choice of theta.
def costFunction(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid.sigmoid(X * theta.T)))
    second = np.multiply((1-y), np.log(1 - sigmoid.sigmoid(X * theta.T)))
    return np.sum(first - second) / len(X)

# Compute the step of deriative of cost function
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    size_theta = int(theta.ravel().shape[1])
    grad = np.zeros(size_theta)

    error = sigmoid.sigmoid(X * theta.T) - y

    for i in range(size_theta):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad