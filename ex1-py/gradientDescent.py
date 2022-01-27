#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha
#   return (theta, J_history)

import numpy as np
import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):

    # Initialize some useful values
    update_theta = np.matrix(np.zeros(theta.shape))
    m = y.shape[0]; # number of training examples
    J_history = np.zeros(num_iters)
    num_params = int(theta.ravel().shape[1])

    for i in range(num_iters):

        error = (X * theta.T) - y

        for j in range(num_params):
            term = np.multiply(error, X[:, j])
            update_theta[0, j] = theta[0, j] - ((alpha / m) * np.sum(term))
            #print('theta', j, 'update from', theta[0, j], 'to', update_theta[0, j])
        
        # update theta simultaneously 
        theta = update_theta

        # Save the cost J in every iteration    
        cost = computeCost.compute_cost(X, y, theta)
        #print('cost update to', cost)
        J_history[i] = cost

    print('final theta is', theta, 'after', num_iters, 'iteration')
    print('cost is reduced to', J_history[-1], 'from', J_history[0])
    return theta, J_history