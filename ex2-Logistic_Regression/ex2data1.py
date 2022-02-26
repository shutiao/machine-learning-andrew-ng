# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# % 
# %  We have the following functions in this exericse:
# %
# %     sigmoid.py
# %     costFunction.py
# %     predict.py
# %     costFunctionReg.py


# Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

import costFunction as costFunction


# %% Load Data
# %  The first two columns contains the exam scores and the third column contains the label.

PATH = ('ex2data1.txt')
data = pd.read_csv(PATH, names = ['Exam_1', 'Exam_2', 'Admitted'])
print(data.head())

# %% ==================== Part 1: Plotting ====================
# %  We start the exercise by first plotting the data to understand the 
# %  the problem we are working with.

pos_mask = data['Admitted']==1
pos = data[pos_mask]
print(pos.head())

neg_mask = data['Admitted']==0
neg = data[neg_mask]
print(neg.head())

plot.scatter(pos['Exam_1'], pos['Exam_2'], marker='o', label='Admitted')
plot.scatter(neg['Exam_1'], neg['Exam_2'], marker='x', label='Not Admitted')

plot.legend()

# hold on to show until boundrary is drawn
# plot.show()

# %% ============ Part 2: Compute Cost and Gradient ============
# %  In this part of the exercise, you will implement the cost and gradient
# %  for logistic regression. You neeed to complete the code in 
# %  costFunction.m

# % Add intercept term to x and X_test
data.insert(0, 'Ones', 1)

# %  Setup the data matrix appropriately
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1: cols]

# % Initialize fitting parameters
X = np.array(X.values)
y = np.array(y.values)
initial_theta = np.zeros(3)
# theta: array([ 0.,  0.,  0.])
# X.shape, theta.shape, y.shape
# ((100, 3), (3,), (100, 1))

# % Compute and display initial cost and gradient
print('Cost at initial theta (zeros): ')
print(costFunction.costFunction(initial_theta, X, y))
print('Expected cost (approx): 0.693');

print('Gradient at initial theta (zeros): \n')
print(costFunction.gradient(initial_theta, X, y))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

# %% ============= Part 3: Optimizing using SciPy's truncated newton  =============

import scipy.optimize as opt
theta,_,_ = opt.fmin_tnc(func=costFunction.costFunction, x0=initial_theta, fprime=costFunction.gradient, args=(X, y))
print('Cost at optimzed theta:')
print(costFunction.costFunction(theta, X, y))

print('Cost at theta found by TNC: %f\n', costFunction.costFunction(theta, X, y))
print('Expected cost (approx): 0.203\n')
print('theta: ', theta)
print('Expected theta (approx):\n');
print(' -25.161\n 0.206\n 0.201\n');

coef = -(theta / theta[2])
x = np.arange(100)
y = coef[0] + coef[1] * x
plot.plot(x, y , label='Decision Boundary')
plot.show()