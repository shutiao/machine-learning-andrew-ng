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
import predict
import sigmoid

from sklearn.metrics import classification_report

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
x_ax = np.arange(100)
y_ax = coef[0] + coef[1] * x_ax
plot.plot(x_ax, y_ax , label='Decision Boundary')
plot.show()

# %% ============== Part 4: Predict and Accuracies ==============
# %  After learning the parameters, you will like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and 
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of 
# %  our model.
# %
# %  Your task is to complete the code in predict.m

# %  Predict probability for a student with score 45 on exam 1 
# %  and score 85 on exam 2 

prob = sigmoid.sigmoid(np.matrix([1, 45, 85]) * np.matrix(theta).T)
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# % Compute accuracy on our training set
y_pred = predict.predict(np.matrix(theta), np.matrix(X))
print(y_pred)
print(y)
# print('Train Accuracy: %f\n', mean(double(p == y)) * 100)
print('Expected accuracy (approx): 89.0\n')

print(classification_report(y, y_pred))