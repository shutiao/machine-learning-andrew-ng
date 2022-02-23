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

plot.show()

