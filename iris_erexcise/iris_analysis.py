import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import xgboost 
import sklearn

df = pd.read_csv("iris_dataset.csv")
data = df

# data preview
setosa_mask = data['Name']=='Iris-setosa'
setosa_df = data[setosa_mask]
#print(setosa_df.head())

versicolor_mask = data['Name']=='Iris-versicolor'
versicolor_df = data[versicolor_mask]
#print(versicolor_df.head())

virginica_mask = data['Name']=='Iris-virginica'
virginica_df = data[virginica_mask]
#print(virginica_df.head())

plot.scatter(setosa_df['SepalLength'], setosa_df['PetalWidth'], marker='o', label='0-setosa')
plot.scatter(versicolor_df['SepalLength'], versicolor_df['PetalWidth'], marker='x', label='1-versicolor')
plot.scatter(virginica_df['SepalLength'], virginica_df['PetalWidth'], marker='*', label='2-virginica')

plot.legend()
plot.show()

# data clean

#print(df[['Name']]['Name'].unique())
# https://towardsdatascience.com/how-to-encode-categorical-columns-using-python-9af10b36f049
data['Cat'] = data['Name'].astype('category').cat.codes

data.insert(0, 'X0', 1)

X = data.iloc[:, [0,4]]
print(X)
X = np.array(X.values)

y = data.iloc[:, [6]]
# map setosa = 0, else to 1
y_setosa = np.array(y.replace(2, 1).values)
print(y_setosa)
initial_theta = np.zeros(X.shape[1])

# train the model

import costFunction as costFunction
import scipy.optimize as opt
theta,_,_ = opt.fmin_tnc(func=costFunction.costFunction, x0=initial_theta, fprime=costFunction.gradient, args=(X, y_setosa))
print('Cost at optimzed theta found by TNC: %f\n', costFunction.costFunction(theta, X, y))
print('theta: ', theta)

# test and validate
from sklearn.metrics import classification_report
import predict

#prob1 = sigmoid.sigmoid(np.matrix([1, 0.2]) * np.matrix(theta).T)
prob1 = predict.predict(np.matrix(theta), np.matrix([1, 0.2]))
print('We expect it is less than 0.5, ideal to be 0')
print('For a setosa data line, we predict an non-setosa probability of ', prob1)


prob2 = predict.predict(np.matrix(theta), np.matrix([1, 1.4]))
print('We expect it is larger than 0.5, ideal to be 1.0')
#prob2 = sigmoid.sigmoid(np.matrix([1, 1.4]) * np.matrix(theta).T)
print('For a versicolor data line, we predict an non-setosa probability of ', prob2)


# % Compute accuracy on our training set
y_pred = predict.predict(np.matrix(theta), np.matrix(X))
y_truth = y_setosa

#print(y_pred)
#print(y)

print(classification_report(y_setosa, y_pred))