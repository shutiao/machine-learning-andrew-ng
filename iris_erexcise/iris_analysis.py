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


# =============== distingush cat = 0 from the remaining =============== 

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
print(classification_report(y_setosa, y_pred))

# update column to data frame
data['predict'] = y_pred


print('=============== distingush cat = 1 from cat = 2 =============== ') 

#non_setosa_mask = data['Name'] != 'Iris-setosa'
non_setosa_mask = data['predict'] != 0
non_setosa_df = data[non_setosa_mask]
print(non_setosa_df)


X = non_setosa_df.iloc[:, [0,1,2,3,4]]
#print(X)
X = np.array(X.values)
#print(X)

y = non_setosa_df.iloc[:, [6]]
# map 2 to 1
y = np.array(y.replace(2, 0).values)
#print(y)
initial_theta = np.zeros(X.shape[1])

theta2,_,_ = opt.fmin_tnc(func=costFunction.costFunction, x0=initial_theta, fprime=costFunction.gradient, args=(X, y))
print('Cost at optimzed theta found by TNC: %f\n', costFunction.costFunction(theta2, X, y))
print('theta2: ', theta2)

# test and validate

prob2 = predict.predict(np.matrix(theta2), np.matrix([1, 7.0,3.2,4.7,1.4]))
print('We expect it is larger than 0.5, ideal to be 1.0')
#prob2 = sigmoid.sigmoid(np.matrix([1, 1.4]) * np.matrix(theta).T)

prob3 = predict.predict(np.matrix(theta2), np.matrix([1, 5.9,3.0,5.1,1.8]))
print('We expect it is less than 0.5, ideal to be 0')
#prob2 = sigmoid.sigmoid(np.matrix([1, 1.4]) * np.matrix(theta).T)

# % Compute accuracy on our training set
y_pred = predict.predict(np.matrix(theta2), np.matrix(X))
print(classification_report(y, y_pred))

# TODO: how to update data on the slice from a DataFrame.
# update column to data frame
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
data.loc[non_setosa_mask, 'predict'] = [2 if x == 0 else x for x in y_pred]
print(data)
print(classification_report(data['Cat'], data['predict']))