import sigmoid

def predict(theta, X):
# %PREDICT Predict whether the label is 0 or 1 using learned logistic 
# %regression parameters theta
# %   p = PREDICT(theta, X) computes the predictions for X using a 
# %   threshold at 0.5 (i.e., if sigmoid(theta*x) >= 0.5, predict 1)

    prob = sigmoid.sigmoid(X * theta.T)
    #print(prob)
    return [1 if x >= 0.5 else 0 for x in prob]